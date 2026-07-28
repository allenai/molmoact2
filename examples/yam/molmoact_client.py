"""MolmoAct2-BimanualYAM policy clients for the eval launcher.

Two interchangeable policies, both producing an ``{"actions": ndarray}`` dict
from an observation:

* :class:`MolmoAct` — HTTP client that POSTs observations to a running
  ``host_server_yam.py`` (see the sibling file) using the ``json_numpy`` wire
  protocol. This is the lightweight default.
* :class:`MolmoActLocal` — loads the checkpoint in-process via ``transformers``
  (no server). The local-policy internals are kept in sync with
  ``host_server_yam.py``; if the bf16 patch needles or the ``predict_action``
  signature change there, mirror the change here.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from abc import ABC
from pathlib import Path
from typing import Any, Dict, List, Optional

import hf_transfer  # noqa: F401
import json_numpy
import numpy as np
import requests
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from gello_min.logging_utils import get_molmoact_logger
from rollout_manifest import RolloutSeedPlan, SEED_STRATEGY

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


class PolicyBase(ABC):
    """Minimal policy interface: ``prepare_input`` -> ``inference`` -> actions."""

    def get_action_horizon(self) -> int:
        raise NotImplementedError

    def prepare_input(self, obs: Dict[str, Any], instruction: str) -> Dict[str, Any]:
        raise NotImplementedError

    def inference(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def begin_rollout(self, rollout_seed: Optional[int]) -> Dict[str, Any]:
        """Reset per-rollout policy state; remote policies have no RNG contract."""
        return {"rollout_seed": rollout_seed, "deterministic_generator": False}

    def reproducibility_metadata(self) -> Dict[str, Any]:
        return {"deterministic_generator": False}


# Default to a server running on this host (host_server_yam.py default port).
DEFAULT_SERVER = "http://127.0.0.1:8202"

REPO_ID = "allenai/MolmoAct2-BimanualYAM"
NORM_TAG = "yam_dual_molmoact2"
STATE_DIM = 14
# ``yam_dual_molmoact2`` emits 30 absolute-pose targets per query.  This is
# kept primarily for status reporting; the rollout executor validates and uses
# the actual returned chunk length.
ACTION_HORIZON = 30
DEFAULT_NUM_STEPS = 10


def require_bimanual_state(state: Any, *, source: str) -> np.ndarray:
    """Validate native checkpoint state order without inventing an arm state.

    ``MolmoAct2-BimanualYAM`` was trained on live 14-D state vectors ordered
    ``[left(7), right(7)]``.  Padding a one-arm observation with zeroes makes
    the model reason about a fictional second arm, and cropping its output
    discards the half it may actually use.  Physical callers must open both
    arms and use the launcher's active-arm hold mask instead.
    """
    result = np.asarray(state, dtype=np.float32).reshape(-1)
    if result.shape != (STATE_DIM,):
        raise ValueError(
            f"{source} requires a live bimanual state shape ({STATE_DIM},) "
            f"[left(7), right(7)], got {result.shape}. Start both arms with "
            "--right-config-path; the unsupported 7-D pad/crop adapter is disabled."
        )
    if not np.isfinite(result).all():
        raise ValueError(f"{source} state contains non-finite values")
    return result


def _normalize_server_url(server: Optional[str]) -> str:
    """Accept ngrok URLs (``https://...``), bare IPs (``10.0.0.5``), or
    ``host:port`` strings, and return a full ``http(s)://host[:port]/act`` URL.

    - If empty/None, falls back to ``DEFAULT_SERVER``.
    - If no scheme is provided, ``http://`` is prepended (suitable for LAN IPs).
    - Trailing ``/act`` is appended unless the input already ends in ``/act``.
    """
    s = (server or DEFAULT_SERVER).strip().rstrip("/")
    if "://" not in s:
        s = "http://" + s
    if not s.endswith("/act"):
        s = s + "/act"
    return s


class MolmoAct(PolicyBase):
    def __init__(self, server: Optional[str] = None):
        self.logger = get_molmoact_logger()
        self.url = _normalize_server_url(server)
        self.multi_views = True
        self.action_horizon = ACTION_HORIZON

        # Log configuration
        self.logger.info(f"MolmoAct initialized with URL: {self.url}")
        self.logger.info(f"Multi-views enabled: {self.multi_views}")
        self.logger.info(f"Action horizon: {self.action_horizon}")

    def get_action_horizon(self):
        return self.action_horizon

    def reproducibility_metadata(self) -> Dict[str, Any]:
        return {
            "backend": "http",
            "endpoint": self.url,
            "deterministic_generator": False,
            "note": "remote server does not expose a per-query generator seed",
        }

    def prepare_input(self, obs, instruction):
        self.logger.info("Preparing input for MolmoAct inference")
        self.logger.info(f"Instruction: '{instruction}'")
        # self.logger.info(f"Camera keys - {obs['left_camera_rgb']}, {obs['front_camera_rgb']}, {obs['right_camera_rgb']}")
        self.logger.info(f"State: {obs['joint_positions']}")

        try:
            # Log image information
            if hasattr(obs['left_camera_rgb'], 'shape'):
                self.logger.info(f"Left image shape: {obs['left_camera_rgb'].shape}")
            if hasattr(obs['front_camera_rgb'], 'shape'):
                self.logger.info(f"Front image shape: {obs['front_camera_rgb'].shape}")
            if hasattr(obs['right_camera_rgb'], 'shape'):
                self.logger.info(f"Right image shape: {obs['right_camera_rgb'].shape}")

            input_dict = {
                "left_camera_rgb": obs["left_camera_rgb"],
                "front_camera_rgb": obs["front_camera_rgb"],
                "right_camera_rgb": obs["right_camera_rgb"],
                "instruction": instruction,
                "state": require_bimanual_state(
                    obs["joint_positions"], source="MolmoAct"
                ),
            }

            self.logger.info("Input preparation completed successfully")
            return input_dict

        except KeyError as e:
            self.logger.error(f"Missing camera key in observation: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error preparing input: {e}")
            raise

    def inference(self, input_dict):
        self.logger.info("Starting MolmoAct inference")

        try:
            images = [input_dict["left_camera_rgb"], input_dict["front_camera_rgb"], input_dict["right_camera_rgb"]]
            lang = input_dict["instruction"]
            state = input_dict["state"]

            self.logger.info(f"Processing instruction: '{lang}'")
            self.logger.info(f"Number of images: {len(images)}")
            self.logger.info(f"Number of joints: {len(state)}")

            start_time = time.time()
            response = self.send_request(images, lang, state, self.url)
            request_time = time.time() - start_time

            self.logger.info(f"Server request completed in {request_time:.3f}s")
            self.logger.info(f"Raw actions received: {len(response['actions'])} actions")

            # processed_actions = self.prepare_output(actions)
            # self.logger.info(f"Processed {len(processed_actions)} actions")

            return response

        except Exception as e:
            self.logger.error(f"Error during inference: {e}")
            raise

    # def prepare_output(self, raw_actions):
    #     self.logger.info("Preparing output actions")

    #     try:
    #         result_actions = raw_actions.copy()

    #         if self.invert_gripper:
    #             self.logger.info("Applying gripper inversion")
    #             for i in range(len(raw_actions)):
    #                 action = raw_actions[i]
    #                 result_actions[i] = invert_gripper(action)
    #                 self.logger.debug(f"Action {i}: gripper value inverted")
    #         else:
    #             self.logger.info("No gripper inversion applied")

    #         self.logger.info(f"Output preparation completed: {len(result_actions)} actions")
    #         return result_actions

    #     except Exception as e:
    #         self.logger.error(f"Error preparing output: {e}")
    #         raise

    def send_request(self, images: List[np.ndarray], instruction: str, state: list, server_url: str):
        """
        Send the captured image and instruction to the inference server using json_numpy.
        Returns the action output as received from the server.
        """
        self.logger.info(f"Sending request to server: {server_url}")

        try:
            if not self.multi_views:
                self.logger.info("Using single view mode")
                # Convert PIL image to a NumPy array
                image_np = np.array(images[0])
                self.logger.info(f"Single image shape: {image_np.shape}")

                # Prepare the payload with the image and instruction from the script
                payload = {
                    "image": image_np, # scene cam
                    "instruction": instruction,
                    "state" : state
                }
            else:
                self.logger.info("Using multi-view mode")
                # Convert PIL image to a NumPy array
                left_img_np = np.array(images[0])
                front_img_np = np.array(images[1])
                right_img_np = np.array(images[2])

                self.logger.info(f"Left image shape: {left_img_np.shape}")
                self.logger.info(f"Front image shape: {front_img_np.shape}")
                self.logger.info(f"Right image shape: {right_img_np.shape}")

                # Prepare the payload with the image and instruction from the script
                payload = {
                    "left_cam": left_img_np,
                    "top_cam": front_img_np,
                    "right_cam": right_img_np,
                    "timestamp": time.time(), # add timestamp for debugging
                    "instruction": instruction,
                    "state": state,
                    "normalization_tag": "yam_dual_molmoact2"
                }

            self.logger.info("Preparing HTTP request")
            headers = {"Content-Type": "application/json"}

            # Serialize payload
            start_time = time.time()
            serialized_payload = json_numpy.dumps(payload)
            serialize_time = time.time() - start_time
            self.logger.info(f"Payload serialized in {serialize_time:.3f}s")

            # Send request
            start_time = time.time()
            response = requests.post(server_url, headers=headers, data=serialized_payload)
            request_time = time.time() - start_time

            self.logger.info(f"HTTP request completed in {request_time:.3f}s")
            self.logger.info(f"Response status code: {response.status_code}")

            if response.status_code != 200:
                error_msg = f"Server error: {response.text}"
                self.logger.error(error_msg)
                raise Exception(error_msg)

            # Parse response. Use json_numpy.loads explicitly so ndarrays
            # round-trip, instead of relying on a global json.loads patch.
            start_time = time.time()
            response_data = json_numpy.loads(response.text)
            parse_time = time.time() - start_time
            self.logger.info(f"Response parsed in {parse_time:.3f}s")

            self.logger.info("Server request completed successfully")
            return response_data

        except requests.exceptions.ConnectionError as e:
            self.logger.error(f"Connection error to server {server_url}: {e}")
            raise
        except requests.exceptions.Timeout as e:
            self.logger.error(f"Request timeout to server {server_url}: {e}")
            raise
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request error to server {server_url}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during server request: {e}")
            raise


# ---------------------------------------------------------------------------
# Local in-process MolmoAct (no HTTP server)
# ---------------------------------------------------------------------------
#
# The `_LocalPolicy`, `_patch_modeling_for_bf16`, and `_to_pil` definitions
# below are inlined from the sibling `host_server_yam.py` with the FastAPI
# server bits stripped. Keep this block in sync with that file if the patch
# needles or the `predict_action` signature change.

_local_log = logging.getLogger("molmoact.local")


def _patch_modeling_for_bf16(local_dir: str) -> None:
    """Idempotently rewrite the cached ``modeling_molmoact2.py`` so bf16 works.

    Most current YAM snapshots already ship with these fixes applied
    upstream — in that case both patches will warn "needle not found" and
    bf16 inference works regardless. Keep the function around so a redownload
    of an older snapshot still loads cleanly.
    """
    patches = [
        (
            "device=device,\n            dtype=torch.float32,\n            generator=generator,",
            "device=device,\n"
            "            dtype=source_tensor.dtype,  # patched_bf16_dtype\n"
            "            generator=generator,",
            "patched_bf16_dtype",
        ),
        (
            "return value.detach().cpu().numpy().astype(np.float32, copy=False)",
            "return value.detach().cpu().float().numpy().astype(np.float32, copy=False)  # patched_bf16_to_array",
            "patched_bf16_to_array",
        ),
    ]
    candidates = [os.path.join(local_dir, "modeling_molmoact2.py")]
    modules_root = os.path.expanduser(
        "~/.cache/huggingface/modules/transformers_modules"
    )
    if os.path.isdir(modules_root):
        for sub in os.listdir(modules_root):
            p = os.path.join(modules_root, sub, "modeling_molmoact2.py")
            if os.path.isfile(p):
                candidates.append(p)
    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as f:
                src = f.read()
        except OSError:
            continue
        new_src = src
        applied: List[str] = []
        for needle, replacement, marker in patches:
            if marker in new_src:
                continue
            if needle not in new_src:
                _local_log.warning("patch %s: needle not found in %s", marker, path)
                continue
            new_src = new_src.replace(needle, replacement, 1)
            applied.append(marker)
        if new_src != src:
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_src)
            _local_log.info("Applied patches %s in %s", applied, path)


def _to_pil(arr: Any):
    if isinstance(arr, Image.Image):
        return arr.convert("RGB")
    a = np.asarray(arr)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"image must be HxWx3, got shape {a.shape}")
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    return Image.fromarray(a, mode="RGB")


class _LocalPolicy:
    """Holds the loaded MolmoAct model + processor; serializes inference calls."""

    def __init__(
        self,
        repo_id: str,
        device: str,
        dtype: torch.dtype,
        enable_cuda_graph: bool = False,
    ) -> None:
        self.default_cuda_graph = enable_cuda_graph
        self.repo_id = str(repo_id)

        local_dir = snapshot_download(repo_id=repo_id)
        self.snapshot_dir = Path(local_dir).resolve()
        self.snapshot_revision = self.snapshot_dir.name
        _local_log.info("Resolved snapshot dir: %s", local_dir)
        _patch_modeling_for_bf16(local_dir)

        _local_log.info("Loading processor")
        self.processor = AutoProcessor.from_pretrained(
            local_dir, trust_remote_code=True, extra_special_tokens={}
        )

        _local_log.info("Loading model (dtype=%s, device=%s)", dtype, device)
        self.model = (
            AutoModelForImageTextToText.from_pretrained(
                local_dir,
                trust_remote_code=True,
                torch_dtype=dtype,
            )
            .to(device)
            .eval()
        )
        self.device = device

        target_dtype = next(self.model.parameters()).dtype

        def _move_and_cast(
            inputs: Any, dev: Any, _target: torch.dtype = target_dtype
        ) -> dict:
            out: dict = {}
            for key, value in inputs.items():
                if torch.is_tensor(value):
                    value = value.to(dev)
                    if value.is_floating_point() and value.dtype != _target:
                        value = value.to(_target)
                out[key] = value
            return out

        self.model._move_inputs_to_device = _move_and_cast
        self._lock = threading.Lock()

    @torch.inference_mode()
    def predict(
        self,
        top_cam: Any,
        left_cam: Any,
        right_cam: Any,
        instruction: str,
        state: Any,
        num_steps: int = DEFAULT_NUM_STEPS,
        enable_cuda_graph: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> np.ndarray:
        images = [_to_pil(top_cam), _to_pil(left_cam), _to_pil(right_cam)]
        state_f32 = np.asarray(state, dtype=np.float32).reshape(-1)
        if state_f32.shape != (STATE_DIM,):
            raise ValueError(
                f"state must be shape ({STATE_DIM},), got {state_f32.shape}"
            )

        with self._lock:
            out = self.model.predict_action(
                processor=self.processor,
                images=images,
                task=instruction,
                state=state_f32,
                norm_tag=NORM_TAG,
                inference_action_mode="continuous",
                enable_depth_reasoning=False,
                num_steps=num_steps,
                generator=generator,
                normalize_language=True,
                enable_cuda_graph=enable_cuda_graph,
            )
        raw = out.actions
        if torch.is_tensor(raw):
            raw = raw.detach().to(dtype=torch.float32, device="cpu").numpy()
        actions = np.asarray(raw, dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        return actions


class MolmoActLocal(PolicyBase):
    """In-process MolmoAct policy. Drop-in replacement for :class:`MolmoAct` (HTTP)."""

    def __init__(
        self,
        repo_id: str = REPO_ID,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        num_steps: int = DEFAULT_NUM_STEPS,
        enable_cuda_graph: bool = False,
        warmup: bool = True,
        single_arm_side: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.logger = get_molmoact_logger()
        self.multi_views = True
        self.action_horizon = ACTION_HORIZON
        self.num_steps = int(num_steps)
        self.enable_cuda_graph = bool(enable_cuda_graph)
        self.dtype_name = str(dtype)
        self._configured_seed_plan = RolloutSeedPlan(seed)
        self._query_seed_plan = RolloutSeedPlan(None)
        self._rollout_seed = self._configured_seed_plan.rollout_seed(0)
        self._inference_index = 0
        # Retain this keyword temporarily so legacy YAML fails at the safer
        # 14-D state gate rather than at config parsing. It is intentionally
        # not used to pad/crop a BimanualYAM policy action.
        if single_arm_side is not None:
            side = str(single_arm_side).lower()
            if side not in ("left", "right"):
                raise ValueError("single_arm_side must be 'left' or 'right'")
            self.logger.warning(
                "single_arm_side=%s is ignored: MolmoAct2-BimanualYAM now "
                "requires native 14-D feedback plus an active-arm hold mask.",
                side,
            )

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if dtype not in dtype_map:
            raise ValueError(
                f"dtype must be one of {sorted(dtype_map)}, got {dtype!r}"
            )

        self.logger.info(
            f"MolmoActLocal: loading {repo_id} (device={device}, dtype={dtype})"
        )
        self.policy = _LocalPolicy(
            repo_id=repo_id,
            device=device,
            dtype=dtype_map[dtype],
            enable_cuda_graph=self.enable_cuda_graph,
        )
        self.logger.info(
            f"MolmoActLocal ready. action_horizon={self.action_horizon}, "
            f"num_steps={self.num_steps}, enable_cuda_graph={self.enable_cuda_graph}, "
            f"generator_seed={'configured' if self._rollout_seed is not None else 'unseeded'}"
        )

        if warmup:
            self._warmup()

    def _warmup(self) -> None:
        self.logger.info("MolmoActLocal warmup ...")
        t0 = time.time()
        dummy = np.zeros((180, 320, 3), dtype=np.uint8)
        try:
            self.policy.predict(
                top_cam=dummy,
                left_cam=dummy,
                right_cam=dummy,
                instruction="warmup",
                state=np.zeros(STATE_DIM, dtype=np.float32),
                num_steps=self.num_steps,
                enable_cuda_graph=self.enable_cuda_graph,
            )
        except Exception:
            self.logger.exception("MolmoActLocal warmup failed (continuing)")
            return
        self.logger.info(f"MolmoActLocal warmup OK ({time.time() - t0:.1f}s)")

    def get_action_horizon(self) -> int:
        return self.action_horizon

    def begin_rollout(self, rollout_seed: Optional[int]) -> Dict[str, Any]:
        """Start an isolated deterministic random stream for this rollout."""
        self._rollout_seed = RolloutSeedPlan(rollout_seed).base_seed
        self._inference_index = 0
        return {
            "rollout_seed": self._rollout_seed,
            "query_seed_strategy": SEED_STRATEGY,
            "generator_device": self.policy.device,
            "deterministic_generator": self._rollout_seed is not None,
        }

    def reproducibility_metadata(self) -> Dict[str, Any]:
        return {
            "backend": "local",
            "repo_id": self.policy.repo_id,
            "snapshot_dir": str(self.policy.snapshot_dir),
            "snapshot_revision": self.policy.snapshot_revision,
            "device": self.policy.device,
            "dtype": self.dtype_name,
            "num_steps": self.num_steps,
            "enable_cuda_graph": self.enable_cuda_graph,
            "configured_seed": self._configured_seed_plan.base_seed,
            "query_seed_strategy": SEED_STRATEGY,
        }

    def _generator_for_next_query(self) -> tuple[Optional[torch.Generator], Dict[str, Any]]:
        query_index = self._inference_index
        query_seed = self._query_seed_plan.query_seed(self._rollout_seed, query_index)
        metadata: Dict[str, Any] = {
            "rollout_seed": self._rollout_seed,
            "query_index": query_index,
            "query_seed": query_seed,
            "seed_strategy": SEED_STRATEGY,
        }
        if query_seed is None:
            return None, metadata
        generator = torch.Generator(device=self.policy.device)
        generator.manual_seed(query_seed)
        return generator, metadata

    def prepare_input(self, obs: dict, instruction: str) -> dict:
        self.logger.info(
            f"Preparing input for MolmoActLocal (instruction: '{instruction}')"
        )
        try:
            state = require_bimanual_state(
                obs["joint_positions"], source="MolmoActLocal"
            )
            return {
                "left_camera_rgb": obs["left_camera_rgb"],
                "front_camera_rgb": obs["front_camera_rgb"],
                "right_camera_rgb": obs["right_camera_rgb"],
                "instruction": instruction,
                "state": state,
            }
        except KeyError as e:
            self.logger.error(f"Missing key in observation: {e}")
            raise

    def inference(self, input_dict: dict) -> dict:
        t0 = time.time()
        generator, reproducibility = self._generator_for_next_query()
        actions = self.policy.predict(
            top_cam=input_dict["front_camera_rgb"],
            left_cam=input_dict["left_camera_rgb"],
            right_cam=input_dict["right_camera_rgb"],
            instruction=input_dict["instruction"],
            state=input_dict["state"],
            num_steps=self.num_steps,
            enable_cuda_graph=self.enable_cuda_graph,
            generator=generator,
        )
        # Advance only after a successful query so an operator retrying a
        # transient model failure receives the same sampled action plan.
        self._inference_index += 1
        self.logger.info(
            f"Local inference completed in {time.time() - t0:.3f}s "
            f"({len(actions)} actions)"
        )
        return {"actions": actions, "reproducibility": reproducibility}
