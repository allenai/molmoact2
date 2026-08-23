#!/usr/bin/env python3
"""
Serve MolmoAct2 inference over HTTP.

This server is intentionally thin: it only
1. parses request payloads into observation dicts,
2. calls the LeRobot MolmoAct2 policy wrapper, and
3. converts the resulting action chunk into an HTTP response.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
import sys
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json_numpy
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
LEROBOT_SRC = REPO_ROOT / "lerobot" / "src"
for candidate in (REPO_ROOT, LEROBOT_SRC):
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from lerobot.policies.molmoact2.configuration_molmoact2 import (  # noqa: E402
    MolmoAct2Config,
)
from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy  # noqa: E402

json_numpy.patch()

log = logging.getLogger(__name__)

# IMAGE_KEY_PRESETS = {
#     "yam": ["top_cam", "left_cam", "right_cam"],
#     "droid": ["external_cam", "wrist_cam"],
# }
IMAGE_KEY_PRESETS = {
    "yam": ["top_cam", "left_cam", "right_cam"],
    "yamc": ["right_cam", "left_cam", "top_cam"],
    "droid": ["external_cam", "wrist_cam"],
    "libero": ["image", "wrist_image"],
    "panda": ["external_cam", "left_wrist_cam",  "right_wrist_cam"],
    "xarm": [
        "zed_gripper_left",
        "zed_high_left_left"
    ],
}


def _require_discrete_action_tokenizer(
    inference_action_mode: str,
    discrete_action_tokenizer: Optional[str],
) -> Optional[str]:
    normalized = str(discrete_action_tokenizer or "").strip() or None
    if str(inference_action_mode).strip().lower() == "discrete" and normalized is None:
        raise ValueError(
            "--discrete_action_tokenizer is required when "
            "--inference_action_mode is set to 'discrete'."
        )
    return normalized


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def _decode_image(payload_item: Any) -> np.ndarray:
    if isinstance(payload_item, Image.Image):
        return np.array(payload_item.convert("RGB"))
    if torch.is_tensor(payload_item):
        payload_item = payload_item.detach().cpu().numpy()
    if isinstance(payload_item, np.ndarray):
        arr = payload_item
    elif isinstance(payload_item, (list, tuple)):
        arr = np.asarray(payload_item)
    elif isinstance(payload_item, bytes):
        arr = np.array(Image.open(io.BytesIO(payload_item)).convert("RGB"))
    elif isinstance(payload_item, str):
        text = payload_item.strip()
        if os.path.exists(text):
            arr = np.array(Image.open(text).convert("RGB"))
        else:
            if text.startswith("data:image"):
                text = text.split(",", 1)[-1]
            raw = base64.b64decode(text)
            arr = np.array(Image.open(io.BytesIO(raw)).convert("RGB"))
    else:
        raise ValueError(f"Unsupported image payload type: {type(payload_item)}")

    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim == 3 and arr.shape[0] in {1, 3, 4} and arr.shape[-1] not in {1, 3, 4}:
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.dtype in (np.float32, np.float64):
        if arr.size > 0 and float(arr.max()) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _parse_image_resize(value: Optional[str]) -> Optional[Tuple[int, int]]:
    if value is None:
        return None
    cleaned = value.strip().lower()
    if not cleaned:
        return None
    for sep in ("x", ","):
        if sep in cleaned:
            parts = [p.strip() for p in cleaned.split(sep) if p.strip()]
            if len(parts) != 2:
                return None
            try:
                h = int(parts[0])
                w = int(parts[1])
            except ValueError:
                return None
            return (h, w) if h > 0 and w > 0 else None
    try:
        size = int(cleaned)
    except ValueError:
        return None
    return (size, size) if size > 0 else None


def _resize_image_array(array: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    target_h, target_w = size
    if array.shape[0] == target_h and array.shape[1] == target_w:
        return array
    pil_image = Image.fromarray(array)
    resized = pil_image.resize((target_w, target_h), Image.BILINEAR)
    return np.asarray(resized)


def _resize_images(images: List[np.ndarray], size: Tuple[int, int]) -> List[np.ndarray]:
    return [_resize_image_array(image, size) for image in images]


def _save_image_array(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(path)


def _sanitize_filename(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in str(text))
    return cleaned.strip("._") or "value"


def _parse_state_payload(state: Any, n_obs_steps: int) -> np.ndarray:
    if torch.is_tensor(state):
        state = state.detach().cpu().numpy()
    arr = np.asarray(state, dtype=np.float32)
    if arr.ndim == 1:
        if arr.size % n_obs_steps != 0:
            raise ValueError(f"State length {arr.size} is not divisible by n_obs_steps={n_obs_steps}")
        arr = arr.reshape(n_obs_steps, -1)
    elif arr.ndim == 2:
        if arr.shape[0] != n_obs_steps:
            raise ValueError(f"State has {arr.shape[0]} obs steps, expected {n_obs_steps}")
    else:
        raise ValueError(f"Unsupported state shape {arr.shape}")
    return arr


def _build_policy(
    checkpoint: str,
    *,
    use_hf_ckpt: bool,
    device: Optional[str],
    seq_len: Optional[int],
    num_steps: Optional[int],
    inference_action_mode: str,
    discrete_action_tokenizer: Optional[str],
    enable_depth_reasoning: bool,
    norm_tag: str,
    enable_inference_cuda_graph: bool,
    verbose: bool,
) -> Any:
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = MolmoAct2Config(
        checkpoint_path=str(Path(checkpoint).expanduser()),
        device=resolved_device,
        seq_len=seq_len,
        num_steps=num_steps,
        inference_action_mode=inference_action_mode,
        discrete_action_tokenizer=discrete_action_tokenizer,
        enable_depth_reasoning=bool(enable_depth_reasoning),
        norm_tag=str(norm_tag or ""),
        enable_inference_cuda_graph=bool(enable_inference_cuda_graph),
        verbose=bool(verbose),
    )
    policy = MolmoAct2Policy(cfg)
    policy.eval()
    return policy


def _build_observations(
    images: List[np.ndarray],
    image_names: List[str],
    state: np.ndarray,
    instruction: str,
) -> List[Dict[str, Any]]:
    observations: List[Dict[str, Any]] = []
    for obs_idx in range(int(state.shape[0])):
        obs: Dict[str, Any] = {
            "task": str(instruction or ""),
            "observation.state": np.asarray(state[obs_idx], dtype=np.float32),
        }
        for image_name, image in zip(image_names, images):
            obs[f"observation.images.{image_name}"] = np.asarray(image)
        observations.append(obs)
    return observations


@dataclass
class PolicySessionState:
    obs_history: Dict[int, deque]
    action_queues: Dict[int, deque]
    depth_caches: Dict[int, Any]


def _clone_state_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, deque):
        return deque((_clone_state_value(item) for item in value), maxlen=value.maxlen)
    if isinstance(value, defaultdict):
        return defaultdict(lambda: deque(), {key: _clone_state_value(item) for key, item in value.items()})
    if isinstance(value, dict):
        return {key: _clone_state_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_state_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_state_value(item) for item in value)
    return value


class MolmoAct2Server:
    def __init__(
        self,
        checkpoint: str,
        use_hf_ckpt: bool = False,
        device: Optional[str] = None,
        seq_len: Optional[int] = None,
        num_steps: Optional[int] = None,
        n_action_steps: Optional[int] = None,
        image_resize: Optional[str] = None,
        image_keys: str = "yam",
        inference_action_mode: str = "continuous",
        discrete_action_tokenizer: Optional[str] = None,
        enable_depth_reasoning: bool = False,
        norm_tag: str = "",
        enable_inference_cuda_graph: bool = True,
        save_image_dir: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        self.policy = _build_policy(
            checkpoint,
            use_hf_ckpt=use_hf_ckpt,
            device=device,
            seq_len=seq_len,
            num_steps=num_steps,
            inference_action_mode=inference_action_mode,
            discrete_action_tokenizer=discrete_action_tokenizer,
            enable_depth_reasoning=enable_depth_reasoning,
            norm_tag=norm_tag,
            enable_inference_cuda_graph=enable_inference_cuda_graph,
            verbose=verbose,
        )
        hf_backend = getattr(self.policy, "_hf_backend", None)
        self.use_hf_ckpt = hf_backend is not None
        self.verbose = bool(verbose)
        if self.use_hf_ckpt:
            self.device = torch.device(self.policy.config.device or "cpu")
            self.inference_action_mode = str(self.policy.config.inference_action_mode)
            self.n_obs_steps = 1
            self.checkpoint_n_action_steps = None
            self.default_seq_len = self.policy.config.seq_len
            self.default_num_steps = self.policy.config.num_steps
            self.default_style = "robot_depth_action" if enable_depth_reasoning else "robot_action"
            self.default_norm_tag = str(norm_tag or self.policy.config.norm_tag or "")
            self.robot_processor = None
        else:
            handles = self.policy._handles
            if handles is None:
                raise RuntimeError("MolmoAct2Policy did not initialize handles.")
            self.device = handles.device
            self.inference_action_mode = str(handles.inference_action_mode)
            self.n_obs_steps = int(handles.n_obs_steps)
            self.checkpoint_n_action_steps = handles.n_action_steps
            self.default_seq_len = self.policy.config.seq_len or getattr(
                self.policy._handles.model.config.llm,
                "max_sequence_length",
                None,
            )
            self.default_num_steps = handles.num_steps
            self.default_style = str(handles.default_style)
            self.default_norm_tag = str(norm_tag or handles.norm_tag)
            self.robot_processor = handles.robot_processor
        self.default_n_action_steps = (
            None if n_action_steps is None else int(n_action_steps)
        )
        if self.default_n_action_steps is None:
            self.default_n_action_steps = (
                None
                if self.checkpoint_n_action_steps is None
                else int(self.checkpoint_n_action_steps)
            )
        if self.default_n_action_steps is not None and self.default_n_action_steps < 1:
            raise ValueError(f"--n_action_steps must be >= 1, got {self.default_n_action_steps}.")
        self.image_resize = _parse_image_resize(image_resize)
        self.image_keys = list(IMAGE_KEY_PRESETS[image_keys])
        self.save_image_dir = Path(save_image_dir).expanduser() if save_image_dir else None
        if self.save_image_dir is not None:
            self.save_image_dir.mkdir(parents=True, exist_ok=True)
        self._request_index = 0
        if self.robot_processor is not None:
            self._validate_norm_tag(self.default_norm_tag)
        self._policy_lock = threading.Lock()
        self._session_states: Dict[str, PolicySessionState] = {}

        self.app = FastAPI()
        self.app.post("/act")(self.predict_action)
        self.app.post("/reset")(self.reset_session)
        self.app.get("/health")(self.health)

    def _state_owner(self) -> Any:
        return getattr(self.policy, "_hf_backend", None) or self.policy

    def _resolve_tag(self, payload: Dict[str, Any]) -> str:
        for key in ("tag", "norm_tag"):
            tag = str(payload.get(key) or "").strip()
            if tag:
                return tag
        return self.default_norm_tag

    def _validate_norm_tag(self, tag: str) -> None:
        if self.robot_processor is None:
            return
        metadata_by_tag = getattr(self.robot_processor, "metadata_by_tag", None)
        if not isinstance(metadata_by_tag, dict) or not metadata_by_tag:
            raise ValueError(
                "MolmoAct2 checkpoint has a robot processor but no configured normalization tags."
            )
        resolved_tag = self.robot_processor.resolve_tag(tag)
        if resolved_tag not in metadata_by_tag:
            allowed_tags = ", ".join(sorted(str(key) for key in metadata_by_tag))
            raise ValueError(
                f"Unknown normalization tag {tag!r}; resolved to {resolved_tag!r}, "
                f"which is not configured. Allowed tags: {allowed_tags}."
            )

    def _parse_request(
        self,
        payload: Dict[str, Any],
    ) -> Tuple[List[np.ndarray], List[str], np.ndarray, str, str, str, Dict[str, Any], Optional[int], Optional[int]]:
        if "encoded" in payload:
            payload = json.loads(payload["encoded"])

        tag = self._resolve_tag(payload)
        self._validate_norm_tag(tag)
        robot_metadata = self.robot_processor.get_metadata(tag) if self.robot_processor is not None else {}

        images_payload = payload.get("images")
        image_names: List[str] = []
        if images_payload is None:
            ordered = []
            for key in self.image_keys:
                value = payload.get(key)
                if value is not None:
                    ordered.append(value)
                    image_names.append(key)
            if ordered:
                images_payload = ordered
            else:
                single = payload.get("image") or payload.get("image1")
                if single is None:
                    raise ValueError("Missing images payload.")
                images_payload = [single]
                image_names = ["image"]
        else:
            if not isinstance(images_payload, (list, tuple)):
                images_payload = [images_payload]
            metadata_camera_keys = robot_metadata.get("camera_keys")
            if isinstance(metadata_camera_keys, list) and len(metadata_camera_keys) == len(images_payload):
                image_names = [str(key) for key in metadata_camera_keys]
            else:
                image_names = [f"image_{idx}" for idx in range(len(images_payload))]

        images = [_decode_image(img) for img in images_payload]
        if self.image_resize is not None:
            images = _resize_images(images, self.image_resize)

        state_payload = payload.get("state")
        if state_payload is None:
            state_payload = payload.get("states")
        if state_payload is None:
            raise ValueError("Missing state payload.")

        requested_n_obs_steps = payload.get("n_obs_steps")
        n_obs_steps = self.n_obs_steps if requested_n_obs_steps is None else int(requested_n_obs_steps)
        if n_obs_steps != self.n_obs_steps:
            raise ValueError(
                f"Per-request n_obs_steps override is not supported. "
                f"Checkpoint expects n_obs_steps={self.n_obs_steps}, got {n_obs_steps}."
            )
        state = _parse_state_payload(state_payload, n_obs_steps)

        instruction = (
            payload.get("instruction")
            or payload.get("task")
            or payload.get("prompt")
            or payload.get("question")
            or ""
        )
        requested_n_action_steps = payload.get("n_action_steps")
        if requested_n_action_steps is None:
            n_action_steps = self.default_n_action_steps
        else:
            n_action_steps = int(requested_n_action_steps)
        num_steps = payload.get("num_steps")
        num_steps = None if num_steps is None else int(num_steps)

        return (
            images,
            image_names,
            state,
            str(instruction or ""),
            tag,
            robot_metadata,
            n_action_steps,
            num_steps,
        )

    def health(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "device": str(self.device),
            "default_seq_len": self.default_seq_len,
            "default_num_steps": self.default_num_steps,
            "n_obs_steps": self.n_obs_steps,
            "n_action_steps": self.default_n_action_steps,
            "default_n_action_steps": self.default_n_action_steps,
            "checkpoint_n_action_steps": self.checkpoint_n_action_steps,
            "use_hf_ckpt": self.use_hf_ckpt,
            "inference_action_mode": self.inference_action_mode,
            "enable_depth_reasoning": bool(self.policy.config.enable_depth_reasoning),
            "enable_inference_cuda_graph": bool(getattr(self.policy.config, "enable_inference_cuda_graph", False)),
            "default_style": self.default_style,
            "default_norm_tag": self.default_norm_tag,
            "save_image_dir": None if self.save_image_dir is None else str(self.save_image_dir),
            "verbose": self.verbose,
            "session_count": len(self._session_states),
        }

    def _capture_policy_state(self) -> PolicySessionState:
        state_owner = self._state_owner()
        raw_depth_caches = getattr(state_owner, "_depth_caches", {})
        return PolicySessionState(
            obs_history=_clone_state_value(getattr(state_owner, "_obs_history", {})),
            action_queues=_clone_state_value(getattr(state_owner, "_action_queues", {})),
            depth_caches=(
                {
                    int(idx): _clone_state_value(cache)
                    for idx, cache in raw_depth_caches.items()
                }
                if isinstance(raw_depth_caches, dict)
                else {}
            ),
        )

    def _restore_policy_state(self, session_id: str) -> None:
        state = self._session_states.get(session_id)
        if state is None:
            self.policy.reset()
            return
        state_owner = self._state_owner()
        state_owner._obs_history = defaultdict(lambda: deque(), _clone_state_value(state.obs_history))
        state_owner._action_queues = defaultdict(lambda: deque(), _clone_state_value(state.action_queues))
        if hasattr(state_owner, "_depth_caches"):
            state_owner._depth_caches = {
                int(idx): _clone_state_value(cache)
                for idx, cache in state.depth_caches.items()
            }
        state_owner._last_depth_video_codes_by_batch = {}
        state_owner._last_model_inference_s = 0.0
        state_owner._last_model_inference_calls = 0

    def reset_session(self, payload: Dict[str, Any]) -> Any:
        session_id = str(payload.get("session_id") or payload.get("prefix") or "").strip()
        if not session_id:
            return JSONResponse({"error": "Missing session_id."}, status_code=400)
        treat_as_prefix = bool(payload.get("prefix", False))
        with self._policy_lock:
            if treat_as_prefix:
                prefix = f"{session_id}:"
                removed = [
                    key
                    for key in self._session_states
                    if key == session_id or key.startswith(prefix)
                ]
                for key in removed:
                    self._session_states.pop(key, None)
            else:
                removed = [session_id] if session_id in self._session_states else []
                self._session_states.pop(session_id, None)
        return JSONResponse({"status": "ok", "removed": removed})

    def _maybe_save_request_images(
        self,
        images: List[np.ndarray],
        image_names: List[str],
        *,
        tag: str,
        style: str,
    ) -> None:
        if self.save_image_dir is None:
            return
        request_idx = self._request_index
        self._request_index += 1
        prefix = (
            f"{request_idx:06d}_"
            f"{_sanitize_filename(tag or 'no_tag')}_"
            f"{_sanitize_filename(style or 'no_style')}"
        )
        for idx, image in enumerate(images):
            image_name = image_names[idx] if idx < len(image_names) else f"image_{idx}"
            output_path = self.save_image_dir / f"{prefix}_{_sanitize_filename(image_name)}.png"
            _save_image_array(image, output_path)

    def predict_action(self, payload: Dict[str, Any]) -> Any:
        t0 = time.time()
        try:
            single_action = str(payload.get("single_action", "false")).strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            action_chunk_request = str(payload.get("action_chunk", "false")).strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            session_id = str(payload.get("session_id") or "default")
            (
                images,
                image_names,
                state,
                instruction,
                tag,
                _robot_metadata,
                n_action_steps,
                num_steps,
            ) = self._parse_request(payload)
            self._maybe_save_request_images(images, image_names, tag=tag, style=self.default_style)

            observations = _build_observations(
                images,
                image_names,
                state,
                instruction,
            )
            if action_chunk_request:
                with self._policy_lock:
                    self._restore_policy_state(session_id)
                    result = self.policy.generate_inference_result_from_observations(
                        observations,
                        norm_tag=tag,
                        num_steps=num_steps,
                        n_action_steps=n_action_steps,
                    )
                    latest_depth_codes = self.policy.get_last_depth_video_codes()
                    inference_calls = int(getattr(self._state_owner(), "_last_model_inference_calls", 1) or 0)
                    self._session_states[session_id] = self._capture_policy_state()
                response_payload: Dict[str, Any] = {
                    "style": result.style,
                    "latency_ms": int((time.time() - t0) * 1000),
                    "inference_calls": inference_calls,
                }
                if result.actions is None:
                    response_payload["error"] = f"Style '{result.style}' does not produce actions."
                    return JSONResponse(response_payload, status_code=500)
                # Public policy APIs return actions in raw environment space.
                action_chunk = result.actions
                actions_np = action_chunk.detach().cpu().numpy()
                response_payload["actions"] = actions_np[0].tolist() if actions_np.ndim >= 3 else actions_np.tolist()
                response_payload["action_shape"] = list(actions_np.shape)
                if latest_depth_codes:
                    first_key = sorted(latest_depth_codes)[0]
                    response_payload["depth_buffer_codes"] = (
                        np.asarray(latest_depth_codes[first_key], dtype=np.int64).reshape(-1).tolist()
                    )
                return JSONResponse(response_payload)

            if single_action:
                with self._policy_lock:
                    self._restore_policy_state(session_id)
                    action_tensor = self.policy.select_action(
                        observations[-1],
                        norm_tag=tag,
                        num_steps=num_steps,
                        n_action_steps=n_action_steps,
                    )
                    latest_depth_codes = self.policy.get_last_depth_video_codes()
                    inference_calls = int(getattr(self._state_owner(), "_last_model_inference_calls", 1) or 0)
                    self._session_states[session_id] = self._capture_policy_state()
                actions_np = action_tensor.detach().cpu().numpy()
                response_payload = {
                    "style": self.default_style,
                    "action": actions_np[0].tolist() if actions_np.ndim >= 2 else actions_np.tolist(),
                    "action_shape": list(actions_np.shape),
                    "latency_ms": int((time.time() - t0) * 1000),
                    "inference_calls": inference_calls,
                }
                if latest_depth_codes:
                    first_key = sorted(latest_depth_codes)[0]
                    response_payload["depth_buffer_codes"] = (
                        np.asarray(latest_depth_codes[first_key], dtype=np.int64).reshape(-1).tolist()
                    )
                return JSONResponse(response_payload)

            # Preserve the previous server semantics: non-streaming chunk requests are stateless.
            with self._policy_lock:
                self.policy.reset()
                result = self.policy.generate_inference_result_from_observations(
                    observations,
                    norm_tag=tag,
                    num_steps=num_steps,
                    n_action_steps=n_action_steps,
                )
            response_payload: Dict[str, Any] = {
                "style": result.style,
            }
            if result.actions is not None:
                # Public policy APIs return actions in raw environment space.
                action_chunk = result.actions
                actions_np = action_chunk.detach().cpu().numpy()
                response_payload["actions"] = actions_np[0].tolist()
                response_payload["action_shape"] = list(actions_np.shape)
            if result.depth_bins is not None:
                depth_np = result.depth_bins.detach().cpu().numpy()
                response_payload["depth_predictions"] = depth_np[0].tolist()
                response_payload["depth_shape"] = list(depth_np.shape)
            latency_ms = int((time.time() - t0) * 1000)
            response_payload["latency_ms"] = latency_ms
            return JSONResponse(response_payload)
        except Exception as exc:
            log.exception("Inference failed")
            return JSONResponse({"error": str(exc)}, status_code=500)

    def run(self, host: str, port: int, timeout_keep_alive: int = 0) -> None:
        uvicorn.run(self.app, host=host, port=port, timeout_keep_alive=timeout_keep_alive)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve MolmoAct2 policy over HTTP.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint directory (unsharded preferred).")
    parser.add_argument(
        "--hf_ckpt",
        dest="hf_ckpt",
        nargs="?",
        const=True,
        type=_parse_bool,
        help=(
            "Whether --checkpoint points to a converted HuggingFace MolmoAct2 checkpoint. "
            "Accepts true/false; default false."
        ),
    )
    parser.set_defaults(hf_ckpt=False)
    parser.add_argument("--host", default="0.0.0.0", help="Bind host.")
    parser.add_argument("--port", type=int, default=8000, help="Bind port.")
    parser.add_argument("--device", default=None, help="Torch device override (e.g., cuda:0).")
    parser.add_argument("--seq_len", type=int, default=None, help="Override max sequence length.")
    parser.add_argument("--num_steps", type=int, default=None, help="Override flow-matching steps.")
    parser.add_argument(
        "--timeout_keep_alive",
        type=int,
        default=0,
        help="Uvicorn keep-alive timeout. Default 0 closes idle HTTP connections immediately.",
    )
    parser.add_argument(
        "--n_action_steps",
        type=int,
        default=None,
        help="Optional server-wide executed-step override. Payload `n_action_steps` overrides this.",
    )
    parser.add_argument("--img_resize", type=str, default=None, help="Optional image resize (e.g., 336x336).")
    parser.add_argument(
        "--image_keys",
        type=str,
        default="yam",
        choices=tuple(IMAGE_KEY_PRESETS.keys()),
        help="Preset image keys to read from the payload.",
    )
    parser.add_argument(
        "--inference_action_mode",
        type=str,
        default="continuous",
        choices=("continuous", "discrete"),
        help="Inference action mode.",
    )
    parser.add_argument(
        "--discrete_action_tokenizer",
        type=str,
        default=None,
        help="Discrete action tokenizer override. Required when --inference_action_mode=discrete.",
    )
    parser.add_argument(
        "--enable_depth_reasoning",
        action="store_true",
        help="Enable depth-token reasoning; serving uses robot_depth_action instead of robot_action.",
    )
    parser.add_argument(
        "--norm_tag",
        dest="norm_tag",
        type=str,
        default="",
        help="Default normalization tag for all requests. Payload tag/norm_tag overrides this.",
    )
    parser.add_argument(
        "--enable_inference_cuda_graph",
        dest="enable_inference_cuda_graph",
        action="store_true",
        default=True,
        help="Enable HF action-expert CUDA graph capture/replay when supported.",
    )
    parser.add_argument(
        "--disable_inference_cuda_graph",
        dest="enable_inference_cuda_graph",
        action="store_false",
        help="Disable HF action-expert CUDA graph capture/replay.",
    )
    parser.add_argument(
        "--save_image_dir",
        type=str,
        default=None,
        help="Optional directory to save each decoded request image.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print model input prompts and generated tokens via the policy wrapper.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args.discrete_action_tokenizer = _require_discrete_action_tokenizer(
        args.inference_action_mode,
        args.discrete_action_tokenizer,
    )
    server = MolmoAct2Server(
        checkpoint=args.checkpoint,
        use_hf_ckpt=bool(args.hf_ckpt),
        device=args.device,
        seq_len=args.seq_len,
        num_steps=args.num_steps,
        n_action_steps=args.n_action_steps,
        image_resize=args.img_resize,
        image_keys=args.image_keys,
        inference_action_mode=args.inference_action_mode,
        discrete_action_tokenizer=args.discrete_action_tokenizer,
        enable_depth_reasoning=bool(args.enable_depth_reasoning),
        norm_tag=args.norm_tag,
        enable_inference_cuda_graph=bool(args.enable_inference_cuda_graph),
        save_image_dir=args.save_image_dir,
        verbose=args.verbose,
    )
    server.run(args.host, args.port, timeout_keep_alive=int(args.timeout_keep_alive))


if __name__ == "__main__":
    main()
