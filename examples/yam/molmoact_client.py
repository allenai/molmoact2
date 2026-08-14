"""MolmoAct2-BimanualYAM policy clients for the eval launcher.

Three interchangeable policies, all producing an ``{"actions": ndarray}`` dict
from an observation:

* :class:`MolmoActServo` — the hosted policy, reached over **one official Servo
  action session** (``sv.session(policy)``) that stays open for the whole run.
  The SDK owns the network path; see :mod:`servo_session_bridge`.
* :class:`MolmoActHTTP` — the sibling ``host_server_yam.py`` ``json_numpy``
  server on this LAN. Self-hosted development path, not a Servo transport.
* :class:`MolmoActLocal` — loads the checkpoint in-process via ``transformers``
  (no server). The local-policy internals are kept in sync with
  ``host_server_yam.py``; if the bf16 patch needles or the ``predict_action``
  signature change there, mirror the change here.
"""

from __future__ import annotations

import importlib.util
import io
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from abc import ABC
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from servo_session_bridge import (
    ACTION_HORIZON,
    ACTION_SPACE,
    CAMERA_KEYS,
    OBSERVATION_ENCODINGS,
    SERVO_PYTHON_ENV,
    STATE_DIM,
    ServoBridgeError,
    ServoDirectHost,
    ServoSessionHost,
    encode_frame,
    frame_parts,
    missing_cameras,
    raw_frame_spec,
    read_frame,
)

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


def _frame_nbytes(frame: Any) -> int:
    """Size of one camera payload, encoded bytes or raw pixel array alike."""
    if isinstance(frame, (bytes, bytearray, memoryview)):
        return len(frame)
    return int(getattr(frame, "nbytes", 0))


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

    def close(self, success: bool = True) -> None:
        """Release any policy-owned session/process. Safe to call more than once.

        ``success`` reports the run's outcome to policies that record it
        remotely (e.g. completing a Servo action session) — pass the real
        rollout outcome, not the default, or every aborted run looks clean on
        the control plane.
        """
        return None


# Default to a server running on this host (host_server_yam.py default port).
DEFAULT_SERVER = "http://127.0.0.1:8202"

REPO_ID = "allenai/MolmoAct2-BimanualYAM"
NORM_TAG = "yam_dual_molmoact2"
DEFAULT_NUM_STEPS = 10

# ``STATE_DIM`` / ``ACTION_HORIZON`` / ``CAMERA_KEYS`` are the BimanualYAM
# contract facts and live in ``servo_session_bridge`` so the robot process and
# the Servo SDK process cannot drift.  ``yam_dual_molmoact2`` emits 30
# absolute-pose targets per query; the rollout executor validates and uses the
# actual returned chunk length.

#: Machine (SDK) credential bundle written for this robot. Mode 0600.
DEFAULT_SERVO_CREDENTIALS = "~/.config/servo/molmoact2-yam-sdk.json"


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


class MolmoActHTTP(PolicyBase):
    """Self-hosted ``host_server_yam.py`` client (``json_numpy`` over plain HTTP).

    Development/LAN path only: no authentication, no generation fencing, no
    control-plane record of the run. Hosted inference goes through
    :class:`MolmoActServo`.
    """

    def __init__(
        self,
        server: Optional[str] = None,
        *,
        request_timeout_sec: float = 60.0,
        jpeg_wire: bool = True,
        # MolmoAct2-BimanualYAM's own image processor (crop_mode "resize")
        # squishes whatever arrives down to this size before the vision
        # tower ever sees it (checkpoint's processor_config.json /
        # Default None: ship SOURCE resolution and let the checkpoint's own
        # processor do its trained 640x360->378x378 bilinear stretch.
        # Measured 2026-08-14 (thor + odin, 10 draws each at the carry
        # fixture): a client-side LANCZOS pre-resize to 378x378 shifts the
        # gripper-open mode fraction +0.3-0.4 on BOTH machines, while
        # native-resolution q85 JPEG is behaviorally indistinguishable from
        # lossless PNG. The wire cost of native 640x360 over 378x378 is
        # ~1.6x pixels -- noise next to the behavioral shift. Pass an
        # explicit (H, W) only for bandwidth experiments, never for rollouts.
        image_size: Optional[Tuple[int, int]] = None,
    ):
        self.logger = get_molmoact_logger()
        if request_timeout_sec <= 0:
            raise ValueError("request_timeout_sec must be positive")
        self.jpeg_wire = bool(jpeg_wire)
        self.request_timeout_sec = float(request_timeout_sec)
        self.image_size = tuple(image_size) if image_size is not None else None
        self.url = _normalize_server_url(server)
        self._http = requests.Session()
        self.multi_views = True
        self.action_horizon = ACTION_HORIZON

        # Log configuration
        self.logger.info(f"MolmoActHTTP initialized with URL: {self.url}")
        self.logger.info(f"Multi-views enabled: {self.multi_views}")
        self.logger.info(f"Action horizon: {self.action_horizon}")

    def get_action_horizon(self):
        return self.action_horizon

    def _resize_for_wire(self, arr: np.ndarray) -> np.ndarray:
        """Downsize before transfer to the checkpoint's own processor input size.

        A no-op once ``arr`` is already at ``self.image_size`` (the common
        case once the camera config is updated) or when resizing is disabled.
        Bilinear to match the image processor's own resample=2 setting.
        """
        if self.image_size is None:
            return arr
        target_h, target_w = self.image_size
        if arr.shape[:2] == (target_h, target_w):
            return arr
        return np.array(
            Image.fromarray(arr).resize((target_w, target_h), resample=Image.BILINEAR)
        )

    def reproducibility_metadata(self) -> Dict[str, Any]:
        return {
            "backend": "http",
            "endpoint": self.url,
            "image_encoding": "json_numpy",
            "deterministic_generator": False,
            "note": "remote server does not expose a per-query generator seed",
        }

    def close(self, success: bool = True) -> None:
        del success  # no remote run record to update
        self._http.close()

    def prepare_input(self, obs, instruction):
        self.logger.info("Preparing input for MolmoActHTTP inference")
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
                    obs["joint_positions"], source="MolmoActHTTP"
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
        self.logger.info("Starting MolmoActHTTP inference")

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
                left_img_np = self._resize_for_wire(np.array(images[0]))
                front_img_np = self._resize_for_wire(np.array(images[1]))
                right_img_np = self._resize_for_wire(np.array(images[2]))

                self.logger.info(f"Left image shape: {left_img_np.shape}")
                self.logger.info(f"Front image shape: {front_img_np.shape}")
                self.logger.info(f"Right image shape: {right_img_np.shape}")

                # Prepare the payload with the image and instruction from the script
                if self.jpeg_wire:
                    # Compact wire format: ~90 KB/act of base64 JPEG q85 instead of
                    # ~2 MB of raw json_numpy arrays. Decisive over the WiFi uplink
                    # (measured 2-4 s/act raw vs sub-second JPEG); pixel impact of
                    # q85 was measured harmless offline. Server decodes back to RGB.
                    import base64 as _b64

                    def _jpg(arr):
                        buf = io.BytesIO()
                        Image.fromarray(arr).save(buf, format="JPEG", quality=85)
                        return _b64.b64encode(buf.getvalue()).decode("ascii")

                    payload = {
                        "images_jpeg": {
                            "left": _jpg(left_img_np),
                            "top": _jpg(front_img_np),
                            "right": _jpg(right_img_np),
                        },
                        "timestamp": time.time(),
                        "instruction": instruction,
                        "state": state,
                        "normalization_tag": "yam_dual_molmoact2",
                    }
                else:
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
            response = self._http.post(
                server_url,
                headers=headers,
                data=serialized_payload,
                timeout=self.request_timeout_sec,
            )
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
# Hosted MolmoAct over an official Servo action session
# ---------------------------------------------------------------------------

_BRIDGE_EOF = object()


class ServoSessionTransport:
    """Owns exactly one official Servo action session for a whole eval run.

    The SDK requires Python >= 3.12 while the robot runtime is 3.11, so the
    session runs in this process when ``servo`` is importable here and in a
    helper subprocess otherwise (see :mod:`servo_session_bridge`). Both cases
    execute the same :class:`ServoSessionHost` code; only the local hop
    differs. The interpreter is chosen once, explicitly — there is no silent
    fallback between the two, and nothing but official SDK traffic leaves the
    machine either way.
    """

    #: Slack subtracted from the parent's wait budget before it is handed to
    #: the child as its own SDK timeout, so the parent is always strictly more
    #: patient and a legitimately slow (but successful) child call is never
    #: mistaken for a dead one.
    _CHILD_TIMEOUT_SLACK_SEC = 10.0

    def __init__(
        self,
        *,
        credentials: Optional[str] = None,
        deployment_id: Optional[str] = None,
        grant: Optional[str] = None,
        instruction: Optional[str] = None,
        servo_python: Optional[str] = None,
        open_timeout_sec: float = 600.0,
        act_timeout_sec: float = 120.0,
        logger: Optional[logging.Logger] = None,
        bridge_env: Optional[Dict[str, str]] = None,
        observation_encoding: str = "jpeg",
        h264_crf: Optional[int] = None,
    ):
        # Cold starts documented for this deployment family run 26s-550s
        # (CLAUDE.md); ``open`` needs a materially longer budget than a
        # steady-state ``act`` or the very first session will always poison
        # itself before the control plane finishes provisioning.
        if open_timeout_sec <= 0:
            raise ValueError("open_timeout_sec must be positive")
        if act_timeout_sec <= 0:
            raise ValueError("act_timeout_sec must be positive")
        self.bridge_env = dict(bridge_env or {})
        if observation_encoding not in OBSERVATION_ENCODINGS:
            raise ServoBridgeError(
                f"observation_encoding must be one of {list(OBSERVATION_ENCODINGS)}, "
                f"got {observation_encoding!r}"
            )
        if h264_crf is not None and observation_encoding != "h264":
            raise ServoBridgeError("h264_crf only applies to observation_encoding='h264'")
        if observation_encoding != "jpeg" and not grant:
            raise ServoBridgeError(
                f"the {observation_encoding!r} observation wire is negotiated per action "
                "session by a self-hosted `servo serve` endpoint; it needs grant= "
                "(--servo-grant / eval.direct.grant), not a managed deployment"
            )
        self.observation_encoding = observation_encoding
        self.h264_crf = int(h264_crf) if h264_crf is not None else None
        self.grant = str(Path(grant).expanduser()) if grant else None
        if self.grant is None:
            # Hosted mode: both halves of the control-plane identity are required.
            if not credentials or not deployment_id:
                raise ServoBridgeError(
                    "hosted mode needs credentials and deployment_id; "
                    "self-hosted mode needs grant="
                )
            self.credentials = str(Path(credentials).expanduser())
            self.deployment_id = str(deployment_id)
        else:
            if credentials or deployment_id:
                raise ServoBridgeError(
                    "grant= names the entire endpoint identity; do not combine it "
                    "with credentials/deployment_id (no fallback path exists)"
                )
            self.credentials = None
            self.deployment_id = "self-hosted"
        self.instruction = instruction
        self.open_timeout_sec = float(open_timeout_sec)
        self.act_timeout_sec = float(act_timeout_sec)
        self.logger = logger or get_molmoact_logger()
        self.identity: Dict[str, Any] = {}
        self._host: Optional[ServoSessionHost] = None
        self._process: Optional[subprocess.Popen] = None
        self._responses: "queue.Queue[Any]" = queue.Queue()
        self._reader: Optional[threading.Thread] = None
        self._request_seq = 0
        self._poisoned: Optional[str] = None
        self._python = self._resolve_python(servo_python)
        self.mode = "in-process" if self._python is None else "subprocess"

    def _child_timeout(self, parent_timeout_sec: float) -> float:
        """The SDK timeout handed to the child: always < the parent's wait."""
        return max(1.0, parent_timeout_sec - self._CHILD_TIMEOUT_SLACK_SEC)

    # -- interpreter selection ------------------------------------------------

    @staticmethod
    def _resolve_python(servo_python: Optional[str]) -> Optional[str]:
        """Return the helper interpreter, or ``None`` to run the SDK in-process."""
        explicit = servo_python or os.environ.get(SERVO_PYTHON_ENV)
        if explicit:
            resolved = Path(explicit).expanduser()
            if not resolved.exists():
                raise ServoBridgeError(f"servo_python interpreter not found: {resolved}")
            return str(resolved)
        if importlib.util.find_spec("servo") is not None:
            return None
        raise ServoBridgeError(
            "the official servo SDK is not importable in the robot runtime "
            f"({sys.executable}, Python {sys.version.split()[0]}; the SDK needs >= 3.12). "
            f"Set eval.server.servo_python, --servo-python, or {SERVO_PYTHON_ENV} to an "
            "interpreter that has servo-client installed, e.g. "
            "/home/npow/code/servo/.venv/bin/python"
        )

    # -- lifecycle ------------------------------------------------------------

    def open(self) -> Dict[str, Any]:
        """Open the single action session; returns its control-plane identity."""
        if self.identity:
            return dict(self.identity)
        # The SDK client's own httpx timeout is set once, from the (longer)
        # open budget, and then covers the rest of the session's calls too —
        # act() bounds itself with the parent-side queue wait below.
        sdk_timeout = self._child_timeout(self.open_timeout_sec)
        if self._python is None:
            if self.grant is not None:
                self._host = ServoDirectHost(
                    grant=self.grant,
                    instruction=self.instruction,
                    timeout_sec=sdk_timeout,
                    observation_encoding=self.observation_encoding,
                    h264_crf=self.h264_crf,
                )
            else:
                self._host = ServoSessionHost(
                    credentials=self.credentials,
                    deployment_id=self.deployment_id,
                    instruction=self.instruction,
                    timeout_sec=sdk_timeout,
                )
            identity = self._host.open()
        else:
            self._spawn()
            header = {
                "op": "open",
                "instruction": self.instruction,
                "timeout_sec": sdk_timeout,
                "observation_encoding": self.observation_encoding,
                "h264_crf": self.h264_crf,
            }
            if self.grant is not None:
                header["grant"] = self.grant
            else:
                header["credentials"] = self.credentials
                header["deployment_id"] = self.deployment_id
            identity = self._request(header, timeout=self.open_timeout_sec)["identity"]
        self.identity = dict(identity)
        self.logger.info(
            "Servo action session %s open on deployment %s (generation %s, %s wire, %s SDK)",
            self.identity.get("session_id"),
            self.identity.get("deployment_id"),
            self.identity.get("generation_id"),
            self.identity.get("observation_encoding") or self.observation_encoding,
            self.mode,
        )
        return dict(self.identity)

    def act(
        self,
        images: Dict[str, Any],
        state: Any,
        instruction: Optional[str] = None,
        noise_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Send one observation over the open session and return its prediction.

        ``images`` carries pre-encoded JPEG bytes on the jpeg wire and raw
        ``HxWx3`` ``uint8`` arrays on a codec wire, matching
        ``self.observation_encoding``. Nothing here encodes: a stateful wire's
        payload is minted by the session's own transport at send time, so the
        only thing that can legitimately cross this pipe is pixels.
        """
        if not self.identity:
            self.open()
        # Both modes must fail identically: the caller only handles
        # ServoBridgeError, and a stalled camera must not surface as a KeyError
        # from one transport and a clean error from the other.
        missing = missing_cameras(images)
        if missing:
            raise ServoBridgeError(f"Servo observation is missing camera bytes: {missing}")
        if self._host is not None:
            return self._host.act(
                images, state, instruction=instruction, noise_seed=noise_seed
            )
        header: Dict[str, Any] = {
            "op": "act",
            "state": [float(value) for value in state],
            "instruction": instruction,
            "noise_seed": None if noise_seed is None else int(noise_seed),
        }
        if self.observation_encoding == "jpeg":
            header["images"] = {key: index for index, key in enumerate(CAMERA_KEYS)}
            buffers: List[Any] = [images[key] for key in CAMERA_KEYS]
        else:
            frames: Dict[str, Any] = {}
            buffers = []
            for index, key in enumerate(CAMERA_KEYS):
                array = images[key]
                shape = getattr(array, "shape", None)
                if shape is None or len(shape) != 3 or shape[2] != 3:
                    raise ServoBridgeError(
                        f"the {self.observation_encoding!r} wire carries raw HxWx3 uint8 "
                        f"pixels; camera {key!r} supplied {type(array).__name__} "
                        f"shape={shape!r}"
                    )
                frames[key] = raw_frame_spec(index, int(shape[0]), int(shape[1]))
                # ``cast("B")`` because ``len(memoryview(hxwx3))`` is the ROW
                # COUNT, not the byte count -- framing the un-cast view would
                # declare a 360-byte buffer for a 691200-byte frame.
                buffers.append(memoryview(np.ascontiguousarray(array, dtype=np.uint8)).cast("B"))
            header["frames"] = frames
        return self._request(header, buffers, timeout=self.act_timeout_sec)["prediction"]

    def close(self, success: bool = True) -> None:
        """Close the session (and the helper process). Safe to call repeatedly."""
        host, self._host = self._host, None
        if host is not None:
            host.close(success=success)
        process, self._process = self._process, None
        if process is None:
            self.identity = {}
            return
        try:
            # A poisoned transport gets no graceful close: its reply stream is
            # already unreliable, so tear the pipe down instead of waiting. A
            # generous budget here (not act_timeout_sec) gives an in-flight
            # session.close() a real chance to complete the control-plane
            # record before SIGTERM would otherwise cut it off mid-call.
            if process.poll() is None and self._poisoned is None:
                try:
                    self._request(
                        {"op": "close", "success": success},
                        process=process,
                        timeout=self.open_timeout_sec,
                    )
                except Exception as exc:  # noqa: BLE001 — teardown is best effort
                    self.logger.warning("Servo session close failed: %s", exc)
            if process.stdin is not None:
                try:
                    process.stdin.close()
                except OSError:
                    pass
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.logger.warning("Servo session bridge did not exit; terminating")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        finally:
            reader, self._reader = self._reader, None
            if reader is not None:
                reader.join(timeout=5)
            # The reader owns the read end; closing it before the thread is done
            # would break its in-flight read.
            if process.stdout is not None:
                try:
                    process.stdout.close()
                except OSError:
                    pass
            self.identity = {}

    # -- helper process -------------------------------------------------------

    def _spawn(self) -> None:
        bridge = Path(__file__).resolve().parent / "servo_session_bridge.py"
        self.logger.info("Starting Servo session bridge: %s %s", self._python, bridge)
        # A fresh child needs a fresh reply channel: a stale sentinel left by a
        # prior child's reader thread (e.g. the _BRIDGE_EOF put on the last
        # close) must never be handed to this child's first request.
        self._responses = queue.Queue()
        self._request_seq = 0
        self._poisoned = None
        self._process = subprocess.Popen(
            [self._python, str(bridge)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # The child's stderr is the operator's view of SDK failures; leave
            # it attached to this process's stderr rather than swallowing it.
            bufsize=0,
            # PYTHONPATH points at the robot runtime's 3.11 modules, which the
            # helper interpreter must not import. ``bridge_env`` is the only
            # way to add environment to the helper.
            env={
                **{key: value for key, value in os.environ.items() if key != "PYTHONPATH"},
                **self.bridge_env,
            },
        )
        self._reader = threading.Thread(
            target=self._read_responses,
            args=(self._process,),
            name="servo-bridge-reader",
            daemon=True,
        )
        self._reader.start()

    def _read_responses(self, process: subprocess.Popen) -> None:
        try:
            while True:
                header, _ = read_frame(process.stdout)
                self._responses.put(header)
        except EOFError:
            self._responses.put(_BRIDGE_EOF)
        except Exception as exc:  # noqa: BLE001 — surfaced on the next request
            self._responses.put(exc)

    @staticmethod
    def _write_all(stream: Any, data: bytes) -> None:
        """Write ``data`` in full even on a raw (``bufsize=0``) pipe.

        A raw ``FileIO.write`` performs a single ``write(2)`` and returns the
        partial count on a full pipe buffer — an act frame (tens of KiB of
        JPEG) is routinely larger than the OS pipe buffer, so a short write is
        the common case here, not an edge case.
        """
        view = memoryview(data)
        while view:
            written = stream.write(view)
            if written is None:  # non-blocking stream reporting no capacity
                continue
            view = view[written:]

    def _request(
        self,
        header: Dict[str, Any],
        buffers: Optional[List[bytes]] = None,
        *,
        process: Optional[subprocess.Popen] = None,
        timeout: float,
    ) -> Dict[str, Any]:
        process = process or self._process
        if process is None or process.stdin is None:
            raise ServoBridgeError("Servo session bridge is not running")
        if self._poisoned is not None:
            raise ServoBridgeError(self._poisoned)
        self._request_seq += 1
        request_id = self._request_seq
        header = {**header, "request_id": request_id}
        # Encode outside the write guard: an encoding failure is a local
        # observation problem (an empty camera frame, a non-finite state) and
        # must not be reported as a dead helper process.
        try:
            parts = frame_parts(header, buffers or ())
        except ValueError as exc:
            raise ServoBridgeError(f"Servo observation cannot be framed: {exc}") from exc
        try:
            for part in parts:
                self._write_all(process.stdin, part)
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise ServoBridgeError(
                f"Servo session bridge exited (returncode={process.poll()})"
            ) from exc
        try:
            response = self._responses.get(timeout=timeout)
        except queue.Empty:
            # A late reply must never be mistaken for the NEXT chunk's actions:
            # once a request has timed out this transport is done, and the run
            # fails closed rather than executing a stale action chunk.
            self._poisoned = (
                f"Servo session bridge did not answer {header.get('op')!r} within "
                f"{timeout:g}s; the session is no longer usable"
            )
            raise ServoBridgeError(self._poisoned) from None
        if response is _BRIDGE_EOF:
            raise ServoBridgeError(
                f"Servo session bridge exited (returncode={process.poll()}); "
                "its stderr is above"
            )
        if isinstance(response, BaseException):
            raise ServoBridgeError(f"Servo session bridge read failed: {response}")
        if response.get("request_id") != request_id:
            self._poisoned = (
                "Servo session bridge answered out of order "
                f"(got request_id={response.get('request_id')!r}, expected {request_id})"
            )
            raise ServoBridgeError(self._poisoned)
        if not response.get("ok"):
            raise ServoBridgeError(
                str(response.get("error") or "Servo session bridge reported a failure"),
                error_type=response.get("error_type"),
            )
        return response


class MolmoActServo(PolicyBase):
    """Hosted MolmoAct2-BimanualYAM over one official Servo action session.

    The session opens on the first query and is reused for every action chunk
    of the run: the control plane issues one signed offer/lease, and each
    observation rides the SDK's protobuf action-session transport with the
    three camera frames as JPEG bytes — no base64, no custom endpoint.
    """

    def __init__(
        self,
        deployment: Optional[str] = None,
        *,
        credentials: Optional[str] = None,
        grant: Optional[str] = None,
        instruction: Optional[str] = None,
        servo_python: Optional[str] = None,
        open_timeout_sec: float = 600.0,
        act_timeout_sec: float = 120.0,
        jpeg_quality: int = 85,
        image_size: Optional[int] = None,
        image_fit: str = "pad",
        observation_encoding: str = "jpeg",
        h264_crf: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        self.logger = get_molmoact_logger()
        if image_fit not in ("pad", "stretch"):
            raise ValueError("image_fit must be 'pad' or 'stretch'")
        self.image_fit = image_fit
        if not deployment and not grant:
            raise ValueError(
                "the hosted policy needs a managed Servo deployment id "
                "(--servo-deployment or eval.server.deployment); the self-hosted "
                "policy needs a grant (--servo-grant or eval.direct.grant)"
            )
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        if image_size is not None and image_size <= 0:
            raise ValueError("image_size must be positive or None")
        self.jpeg_quality = int(jpeg_quality)
        self.image_size = int(image_size) if image_size is not None else None
        self.observation_encoding = str(observation_encoding)
        self.h264_crf = int(h264_crf) if h264_crf is not None else None
        self.action_horizon = ACTION_HORIZON
        # Seeding mirrors MolmoActLocal exactly -- same RolloutSeedPlan, same
        # per-query derivation -- because the point is a PAIRED comparison: the
        # same base seed must produce the same noise draw whether the model runs
        # on this Jetson or on a remote GPU. Servo's raw-transformers backend
        # seeds a torch.Generator on the serve the same way MolmoActLocal does
        # here, so identical query seeds mean identical draws.
        self._configured_seed_plan = RolloutSeedPlan(seed)
        self._query_seed_plan = RolloutSeedPlan(None)
        self._rollout_seed = self._configured_seed_plan.rollout_seed(0)
        self._inference_index = 0
        self._seed_proven = False
        self._transport = ServoSessionTransport(
            credentials=(credentials or DEFAULT_SERVO_CREDENTIALS) if not grant else None,
            deployment_id=deployment,
            grant=grant,
            instruction=instruction,
            servo_python=servo_python,
            open_timeout_sec=open_timeout_sec,
            act_timeout_sec=act_timeout_sec,
            logger=self.logger,
            observation_encoding=self.observation_encoding,
            h264_crf=self.h264_crf,
        )
        self.logger.info(
            "MolmoActServo bound to %s (%s wire, %s SDK)",
            f"grant {grant} (self-hosted, no fallback)" if grant else
            f"deployment {deployment} (credentials {self._transport.credentials})",
            self.observation_encoding,
            self._transport.mode,
        )

    @property
    def identity(self) -> Dict[str, Any]:
        """Control-plane identity of the open session (empty until it opens)."""
        return dict(self._transport.identity)

    def get_action_horizon(self) -> int:
        return self.action_horizon

    def open(self) -> Dict[str, Any]:
        """Open the session up front so credential/deployment errors surface early."""
        return self._transport.open()

    def close(self, success: bool = True) -> None:
        self._transport.close(success=success)

    def begin_rollout(self, rollout_seed: Optional[int]) -> Dict[str, Any]:
        """Start an isolated deterministic noise stream for this rollout."""
        self._rollout_seed = RolloutSeedPlan(rollout_seed).base_seed
        self._inference_index = 0
        return {
            "rollout_seed": self._rollout_seed,
            "query_seed_strategy": SEED_STRATEGY,
            "generator_device": "remote-serve",
            "deterministic_generator": self._rollout_seed is not None,
        }

    def _noise_seed_for_next_query(self) -> tuple[Optional[int], Dict[str, Any]]:
        query_index = self._inference_index
        query_seed = self._query_seed_plan.query_seed(self._rollout_seed, query_index)
        return query_seed, {
            "rollout_seed": self._rollout_seed,
            "query_index": query_index,
            "query_seed": query_seed,
            "seed_strategy": SEED_STRATEGY,
        }

    def reproducibility_metadata(self) -> Dict[str, Any]:
        identity = self._transport.identity
        return {
            "backend": "servo-action-session",
            "base_url": identity.get("base_url"),
            "deployment_id": self._transport.deployment_id,
            "session_id": identity.get("session_id"),
            "generation_id": identity.get("generation_id"),
            "checkpoint_digest": identity.get("checkpoint_digest"),
            "manifest_hash": identity.get("manifest_hash"),
            "image_encoding": (
                f"jpeg-quality-{self.jpeg_quality}-size-{self.image_size or 'source'}"
                if self.observation_encoding == "jpeg"
                else f"{self.observation_encoding}-crf-{self.h264_crf or 'default'}"
                f"-size-{self.image_size or 'source'}"
            ),
            "observation_encoding": self.observation_encoding,
            "configured_seed": self._configured_seed_plan.base_seed,
            "query_seed_strategy": SEED_STRATEGY,
            # A FACT about acts already returned, not an intention. The serve
            # reports the scheme it actually sampled under and the SDK refuses
            # an act whose response does not carry one, so a run that reaches
            # here with a seed requested is a run whose seed was honoured.
            "deterministic_generator": self._seed_proven,
            "seed_requested": self._rollout_seed is not None,
        }

    def prepare_input(self, obs: Dict[str, Any], instruction: str) -> Dict[str, Any]:
        self.logger.info("Preparing input for MolmoActServo inference")
        self.logger.info(f"Instruction: '{instruction}'")
        return {
            "left_camera_rgb": obs["left_camera_rgb"],
            "front_camera_rgb": obs["front_camera_rgb"],
            "right_camera_rgb": obs["right_camera_rgb"],
            "instruction": instruction,
            "state": require_bimanual_state(
                obs["joint_positions"], source="MolmoActServo"
            ),
        }

    def inference(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        encode_started = time.perf_counter()
        # Servo's stable camera names for the YAM embodiment; the runtime maps
        # them to the checkpoint's own keys through the deployment manifest.
        #
        # On the jpeg wire the client mints the payload (unchanged). On a codec
        # wire it must NOT: an h264 access unit is only valid against the
        # decoder state its predecessor left on the session it was minted for,
        # so the session's transport encodes at send time and pre-encoded bytes
        # are refused. What crosses this boundary is therefore pixels.
        sources = {
            "top": input_dict["front_camera_rgb"],
            "left": input_dict["left_camera_rgb"],
            "right": input_dict["right_camera_rgb"],
        }
        if self.observation_encoding == "jpeg":
            images: Dict[str, Any] = {
                key: self._jpeg_bytes(value) for key, value in sources.items()
            }
        else:
            images = {key: self._raw_pixels(value) for key, value in sources.items()}
        encode_ms = (time.perf_counter() - encode_started) * 1000.0
        state = require_bimanual_state(
            input_dict["state"], source="MolmoActServo request"
        )

        noise_seed, seed_metadata = self._noise_seed_for_next_query()
        act_started = time.perf_counter()
        prediction = self._transport.act(
            images,
            state.tolist(),
            instruction=input_dict.get("instruction"),
            noise_seed=noise_seed,
        )
        act_ms = (time.perf_counter() - act_started) * 1000.0
        # Advance only on success, matching MolmoActLocal: a retried act must
        # replay its own seed, not consume the next one, or a recovered run
        # silently stops being the run its manifest describes.
        self._inference_index += 1

        actions = np.asarray(prediction["actions"], dtype=np.float32)
        if actions.shape != (ACTION_HORIZON, STATE_DIM) or not np.isfinite(actions).all():
            raise RuntimeError(
                f"Servo returned a malformed action chunk: shape {actions.shape}"
            )
        telemetry = prediction.get("telemetry") or {}
        # The serve reports the mechanism it actually sampled under. The SDK
        # already refuses a seeded act whose response carries no scheme, so
        # reaching here with one recorded is the proof -- not the intent --
        # that this query's seed reached the sampler.
        noise_scheme = telemetry.get("noise_scheme")
        if noise_seed is not None and noise_scheme:
            self._seed_proven = True
        seed_metadata["noise_scheme"] = noise_scheme
        # The SDK's ``ActionPrediction.telemetry`` flattens
        # ``runtime.inference_ms`` to the top-level key ``server_infer_ms``
        # (servo/types.py:_telemetry_view) — there is no ``infer_ms`` key.
        server_infer_ms = telemetry.get("server_infer_ms")
        image_bytes = sum(_frame_nbytes(frame) for frame in images.values())
        # The bytes that actually left this machine. On the codec wire
        # ``image_bytes`` is the raw pixel buffer handed to the SDK, which is
        # ~2 MB and says nothing about the wire; ``request_bytes`` is the
        # payload the session sent, and it is the operator's proof the wire
        # switched (~88 KB jpeg vs ~21 KB h264).
        request_bytes = telemetry.get("request_bytes")
        self.logger.info(
            "Servo action session step in %.1fms (%s wire, capture %.1fms, "
            "request_bytes %s, %d source bytes, server inference %sms)",
            act_ms,
            self.observation_encoding,
            encode_ms,
            request_bytes if request_bytes is not None else "unknown",
            image_bytes,
            server_infer_ms if server_infer_ms is not None else "unknown",
        )
        identity = self._transport.identity
        return {
            "actions": actions,
            "servo": prediction,
            "reproducibility": {**self.reproducibility_metadata(), **seed_metadata},
            "transport": {
                "protocol": "servo-action-session",
                "action_space": prediction.get("action_space", ACTION_SPACE),
                "session_id": identity.get("session_id"),
                "observation_encoding": self.observation_encoding,
                "image_bytes": image_bytes,
                "request_bytes": request_bytes,
                "act_ms": act_ms,
                "encode_ms": encode_ms,
                "server_inference_ms": server_infer_ms,
                "telemetry": telemetry,
            },
        }

    def _raw_pixels(self, image: Any) -> np.ndarray:
        """Fit one camera to a contiguous ``HxWx3`` ``uint8`` array, unencoded.

        Deliberately the same normalization and the same optional
        ``image_size``/``image_fit`` diagnostic resize the jpeg path applies, so
        the two wires are compared on the same pixels and only the encoding
        differs. No encode happens here: the session owns that.
        """
        array = self._fit_source(image)
        return np.ascontiguousarray(array, dtype=np.uint8)

    def _fit_source(self, image: Any) -> np.ndarray:
        """Normalize dtype/shape and apply the optional diagnostic resize."""
        array = np.asarray(image)
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"Servo camera image must be HxWx3, got {array.shape}")
        if self.image_size is not None:
            if self.image_fit == "stretch":
                # The signed BimanualYAM camera contract is stretch; sending
                # letterboxed pixels to that endpoint changes the model's input
                # distribution (the 2026-08-11 client-fit audit bug).
                array = np.asarray(
                    Image.fromarray(array).resize(
                        (self.image_size, self.image_size), Image.BILINEAR
                    )
                )
            else:
                array = self._resize_with_pad(array, self.image_size)
        return array

    def _jpeg_bytes(self, image: Any) -> bytes:
        array = np.asarray(image)
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"Servo camera image must be HxWx3, got {array.shape}")
        if self.image_size is not None:
            if self.image_fit == "stretch":
                # The signed BimanualYAM camera contract is stretch; sending
                # letterboxed pixels to that endpoint changes the model's input
                # distribution (the 2026-08-11 client-fit audit bug).
                array = np.asarray(
                    Image.fromarray(array).resize(
                        (self.image_size, self.image_size), Image.BILINEAR
                    )
                )
            else:
                array = self._resize_with_pad(array, self.image_size)
        output = io.BytesIO()
        Image.fromarray(array).convert("RGB").save(
            output,
            format="JPEG",
            quality=self.jpeg_quality,
            optimize=False,
        )
        return output.getvalue()

    @staticmethod
    def _resize_with_pad(image: np.ndarray, size: int) -> np.ndarray:
        """Optional diagnostic resize; Servo normally owns model preprocessing."""

        source_h, source_w = image.shape[:2]
        scale = min(size / source_w, size / source_h)
        resized_h = max(1, round(source_h * scale))
        resized_w = max(1, round(source_w * scale))
        y_index = np.linspace(0, source_h - 1, resized_h).astype(int)
        x_index = np.linspace(0, source_w - 1, resized_w).astype(int)
        resized = image[np.ix_(y_index, x_index)].astype(np.uint8)
        output = np.zeros((size, size, 3), dtype=np.uint8)
        top = (size - resized_h) // 2
        left = (size - resized_w) // 2
        output[top : top + resized_h, left : left + resized_w] = resized
        return output


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
    """In-process MolmoAct policy. Drop-in replacement for :class:`MolmoActServo`."""

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
