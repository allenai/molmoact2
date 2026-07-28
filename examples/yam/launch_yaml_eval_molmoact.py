"""MolmoAct eval launcher.

Runs N rollouts, prompting for an instruction each time. Saves all three
cameras frame-by-frame (PNG) plus the joint trajectory (``episode.h5``) per
rollout, classifies rollouts via cv2 keypress (y/n/q) or a post-timeout
stdin prompt, and converts the session's labeled rollouts to a LeRobot v3.0
dataset on the way out.

CLI::

    python examples/yam/launch_yaml_eval_molmoact.py \
        --config_path examples/yam/configs/yam_left.yaml \
        --right-config-path examples/yam/configs/yam_right.yaml \
        -n 10
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import tyro
from omegaconf import OmegaConf

from camera_client import CameraClient
from gello_min.realsense_camera import RealSenseCamera, get_device_ids
from gello_min.v4l2_camera import V4L2Camera
from gello_min.env import RobotEnv
from eval_utils import (
    EvalRolloutSaver,
    LiveCameraView,
    RolloutOutcome,
    convert_session_to_lerobot,
    move_rollout,
    prompt_instruction,
    resolve_label,
)
from gello_min.robot import BimanualRobot
from gello_min.launch_utils import instantiate_from_dict, move_to_start_position
from gello_min.logging_utils import log_collect_demos
from molmoact_client import MolmoAct, MolmoActLocal
from rollout_manifest import (
    RolloutSeedPlan,
    build_rollout_manifest,
    configure_process_seed,
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
DEVICE = os.environ.get("LEROBOT_TEST_DEVICE", "cuda") if torch.cuda.is_available() else "cpu"


class _DisabledCamera:
    """Camera placeholder that preserves the model's three-image input shape."""

    def read(self):
        return np.zeros((360, 640, 3), dtype=np.uint8), None


# ---------------------------------------------------------------------------
# Runtime cleanup
# ---------------------------------------------------------------------------

_env: Optional[RobotEnv] = None
_bimanual: bool = False
_left_cfg: Optional[Dict[str, Any]] = None
_right_cfg: Optional[Dict[str, Any]] = None
_park_done: bool = False
_resources_closed: bool = False
_bimanual_execution_mask: Optional["BimanualActiveArmHoldMask"] = None


def _park_robot() -> None:
    """Best-effort optional move to configured start joints before shutdown."""
    global _park_done
    if _park_done or _env is None:
        return
    _park_done = True
    if _bimanual_execution_mask is not None:
        # In active-arm-hold and shadow modes, an implicit park target would
        # move the supposedly inactive arm during teardown.  Leave both arms
        # at their final measured positions; the regular close path then
        # disables their motor-control loops.
        logger.info("Skipping implicit park in bimanual active-arm hold mode")
        return
    print("Parking robot at start position...")
    try:
        if _bimanual:
            move_to_start_position(_env, True, _left_cfg, _right_cfg)
        else:
            move_to_start_position(_env, False, _left_cfg)
    except Exception as exc:  # noqa: BLE001 — best-effort cleanup
        logger.warning("Parking failed: %s", exc)


def _shutdown_runtime() -> None:
    """Park if requested, then deterministically release robot and cameras.

    ``MotorChainRobot`` owns a non-daemon control thread. This must run in an
    explicit ``finally`` in ``main``; an atexit hook alone is too late because
    Python waits for that thread before entering atexit handlers.
    """
    global _resources_closed
    if _resources_closed:
        return
    _resources_closed = True

    _park_robot()
    if _env is None:
        return
    try:
        _env.close()
    except Exception as exc:  # noqa: BLE001 — all teardown is best-effort
        logger.warning("Runtime close failed: %s", exc)


def _tail_text(path: Path, limit: int = 2000) -> str:
    try:
        return path.read_text(encoding="utf-8")[-limit:]
    except OSError:
        return "<no watchdog log available>"


def _start_rerun_export_watchdog(
    base_save_dir: Path,
    rerun_cfg: Dict[str, Any],
    control_hz: float,
) -> subprocess.Popen[bytes] | None:
    """Start a detached offline exporter *before* any robot resources exist.

    The child runs no model, camera, CAN, or i2rt code. It waits for this
    launcher to exit, then creates/retries each rollout's RRD from the durable
    raw files. Starting and verifying it before model/robot startup means an
    unexpected control-process exit cannot skip the replay handoff.
    """
    if not bool(rerun_cfg.get("enabled", True)):
        print("[rerun] detached export explicitly disabled by eval.rerun.enabled=false")
        return None

    base_save_dir = base_save_dir.expanduser().resolve()
    base_save_dir.mkdir(parents=True, exist_ok=True)
    watchdog_script = Path(__file__).with_name("rerun_export_watchdog.py").resolve()
    if not watchdog_script.is_file():
        raise RuntimeError(f"Missing detached Rerun exporter: {watchdog_script}")

    ready_path = base_save_dir / f".rerun_export_watchdog.{os.getpid()}.ready.json"
    ready_path.unlink(missing_ok=True)
    log_path = base_save_dir / ".rerun_export_watchdog.log"
    command = [
        sys.executable,
        str(watchdog_script),
        "--root",
        str(base_save_dir),
        "--parent-pid",
        str(os.getpid()),
        "--fps",
        str(float(control_hz)),
        "--image-stride",
        str(int(rerun_cfg.get("image_stride", 6))),
        "--jpeg-quality",
        str(int(rerun_cfg.get("jpeg_quality", 75))),
        "--action-chunk-size",
        str(int(rerun_cfg.get("action_chunk_size", 30))),
        "--ready-file",
        str(ready_path),
    ]
    child_env = os.environ.copy()
    # The exporter only reads disk data. Prevent an accidental CUDA context
    # from contending with policy inference if a dependency probes hardware.
    child_env["CUDA_VISIBLE_DEVICES"] = ""
    child_env["PYTHONUNBUFFERED"] = "1"
    with open(log_path, "a", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=str(base_save_dir),
            stdin=subprocess.DEVNULL,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
            env=child_env,
        )

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if ready_path.is_file():
            try:
                ready = json.loads(ready_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                ready = {}
            if ready.get("state") == "ready" and process.poll() is None:
                print(
                    "[rerun] detached exporter armed before robot startup "
                    f"(pid={process.pid}; log={log_path})"
                )
                return process
        exit_code = process.poll()
        if exit_code is not None:
            raise RuntimeError(
                "Detached Rerun exporter failed before robot startup "
                f"(exit={exit_code}). Log tail:\n{_tail_text(log_path)}"
            )
        time.sleep(0.05)

    process.terminate()
    raise RuntimeError(
        "Detached Rerun exporter did not become ready before robot startup. "
        f"Log tail:\n{_tail_text(log_path)}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@dataclass
class Args:
    config_path: Annotated[str, tyro.conf.arg(aliases=("--config_path",))]
    """Primary robot/session configuration YAML file."""

    right_config_path: Optional[str] = None
    """Path to the right arm configuration YAML file (for bimanual operation)."""

    active_arm_side: Optional[Literal["left", "right", "both"]] = None
    """Bimanual mode only: execute the selected policy half, or both 7-DoF halves."""

    execution_mode: Optional[Literal["active_arm_hold", "shadow"]] = None
    """Bimanual mode only: ``shadow`` holds both arms; ``active_arm_hold`` executes active halves."""

    confirm_bimanual_clearance: bool = False
    """Required for active dual-arm motion after the operator verifies a clear, separated start pose."""

    num_rollouts: Annotated[int, tyro.conf.arg(aliases=("-n",))] = 1
    """How many rollouts to run in this session."""


def has_explicit_both_arm_cli_opt_in(args: Args) -> bool:
    """Return whether the user explicitly opted in to active both-arm motion.

    A YAML file may select ``both`` with ``shadow`` for an audit-only run, but
    it must not be able to turn an ordinary invocation into an active dual-arm
    run.  Keep this check on the parsed CLI values rather than the resolved
    config values for that reason.
    """
    return (
        args.active_arm_side == "both"
        and args.execution_mode == "active_arm_hold"
        and args.confirm_bimanual_clearance
    )


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def wait_for_camera_visual_preflight(
    camera_dict: Dict[str, Any],
    *,
    timeout_sec: float = 10.0,
    required_consecutive_frames: int = 2,
    expected_shape: Tuple[int, int, int] = (360, 640, 3),
    max_clipped_white_fraction: float = 0.15,
    min_mean_luma: float = 15.0,
    max_mean_luma: float = 240.0,
    max_luma_delta: float = 15.0,
    min_warmup_sec: float = 2.0,
    poll_sec: float = 0.25,
) -> None:
    """Verify usable, settled policy images before enabling any motor.

    The D405 wrist cameras can expose their first few UVC frames almost
    entirely white while auto-exposure settles.  Starting the robot and
    querying the policy from those frames makes the first action plan
    untrustworthy.  This check deliberately runs immediately after camera
    startup and before ``get_yam_robot`` can enable a motor.
    """
    if not camera_dict:
        return
    if timeout_sec <= 0 or required_consecutive_frames < 1:
        raise ValueError("camera preflight timeout and required frame count must be positive")

    started_at = time.monotonic()
    deadline = started_at + timeout_sec
    consecutive = {name: 0 for name in camera_dict}
    previous_luma: Dict[str, float] = {}
    last_stats: Dict[str, str] = {}

    while time.monotonic() < deadline:
        all_ready = True
        for name, camera in camera_dict.items():
            try:
                rgb, _ = camera.read()
                image = np.asarray(rgb)
                if image.shape != expected_shape or image.dtype != np.uint8:
                    raise RuntimeError(
                        f"expected uint8 {expected_shape}, got {image.dtype} {image.shape}"
                    )
                luma = float(
                    np.mean(
                        0.2126 * image[..., 0]
                        + 0.7152 * image[..., 1]
                        + 0.0722 * image[..., 2]
                    )
                )
                clipped_white = float(np.mean(np.all(image >= 250, axis=-1)))
                usable = (
                    min_mean_luma <= luma <= max_mean_luma
                    and clipped_white <= max_clipped_white_fraction
                )
                settled = name not in previous_luma or abs(luma - previous_luma[name]) <= max_luma_delta
                warmed_up = time.monotonic() - started_at >= min_warmup_sec
                consecutive[name] = consecutive[name] + 1 if usable and settled and warmed_up else 0
                previous_luma[name] = luma
                last_stats[name] = (
                    f"luma={luma:.1f}, clipped_white={clipped_white:.1%}, "
                    f"stable_frames={consecutive[name]}/{required_consecutive_frames}, "
                    f"warmup={'done' if warmed_up else 'pending'}"
                )
            except Exception as exc:  # noqa: BLE001 -- report camera-specific startup failure
                consecutive[name] = 0
                last_stats[name] = f"unusable: {exc}"
            if consecutive[name] < required_consecutive_frames:
                all_ready = False

        if all_ready:
            print(
                "[camera preflight] visual inputs ready before motor enable: "
                + "; ".join(f"{name} ({last_stats[name]})" for name in camera_dict)
            )
            return
        time.sleep(poll_sec)

    details = "; ".join(f"{name} ({last_stats.get(name, 'no frame')})" for name in camera_dict)
    raise RuntimeError(
        "Camera visual preflight failed before motor enable. "
        "Check camera exposure/positioning instead of running the policy on bad frames: "
        + details
    )


def _close_cameras_after_failed_startup(camera_dict: Dict[str, Any]) -> None:
    """Best-effort camera release before an environment has been constructed."""
    for camera in camera_dict.values():
        close = getattr(camera, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001 -- preserve preflight error
                logger.debug("Camera close after failed startup failed", exc_info=True)


def _close_failed_build_resources(
    *,
    camera_dict: Optional[Dict[str, Any]],
    camera_client: Optional[CameraClient],
    left_robot: Optional[Any],
    right_robot: Optional[Any],
) -> None:
    """Release resources if camera setup succeeded but robot setup did not.

    ``_build_env`` opens camera capture threads before it opens either CAN
    controller.  A motor-enable failure must not leave those threads or an
    already-enabled primary arm alive until interpreter teardown.  This runs
    only before ``RobotEnv`` exists, so each resource is closed directly and
    all cleanup is best-effort to preserve the original startup exception.
    """
    for robot in (right_robot, left_robot):
        close = getattr(robot, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001 -- preserve the startup failure
                logger.debug("Robot close after failed startup failed", exc_info=True)

    if camera_client is not None:
        close = getattr(camera_client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001 -- preserve the startup failure
                logger.debug("Camera client close after failed startup failed", exc_info=True)

    if camera_dict is not None:
        _close_cameras_after_failed_startup(camera_dict)


def _build_env(
    args: Args,
) -> Tuple[RobotEnv, Dict[str, Any], Optional[Dict[str, Any]], bool]:
    """Build cameras + robot(s) + RobotEnv from the launch configs.

    Camera source is decided by the ``eval.camera_server.enabled`` flag in the
    left config:

    * ``true``  -> connect to the long-lived camera server over ZMQ. RealSense
      devices are owned by that server; this process never opens them.
    * ``false`` -> open ``RealSenseCamera`` objects in-process (legacy path).
    """
    left_cfg = OmegaConf.to_container(OmegaConf.load(args.config_path), resolve=True)
    bimanual = args.right_config_path is not None
    right_cfg = (
        OmegaConf.to_container(OmegaConf.load(args.right_config_path), resolve=True)
        if bimanual else None
    )

    cam_server_cfg = ((left_cfg.get("eval") or {}).get("camera_server") or {})
    use_server = bool(cam_server_cfg.get("enabled", False))

    camera_dict = None
    camera_client = None
    if use_server:
        endpoint = str(cam_server_cfg.get("endpoint", "tcp://127.0.0.1:5555"))
        timeout_ms = int(cam_server_cfg.get("request_timeout_ms", 500))
        max_age = cam_server_cfg.get("max_frame_age_sec", 0.5)
        max_age = float(max_age) if max_age is not None else None
        print(f"[eval] Using camera server at {endpoint} (timeout={timeout_ms} ms)")
        camera_client = CameraClient(
            endpoint=endpoint,
            request_timeout_ms=timeout_ms,
            max_frame_age_sec=max_age,
        )
        if not camera_client.ping():
            raise RuntimeError(
                f"Camera server at {endpoint} did not respond to ping. "
                "Start it with scripts/start_camera_server.sh."
            )
    else:
        camera_cfg = left_cfg["sensors"]["cameras"]
        camera_names = ("left_camera", "front_camera", "right_camera")
        device_ids = [str(camera_cfg[name]["device_id"]) for name in camera_names]
        if all(device.startswith("/dev/") for device in device_ids):
            camera_dict = {
                name: (_DisabledCamera() if not bool(camera_cfg[name].get("enabled", True)) else V4L2Camera(device))
                for name, device in zip(camera_names, device_ids)
            }
            print(f"Using Jetson V4L2 RGB cameras: {device_ids} (disabled={[n for n in camera_names if not bool(camera_cfg[n].get('enabled', True))]})")
        else:
            ids = get_device_ids()
            print(f"Found {len(ids)} camera devices: {ids}")
            camera_dict = {
                "left_camera": RealSenseCamera(camera_cfg["left_camera"]["device_id"]),
                "front_camera": RealSenseCamera(camera_cfg["front_camera"]["device_id"]),
                "right_camera": RealSenseCamera(camera_cfg["right_camera"]["device_id"]),
            }

        preflight_cfg = (left_cfg.get("eval") or {}).get("camera_preflight") or {}
        if bool(preflight_cfg.get("enabled", True)):
            try:
                wait_for_camera_visual_preflight(
                    camera_dict,
                    timeout_sec=float(preflight_cfg.get("timeout_sec", 10.0)),
                    required_consecutive_frames=int(
                        preflight_cfg.get("required_consecutive_frames", 2)
                    ),
                    expected_shape=tuple(preflight_cfg.get("expected_shape", (360, 640, 3))),
                    max_clipped_white_fraction=float(
                        preflight_cfg.get("max_clipped_white_fraction", 0.15)
                    ),
                    min_warmup_sec=float(preflight_cfg.get("min_warmup_sec", 2.0)),
                )
            except Exception:
                _close_cameras_after_failed_startup(camera_dict)
                raise

    left_robot = None
    right_robot = None
    try:
        left_robot_cfg = left_cfg["robot"]
        if isinstance(left_robot_cfg.get("config"), str):
            left_robot_cfg["config"] = OmegaConf.to_container(
                OmegaConf.load(left_robot_cfg["config"]), resolve=True
            )
        print(f"Opening primary YAM robot on CAN channel: {left_robot_cfg.get('channel', '<unspecified>')}")
        left_robot = instantiate_from_dict(left_robot_cfg)

        if bimanual:
            right_robot_cfg = right_cfg["robot"]
            if isinstance(right_robot_cfg.get("config"), str):
                right_robot_cfg["config"] = OmegaConf.to_container(
                    OmegaConf.load(right_robot_cfg["config"]), resolve=True
                )
            print(f"Opening secondary YAM robot on CAN channel: {right_robot_cfg.get('channel', '<unspecified>')}")
            right_robot = instantiate_from_dict(right_robot_cfg)
            robot = BimanualRobot(left_robot, right_robot)
        else:
            robot = left_robot

        env = RobotEnv(
            robot,
            control_rate_hz=left_cfg.get("hz", 30),
            camera_dict=camera_dict,
            camera_client=camera_client,
        )
    except Exception:
        _close_failed_build_resources(
            camera_dict=camera_dict,
            camera_client=camera_client,
            left_robot=left_robot,
            right_robot=right_robot,
        )
        raise
    return env, left_cfg, right_cfg, bimanual


# ---------------------------------------------------------------------------
# Inner loop
# ---------------------------------------------------------------------------


def _coerce_bimanual_delta_limit(
    value: Any,
    *,
    name: str,
    state_dim: int,
) -> np.ndarray:
    """Normalize a scalar or native 14-D dual-arm action safety limit."""
    if value is None:
        raise ValueError(
            f"Active both-arm execution requires eval.bimanual.{name} in the "
            "primary configuration."
        )
    result = np.asarray(value, dtype=np.float32)
    if result.ndim == 0:
        result = np.full(state_dim, float(result), dtype=np.float32)
    else:
        result = result.reshape(-1)
        if result.shape != (state_dim,):
            raise ValueError(
                f"eval.bimanual.{name} must be a positive scalar or {state_dim}-D "
                f"[left(7), right(7)] vector, got {result.shape}"
            )
    if not np.isfinite(result).all() or np.any(result <= 0.0):
        raise ValueError(
            f"eval.bimanual.{name} must contain only finite positive values"
        )
    return result


@dataclass(frozen=True)
class BimanualActiveArmHoldMask:
    """Safely select which halves of a 14-DoF BimanualYAM plan may execute.

    The released MolmoAct checkpoint was trained with state/action order
    ``[left arm (7), right arm (7)]``.  It must therefore receive feedback
    from *both* physical arms.  The model is still free to predict both
    halves.  In single-arm mode this adapter replaces the inactive half with
    the latest encoder feedback just before each command is sent. Holding from
    fresh feedback (rather than an old start pose) avoids fighting a small
    passive displacement of the inactive arm. In explicit ``both`` mode, a
    native 14-D target is rejected on a large discontinuity and otherwise
    rate-bounded around fresh feedback before it is sent.

    ``shadow`` is deliberately stronger: it replaces both halves with live
    feedback so that a policy rollout can be inspected without intentionally
    changing either arm's pose.
    """

    active_arm_side: Literal["left", "right", "both"]
    execution_mode: Literal["active_arm_hold", "shadow"] = "active_arm_hold"

    state_dim: int = 14
    arm_dim: int = 7
    # Required only for active both-arm execution.  A scalar expands to all
    # fields; a 14-D vector lets gripper aperture use a different bound from
    # arm-joint radians.
    both_arm_max_delta: Optional[Any] = None
    both_arm_reject_delta: Optional[Any] = None

    def __post_init__(self) -> None:
        if self.active_arm_side not in {"left", "right", "both"}:
            raise ValueError(
                "active_arm_side must be 'left', 'right', or 'both', got "
                f"{self.active_arm_side!r}"
            )
        if self.execution_mode not in {"active_arm_hold", "shadow"}:
            raise ValueError(
                "execution_mode must be 'active_arm_hold' or 'shadow', got "
                f"{self.execution_mode!r}"
            )
        if self.active_arm_side == "both" and self.execution_mode == "active_arm_hold":
            max_delta = _coerce_bimanual_delta_limit(
                self.both_arm_max_delta,
                name="both_arm_max_delta",
                state_dim=self.state_dim,
            )
            reject_delta = _coerce_bimanual_delta_limit(
                self.both_arm_reject_delta,
                name="both_arm_reject_delta",
                state_dim=self.state_dim,
            )
            if np.any(reject_delta < max_delta):
                raise ValueError(
                    "eval.bimanual.both_arm_reject_delta must be at least "
                    "both_arm_max_delta for every field"
                )
            object.__setattr__(self, "both_arm_max_delta", max_delta)
            object.__setattr__(self, "both_arm_reject_delta", reject_delta)

    @property
    def active_slice(self) -> slice:
        if self.active_arm_side == "both":
            return slice(0, self.state_dim)
        return (
            slice(0, self.arm_dim)
            if self.active_arm_side == "left"
            else slice(self.arm_dim, self.state_dim)
        )

    @property
    def inactive_slice(self) -> slice:
        if self.active_arm_side == "both":
            # An empty slice is convenient for callers that need to apply an
            # inactive-arm operation without special casing the both-arm mode.
            return slice(0, 0)
        return (
            slice(self.arm_dim, self.state_dim)
            if self.active_arm_side == "left"
            else slice(0, self.arm_dim)
        )

    @property
    def active_gripper_index(self) -> int:
        if self.active_arm_side == "both":
            raise ValueError(
                "both-arm execution has two active grippers; use active_grippers"
            )
        return self.active_slice.stop - 1

    @property
    def active_grippers(self) -> Tuple[Tuple[str, int], ...]:
        """Named active gripper action fields for logging and audit output."""
        if self.active_arm_side == "both":
            return (("left", self.arm_dim - 1), ("right", self.state_dim - 1))
        return ((self.active_arm_side, self.active_gripper_index),)

    @property
    def execution_summary(self) -> str:
        if self.execution_mode == "shadow":
            return "both arms held from live encoder feedback (shadow)"
        if self.active_arm_side == "both":
            return "both native policy halves enabled with delta guard"
        return f"{self.active_arm_side} policy half enabled; opposite arm held from live feedback"

    def manifest_metadata(self) -> Dict[str, Any]:
        """Return the execution contract recorded alongside a rollout."""
        result: Dict[str, Any] = {
            "model_state_order": "left(0:7), right(7:14)",
            "active_arm_side": self.active_arm_side,
            "execution_mode": self.execution_mode,
            "commanded_policy_halves": (
                ["left", "right"]
                if self.execution_mode == "active_arm_hold"
                and self.active_arm_side == "both"
                else ([self.active_arm_side] if self.execution_mode == "active_arm_hold" else [])
            ),
        }
        if self.active_arm_side == "both" and self.execution_mode == "active_arm_hold":
            result["both_arm_delta_guard"] = {
                "max_delta": np.asarray(self.both_arm_max_delta).tolist(),
                "reject_delta": np.asarray(self.both_arm_reject_delta).tolist(),
                "reference": "fresh_encoder_feedback_each_tick",
            }
        return result

    def validate_state(self, state: Any, *, name: str = "state") -> np.ndarray:
        result = np.asarray(state, dtype=np.float32).reshape(-1)
        if result.shape != (self.state_dim,):
            raise ValueError(
                f"{name} must be a live {self.state_dim}-D bimanual vector "
                f"[left(7), right(7)], got shape {result.shape}"
            )
        if not np.isfinite(result).all():
            raise ValueError(f"{name} contains non-finite values")
        return result

    def command_target(self, policy_action: Any, measured_state: Any) -> np.ndarray:
        """Return the 14-D target safe to send on this control tick.

        ``measured_state`` must be sampled immediately before this target is
        sent.  It is intentionally not a cached rollout-start state.
        """
        measured = self.validate_state(measured_state, name="measured_state")
        action = self.validate_state(policy_action, name="policy_action")
        command = measured.copy()
        if self.execution_mode == "active_arm_hold":
            if self.active_arm_side == "both":
                # Grippers are normalized apertures, not motor coordinates.
                # Validate both raw model values before any half reaches the
                # serial left-then-right dispatcher.
                for arm_label, gripper_index in self.active_grippers:
                    aperture = float(action[gripper_index])
                    if not -0.05 <= aperture <= 1.05:
                        raise RuntimeError(
                            "Bimanual action guard rejected an invalid "
                            f"{arm_label} gripper aperture {aperture:.4f}; "
                            "no target was sent."
                        )

                delta = action - measured
                reject_delta = np.asarray(self.both_arm_reject_delta, dtype=np.float32)
                rejected = np.flatnonzero(np.abs(delta) > reject_delta)
                if rejected.size:
                    details = ", ".join(
                        f"{int(index)}: |{float(delta[index]):.3f}|>"
                        f"{float(reject_delta[index]):.3f}"
                        for index in rejected[:4]
                    )
                    suffix = "" if rejected.size <= 4 else f" (+{rejected.size - 4} more)"
                    raise RuntimeError(
                        "Bimanual action guard rejected a discontinuous "
                        f"absolute target ({details}{suffix}); no target was sent."
                    )
                max_delta = np.asarray(self.both_arm_max_delta, dtype=np.float32)
                command = measured + np.clip(delta, -max_delta, max_delta)
                return command.astype(np.float32, copy=False)
            command[self.active_slice] = action[self.active_slice]
        return command


def resolve_bimanual_execution_mask(
    *,
    bimanual: bool,
    active_arm_side: Optional[str],
    execution_mode: Optional[str],
    both_arm_active_cli_confirmed: bool = False,
    both_arm_max_delta: Any = None,
    both_arm_reject_delta: Any = None,
) -> Optional[BimanualActiveArmHoldMask]:
    """Build the explicit safe execution adapter before opening motors.

    There is intentionally no implicit "execute both arms" setting for this
    physical launcher. A full 14-D policy output without an explicit selector
    is too easy to send to the wrong physical side. ``both`` additionally
    requires an explicit execution mode, and active both-arm motion requires
    a separate CLI-origin confirmation from ``main``. Configuration defaults
    may therefore select only the safe both-arm ``shadow`` path.
    """
    if not bimanual:
        raise ValueError(
            "MolmoAct2-BimanualYAM physical execution requires --right-config-path "
            "and live 14-D [left(7), right(7)] feedback. The old 7-D "
            "zero-padding/action-cropping adapter is disabled."
        )

    if active_arm_side is None:
        raise ValueError(
            "Bimanual physical execution requires --active-arm-side {left,right,both}. "
            "For left/right the other arm is held at live encoder feedback."
        )
    side = str(active_arm_side).lower()
    if side == "both" and execution_mode is None:
        raise ValueError(
            "--active-arm-side both requires an explicit --execution-mode "
            "{active_arm_hold,shadow}; use active_arm_hold only when both "
            "physical arms are clear to move."
        )
    mode = "active_arm_hold" if execution_mode is None else str(execution_mode)
    if side == "both" and mode == "active_arm_hold" and not both_arm_active_cli_confirmed:
        raise ValueError(
            "Active both-arm motion requires these explicit CLI flags: "
            "--active-arm-side both --execution-mode active_arm_hold "
            "--confirm-bimanual-clearance. "
            "A config file may select both arms only with execution_mode: shadow."
        )
    return BimanualActiveArmHoldMask(
        active_arm_side=side,
        execution_mode=mode,
        both_arm_max_delta=both_arm_max_delta,
        both_arm_reject_delta=both_arm_reject_delta,
    )


def validate_bimanual_model_arm_order(
    left_cfg: Dict[str, Any], right_cfg: Dict[str, Any]
) -> None:
    """Require an explicit config declaration for checkpoint state ordering.

    The primary config is not merely an arbitrary first robot: it supplies
    state fields 0..6 to the checkpoint and must be the physical left arm.
    Reversing the two config paths would make a right-arm action look like a
    left-arm action to the model even though every vector still has shape 14.
    """
    declared_left = left_cfg.get("model_arm_side")
    declared_right = right_cfg.get("model_arm_side")
    if declared_left != "left" or declared_right != "right":
        raise ValueError(
            "Bimanual configs must explicitly declare model_arm_side: left for "
            "--config-path and model_arm_side: right for --right-config-path. "
            f"Got primary={declared_left!r}, secondary={declared_right!r}."
        )


def move_to_rollout_start(
    env: RobotEnv,
    *,
    bimanual: bool,
    left_cfg: Dict[str, Any],
    right_cfg: Optional[Dict[str, Any]],
    execution_mask: Optional[BimanualActiveArmHoldMask],
) -> None:
    """Move to a configured start only when it cannot violate the hold mask."""
    if execution_mask is not None:
        logger.info(
            "Skipping configured start pose in %s mode; both arms begin from live feedback",
            execution_mask.execution_mode,
        )
        return
    move_to_start_position(env, bimanual, left_cfg, right_cfg)


def dynamic_smoothing(env: RobotEnv, target_joints: np.ndarray) -> Dict[str, Any]:
    """Apply one policy target on exactly one control tick.

    The post-step record only needs measured robot state.  Do not perform a
    second three-camera snapshot here: images are already present in
    ``obs_pre`` and rereading V4L2 streams would couple control rate to camera
    rate.  The policy already provides a 30-target trajectory; rapidly
    overwriting one command slot with synthetic interpolation does not create
    a trajectory at the motor and made rollouts appear to run at a few Hz.
    """
    env.step_command_only(target_joints)
    return env.get_robot_state()


def run_one_rollout(
    env: RobotEnv,
    policy: MolmoAct,
    saver: EvalRolloutSaver,
    instruction: str,
    rollout_idx: int,
    num_rollouts: int,
    max_steps: int,
    live_view: LiveCameraView,
    execution_mask: Optional[BimanualActiveArmHoldMask] = None,
) -> RolloutOutcome:
    """Execute one rollout and buffer per-step observations into ``saver``.

    End conditions:

    * ``cv2`` keypress ``y`` -> success (labeled)
    * ``cv2`` keypress ``n`` -> failure (labeled)
    * ``cv2`` keypress ``q`` -> quit (no label; rollout stays in ``eval/``)
    * step >= ``max_steps`` -> timeout (stdin prompt afterwards)

    Does NOT flush the saver — the caller does that so the Ctrl-C path can
    also flush the partial buffer.
    """
    action_chunk: Optional[np.ndarray] = None
    chunk_index = 0
    policy_chunk_index = -1
    policy_inference_sec = float("nan")
    policy_inference_metadata: Dict[str, Any] = {}
    chunk_started_at: Optional[float] = None

    def infer(obs_for_policy: Dict[str, Any]) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        input_dict = policy.prepare_input(obs_for_policy, instruction)
        t0 = time.perf_counter()
        policy_result = policy.inference(input_dict)
        result = policy_result["actions"]
        inference_sec = time.perf_counter() - t0
        log_collect_demos(
            f"Policy inference {inference_sec:.3f}s ({len(result)} actions)",
            "data_info",
        )
        reproducibility = policy_result.get("reproducibility", {})
        if not isinstance(reproducibility, dict):
            reproducibility = {"metadata_error": "policy returned non-mapping reproducibility metadata"}
        return np.asarray(result, dtype=np.float32), inference_sec, reproducibility

    for step in range(max_steps):
        # MolmoAct's YAM checkpoint predicts *absolute* joint targets.  A
        # prediction prefetched before the current chunk finishes is therefore
        # stale at the next boundary and can command the arm back toward its
        # old pose.  Always replan from a fresh observation instead.  The
        # I2RT motor-control thread continuously holds the previous target
        # while inference runs, so this does not starve the motor watchdog.
        obs_pre = env.get_obs()
        if action_chunk is None or chunk_index >= len(action_chunk):
            if action_chunk is not None and chunk_started_at is not None:
                execution_sec = time.monotonic() - chunk_started_at
                logger.info(
                    "Completed %d action commands in %.3fs (%.2f Hz)",
                    len(action_chunk),
                    execution_sec,
                    len(action_chunk) / max(execution_sec, 1e-6),
                )
            action_chunk, policy_inference_sec, policy_inference_metadata = infer(obs_pre)
            expected_dofs = env.robot().num_dofs()
            if execution_mask is not None:
                # Do this at every inference boundary as well as per tick so
                # a bad/cropped policy response cannot reach either arm.
                execution_mask.validate_state(
                    obs_pre["joint_positions"], name="policy input state"
                )
            if (
                action_chunk.ndim != 2
                or len(action_chunk) == 0
                or action_chunk.shape[1] != expected_dofs
                or not np.isfinite(action_chunk).all()
            ):
                raise RuntimeError(
                    "Policy returned an invalid action chunk: "
                    f"shape={action_chunk.shape}, expected (N, {expected_dofs}) finite values"
            )
            chunk_index = 0
            policy_chunk_index += 1
            saver.add_policy_action_chunk(
                start_step=step,
                actions=action_chunk,
                inference_sec=policy_inference_sec,
                metadata=policy_inference_metadata,
            )
            logger.info(
                "Fresh action plan at rollout step %d: %d absolute-pose actions",
                step + 1,
                len(action_chunk),
            )
            # The YAM checkpoint emits a normalized aperture: 0=closed,
            # 1=open. The robot-side mapper is solely responsible for mapping
            # this value into its calibrated physical gripper coordinate.
            grippers = (
                execution_mask.active_grippers
                if execution_mask is not None
                else (("primary", expected_dofs - 1),)
            )
            for arm_label, gripper_index in grippers:
                current_gripper = float(
                    np.asarray(obs_pre["joint_positions"])[gripper_index]
                )
                gripper_targets = action_chunk[:, gripper_index]
                logger.info(
                    "Gripper targets (normalized aperture, %s arm%s): "
                    "current=%.4f, first=%.4f, last=%.4f, range=[%.4f, %.4f]",
                    arm_label,
                    (
                        f", {execution_mask.execution_summary}"
                        if execution_mask is not None
                        else ""
                    ),
                    current_gripper,
                    float(gripper_targets[0]),
                    float(gripper_targets[-1]),
                    float(np.min(gripper_targets)),
                    float(np.max(gripper_targets)),
                )
            chunk_started_at = time.monotonic()

        action_index_in_chunk = chunk_index
        policy_action = action_chunk[action_index_in_chunk]
        chunk_index += 1
        action = (
            execution_mask.command_target(policy_action, obs_pre["joint_positions"])
            if execution_mask is not None
            else policy_action
        )
        obs_post = dynamic_smoothing(env, action) or obs_pre

        saver.add_step(
            obs_pre=obs_pre,
            obs_post=obs_post,
            action=action,
            policy_chunk_index=policy_chunk_index,
            policy_action_index=action_index_in_chunk,
            policy_inference_sec=(
                policy_inference_sec if action_index_in_chunk == 0 else float("nan")
            ),
        )

        key = live_view.update(
            obs=obs_pre,
            rollout_idx=rollout_idx,
            num_rollouts=num_rollouts,
            step=step + 1,
            max_steps=max_steps,
            instruction=instruction,
        )
        if key == "y":
            return RolloutOutcome(end_reason="success", last_step=step + 1)
        if key == "n":
            return RolloutOutcome(end_reason="failure", last_step=step + 1)
        if key == "q":
            return RolloutOutcome(end_reason="quit", last_step=step + 1)

    return RolloutOutcome(end_reason="timeout", last_step=max_steps)


# ---------------------------------------------------------------------------
# Session driver
# ---------------------------------------------------------------------------


def with_active_arm_instruction(
    instruction: str,
    active_arm_side: Optional[str],
) -> str:
    """Make the physically enabled arm set unambiguous in the language."""
    if active_arm_side is None:
        return instruction
    side = str(active_arm_side).lower()
    if side not in {"left", "right", "both"}:
        raise ValueError(
            "active_arm_side must be 'left', 'right', or 'both', got "
            f"{side!r}"
        )
    lower = instruction.lower()
    if side == "both":
        has_left = "left arm" in lower
        has_right = "right arm" in lower
        if has_left != has_right:
            named_side = "left" if has_left else "right"
            raise ValueError(
                "Both-arm execution enables both physical arms, but the instruction "
                f"only names the {named_side} arm. Rewrite it to describe both arms."
            )
        if "both arms" in lower or (has_left and has_right):
            return instruction
        return f"{instruction.rstrip('.')} using both arms."
    other_side = "right" if side == "left" else "left"
    if f"{other_side} arm" in lower:
        raise ValueError(
            f"This execution mode controls only the {side} arm, but the instruction "
            f"asks for the {other_side} arm."
        )
    if f"{side} arm" in lower:
        return instruction
    return f"{instruction.rstrip('.')} using the {side} arm."


def with_active_single_arm_instruction(
    instruction: str,
    single_arm_side: Optional[str],
) -> str:
    """Backward-compatible alias for the single-arm instruction helper."""
    return with_active_arm_instruction(instruction, single_arm_side)


def run_session(
    env: RobotEnv,
    policy: MolmoAct,
    left_cfg: Dict[str, Any],
    right_cfg: Optional[Dict[str, Any]],
    bimanual: bool,
    num_rollouts: int,
    execution_mask: Optional[BimanualActiveArmHoldMask] = None,
    *,
    seed_plan: Optional[RolloutSeedPlan] = None,
    primary_config_path: Optional[str] = None,
    secondary_config_path: Optional[str] = None,
    process_seed_metadata: Optional[Dict[str, Any]] = None,
) -> List[Path]:
    """Drive ``num_rollouts`` rollouts; convert the labeled set to LeRobot at the end.

    Catches ``KeyboardInterrupt`` so an in-progress rollout still gets flushed
    (as incomplete, with ``err.md``) and any rollouts already labeled in this
    session are still converted. Returns saved rollout paths for post-shutdown
    diagnostics such as Rerun export.
    """
    storage = left_cfg["storage"]
    base_save_dir = Path(storage["base_dir"]) / "data" / storage["task_directory"]
    max_steps = int(left_cfg.get("max_steps", 1000))
    last_prompt = storage.get("language_instruction") or ""
    seed_plan = seed_plan or RolloutSeedPlan(None)

    # Make the controlled side explicit before the first physical command.
    # In the bimanual path the model still sees the full 14-D state, while the
    # execution mask blocks motion from the inactive predicted half.
    local_policy_cfg = ((left_cfg.get("eval") or {}).get("local") or {})
    single_arm_side = local_policy_cfg.get("single_arm_side") if not bimanual else None
    instruction_side = (
        execution_mask.active_arm_side
        if execution_mask is not None
        else single_arm_side
    )
    eval_cfg = left_cfg.get("eval") or {}
    cam_srv_cfg = eval_cfg.get("camera_server") or {}
    pub_endpoint = cam_srv_cfg.get("pub_endpoint") if cam_srv_cfg.get("enabled") else None
    live_view = LiveCameraView(
        enabled=bool(eval_cfg.get("live_view_enabled", True)),
        pub_endpoint=pub_endpoint,
        recv_timeout_ms=int(cam_srv_cfg.get("recv_timeout_ms", 100)),
    )

    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    labeled_rollouts: List[Path] = []
    saved_rollouts: List[Path] = []
    saver: Optional[EvalRolloutSaver] = None
    outcome: Optional[RolloutOutcome] = None

    try:
        for rollout_idx in range(num_rollouts):
            move_to_rollout_start(
                env,
                bimanual=bimanual,
                left_cfg=left_cfg,
                right_cfg=right_cfg,
                execution_mask=execution_mask,
            )
            instruction = with_active_arm_instruction(
                prompt_instruction(rollout_idx, num_rollouts, last_prompt),
                instruction_side,
            )
            last_prompt = instruction

            # Reset a local policy's generator stream before the first model
            # query. Remote policies simply report that no generator contract
            # exists, which is also retained in the manifest.
            rollout_seed = seed_plan.rollout_seed(rollout_idx)
            policy_rollout_metadata: Dict[str, Any] = {}
            begin_rollout = getattr(policy, "begin_rollout", None)
            if callable(begin_rollout):
                returned_metadata = begin_rollout(rollout_seed)
                if isinstance(returned_metadata, dict):
                    policy_rollout_metadata = returned_metadata

            rollout_manifest = build_rollout_manifest(
                instruction=instruction,
                rollout_index=rollout_idx,
                seed_plan=seed_plan,
                policy=policy,
                primary_config=left_cfg,
                primary_config_path=primary_config_path,
                secondary_config=right_cfg if bimanual else None,
                secondary_config_path=secondary_config_path if bimanual else None,
                process_seed_metadata=process_seed_metadata,
                project_root=Path(__file__).resolve().parents[2],
            )
            if execution_mask is not None:
                rollout_manifest["execution"] = execution_mask.manifest_metadata()
            if policy_rollout_metadata:
                rollout_manifest["policy"]["rollout"] = policy_rollout_metadata

            rollout_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            rollout_dir = base_save_dir / "eval" / rollout_timestamp
            saver = EvalRolloutSaver(
                rollout_dir=rollout_dir,
                instruction=instruction,
                max_workers=int(storage.get("saver_max_workers", 2)),
                png_compress_level=int(storage.get("png_compress_level", 1)),
                rollout_manifest=rollout_manifest,
            )

            print(f"\n--- Rollout {rollout_idx + 1}/{num_rollouts} ---")
            print(f"  instruction: {instruction}")
            print(f"  rollout_dir: {rollout_dir}")

            outcome = run_one_rollout(
                env=env,
                policy=policy,
                saver=saver,
                instruction=instruction,
                rollout_idx=rollout_idx,
                num_rollouts=num_rollouts,
                max_steps=max_steps,
                live_view=live_view,
                execution_mask=execution_mask,
            )

            saver.flush()
            label = resolve_label(outcome)
            final_rollout_dir = rollout_dir
            if label is not None:
                new_path = move_rollout(rollout_dir, label, base_save_dir)
                labeled_rollouts.append(new_path)
                final_rollout_dir = new_path
                print(f"  -> labeled '{label}': {new_path}")
            else:
                print(f"  -> kept in eval/: {rollout_dir}")

            saved_rollouts.append(final_rollout_dir)

            saver = None
            outcome = None
    except KeyboardInterrupt:
        print("\n[interrupt] Ctrl-C received — saving incomplete rollout, then converting...")
        if saver is not None:
            try:
                saver.flush()
                saver.write_err(
                    reason="KeyboardInterrupt",
                    step=outcome.last_step if outcome else saver.num_steps,
                )
                print(f"  -> incomplete rollout saved: {saver.rollout_dir}")
                saved_rollouts.append(saver.rollout_dir)
            except Exception as exc:  # noqa: BLE001 — best-effort cleanup
                logger.exception("Failed to flush incomplete rollout: %s", exc)
    finally:
        live_view.close()
        _convert_if_any(labeled_rollouts, base_save_dir, session_timestamp, left_cfg)

    return saved_rollouts


def _convert_if_any(
    labeled_rollouts: List[Path],
    base_save_dir: Path,
    session_timestamp: str,
    left_cfg: Dict[str, Any],
) -> None:
    """Best-effort LeRobot conversion of this session's labeled rollouts."""
    if not labeled_rollouts:
        print("\n[session] No labeled rollouts this session — nothing to convert.")
        return

    lerobot_cfg = left_cfg.get("lerobot", {}) or {}
    output_dir = base_save_dir / "eval_lerobot_v30" / session_timestamp
    print(
        f"\n[session] Converting {len(labeled_rollouts)} labeled rollouts "
        f"to LeRobot v3.0 at {output_dir} ..."
    )
    try:
        convert_session_to_lerobot(
            session_rollout_dirs=labeled_rollouts,
            output_dir=output_dir,
            fps=int(lerobot_cfg.get("fps", left_cfg.get("hz", 30))),
            robot_type=str(lerobot_cfg.get("robot_type", "molmoact_dual_arm")),
            repo_id=str(lerobot_cfg.get("hf_repo_id", "local/eval_session")),
            action_mode=str(lerobot_cfg.get("action_mode", "next_joint_fields")),
            vcodec=str(lerobot_cfg.get("vcodec", "libsvtav1")),
            sanitize_online_viz_meta=bool(lerobot_cfg.get("sanitize_online_viz_meta", True)),
            image_writer_processes=int(lerobot_cfg.get("image_writer_processes", 0)),
            image_writer_threads=int(lerobot_cfg.get("image_writer_threads", 0)),
            parallel_encoding=bool(lerobot_cfg.get("parallel_encoding", True)),
        )
    except Exception as exc:  # noqa: BLE001 — keep raw rollouts even if conversion fails
        logger.exception("LeRobot conversion failed: %s", exc)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    # Fallback only. The explicit ``finally`` below is the normal shutdown
    # path because Python waits for the robot's non-daemon server thread before
    # running atexit handlers.
    atexit.register(_shutdown_runtime)

    args = tyro.cli(Args)
    if args.num_rollouts < 1:
        raise SystemExit("--num_rollouts must be >= 1")

    # Load the policy before powering/starting the robot control thread.  The
    # model may take seconds to load; leaving an active CAN control loop running
    # during that window makes intermittent motor replies much more likely.
    raw_left_cfg = OmegaConf.to_container(OmegaConf.load(args.config_path), resolve=True)
    eval_cfg = raw_left_cfg.get("eval") or {}
    reproducibility_cfg = eval_cfg.get("reproducibility") or {}
    seed_plan = RolloutSeedPlan(reproducibility_cfg.get("seed"))
    process_seed_metadata = configure_process_seed(
        seed_plan,
        deterministic_algorithms=bool(
            reproducibility_cfg.get("deterministic_algorithms", False)
        ),
    )
    if seed_plan.enabled:
        print(
            "[reproducibility] deterministic per-query policy generators armed "
            f"(base_seed={seed_plan.base_seed}, strategy=per_rollout_per_query_v1)"
        )
    else:
        print(
            "[reproducibility] no eval.reproducibility.seed configured; "
            "policy action sampling remains stochastic"
        )
    raw_right_cfg = (
        OmegaConf.to_container(OmegaConf.load(args.right_config_path), resolve=True)
        if args.right_config_path is not None
        else None
    )
    bimanual_requested = raw_right_cfg is not None
    bimanual_cfg = eval_cfg.get("bimanual") or {}
    active_arm_side = (
        args.active_arm_side
        if args.active_arm_side is not None
        else bimanual_cfg.get("active_arm_side")
    )
    execution_mode = (
        args.execution_mode
        if args.execution_mode is not None
        else bimanual_cfg.get("execution_mode")
    )
    execution_mask = resolve_bimanual_execution_mask(
        bimanual=bimanual_requested,
        active_arm_side=active_arm_side,
        execution_mode=execution_mode,
        both_arm_active_cli_confirmed=has_explicit_both_arm_cli_opt_in(args),
        both_arm_max_delta=bimanual_cfg.get("both_arm_max_delta"),
        both_arm_reject_delta=bimanual_cfg.get("both_arm_reject_delta"),
    )
    assert raw_right_cfg is not None  # established by the resolver above
    validate_bimanual_model_arm_order(raw_left_cfg, raw_right_cfg)
    print(
        "[bimanual] model state order: left(0:7), right(7:14); "
        f"execution={execution_mask.execution_mode}, "
        f"active={execution_mask.active_arm_side}; "
        f"{execution_mask.execution_summary}"
    )

    storage_cfg = raw_left_cfg.get("storage") or {}
    try:
        base_save_dir = Path(storage_cfg["base_dir"]) / "data" / str(storage_cfg["task_directory"])
    except KeyError as exc:
        raise RuntimeError("The rollout config must define storage.base_dir and storage.task_directory") from exc
    rerun_cfg = eval_cfg.get("rerun") or {}
    _start_rerun_export_watchdog(
        base_save_dir=base_save_dir,
        rerun_cfg=rerun_cfg,
        control_hz=float(raw_left_cfg.get("hz", 30)),
    )

    mode = eval_cfg.get("mode", "server")
    if mode == "local":
        policy = MolmoActLocal(**(eval_cfg.get("local") or {}))
    elif mode == "server":
        policy = MolmoAct(server=eval_cfg.get("molmoact_server"))
    else:
        raise SystemExit(f"eval.mode must be 'server' or 'local', got {mode!r}")

    env, left_cfg, right_cfg, bimanual = _build_env(args)

    global _env, _bimanual, _left_cfg, _right_cfg, _bimanual_execution_mask
    _env = env
    _bimanual = bimanual
    _left_cfg = left_cfg
    _right_cfg = right_cfg
    _bimanual_execution_mask = execution_mask

    saved_rollouts: List[Path] = []
    try:
        move_to_rollout_start(
            env,
            bimanual=bimanual,
            left_cfg=left_cfg,
            right_cfg=right_cfg,
            execution_mask=execution_mask,
        )

        print(f"Launching robot: {env.robot().__class__.__name__}")
        print(f"Control loop: {left_cfg.get('hz', 30)} Hz")
        print(
            f"Rollouts this session: {args.num_rollouts}, "
            f"max_steps: {left_cfg.get('max_steps', 1000)}"
        )

        saved_rollouts = run_session(
            env=env,
            policy=policy,
            left_cfg=left_cfg,
            right_cfg=right_cfg,
            bimanual=bimanual,
            num_rollouts=args.num_rollouts,
            execution_mask=execution_mask,
            seed_plan=seed_plan,
            primary_config_path=args.config_path,
            secondary_config_path=args.right_config_path,
            process_seed_metadata=process_seed_metadata,
        )
    finally:
        # Close robot/CAN and V4L2 resources before any post-rollout
        # visualization work. A hung exporter must never retain the controller
        # or cameras after motion is over.
        _shutdown_runtime()

    if saved_rollouts:
        print(
            "[rerun] raw rollouts are handed to the detached exporter after this "
            "launcher exits; each directory will receive an atomic rollout.rrd."
        )


if __name__ == "__main__":
    main()
