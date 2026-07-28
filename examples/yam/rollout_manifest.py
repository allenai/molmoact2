"""Deterministic-seed and provenance helpers for physical YAM rollouts.

This module deliberately has no camera, CAN, or robot imports.  It is used by
the launcher before a rollout starts to make the policy's stochastic action
sampling reproducible and to write a durable description of exactly what was
asked to run.  The manifest records *configured* hardware identities; it does
not probe or enable hardware.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np


MANIFEST_SCHEMA_VERSION = 1
SEED_STRATEGY = "molmoact_yam_per_rollout_per_query_v1"
_MAX_SEED = (1 << 63) - 1
_SENSITIVE_KEY_PARTS = (
    "token",
    "password",
    "secret",
    "api_key",
    "apikey",
    "authorization",
    "credential",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_safe(value: Any) -> Any:
    """Convert ordinary config/runtime values to deterministic JSON values."""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        # JSON's NaN/Infinity spellings are not portable provenance data.
        return value if np.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    # Keep this module independent of torch while still handling torch dtypes,
    # devices, and other small runtime objects safely.
    return str(value)


def _redact_config(value: Any, *, key: str = "") -> Any:
    """Return a JSON-safe config copy with credentials redacted."""
    normalized_key = key.lower().replace("-", "_")
    if any(part in normalized_key for part in _SENSITIVE_KEY_PARTS):
        return "<redacted>"
    if isinstance(value, Mapping):
        return {
            str(child_key): _redact_config(child_value, key=str(child_key))
            for child_key, child_value in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_redact_config(item, key=key) for item in value]
    return _json_safe(value)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> Optional[str]:
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _normalize_seed(seed: Optional[int]) -> Optional[int]:
    if seed is None:
        return None
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError(f"reproducibility.seed must be an integer or null, got {seed!r}")
    seed = int(seed)
    if seed < 0 or seed > _MAX_SEED:
        raise ValueError(f"reproducibility.seed must be in [0, {_MAX_SEED}], got {seed}")
    return seed


def _derive_seed(base_seed: int, stream: str, index: int) -> int:
    if index < 0:
        raise ValueError(f"seed index must be non-negative, got {index}")
    payload = f"{SEED_STRATEGY}\0{base_seed}\0{stream}\0{index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & _MAX_SEED


@dataclass(frozen=True)
class RolloutSeedPlan:
    """Stable namespacing for one session's policy random streams.

    A policy receives a distinct generator seed for each inference request,
    rather than sharing an implicit global CUDA RNG.  Therefore a warmup or a
    logging operation cannot silently change the first physical plan.
    """

    base_seed: Optional[int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "base_seed", _normalize_seed(self.base_seed))

    @property
    def enabled(self) -> bool:
        return self.base_seed is not None

    def rollout_seed(self, rollout_index: int) -> Optional[int]:
        if self.base_seed is None:
            return None
        return _derive_seed(self.base_seed, "rollout", int(rollout_index))

    def query_seed(self, rollout_seed: Optional[int], query_index: int) -> Optional[int]:
        if rollout_seed is None:
            return None
        return _derive_seed(int(rollout_seed), "query", int(query_index))

    def metadata(self, rollout_index: int) -> Dict[str, Any]:
        rollout_seed = self.rollout_seed(rollout_index)
        return {
            "strategy": SEED_STRATEGY,
            "base_seed": self.base_seed,
            "rollout_index": int(rollout_index),
            "rollout_seed": rollout_seed,
            "deterministic_policy_generator": rollout_seed is not None,
        }


def configure_process_seed(
    seed_plan: RolloutSeedPlan,
    *,
    deterministic_algorithms: bool = False,
) -> Dict[str, Any]:
    """Seed ordinary process RNGs once before policy construction.

    Per-query policy generators remain the authoritative action-sampling RNG.
    This process-level setup additionally makes incidental Python/NumPy/Torch
    randomness auditable.  Deterministic CUDA kernels are opt-in because they
    can be unavailable or materially slower on a deployment.
    """
    result: Dict[str, Any] = {
        "configured": False,
        "base_seed": seed_plan.base_seed,
        "deterministic_algorithms_requested": bool(deterministic_algorithms),
        "deterministic_algorithms_enabled": False,
    }
    if seed_plan.base_seed is None:
        return result

    seed = seed_plan.base_seed
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_algorithms:
            torch.use_deterministic_algorithms(True, warn_only=True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
            result["deterministic_algorithms_enabled"] = True
        result["torch_version"] = str(torch.__version__)
    except ImportError:
        result["torch_version"] = None
    result["configured"] = True
    return result


def _git_metadata(project_root: Optional[Path]) -> Dict[str, Any]:
    if project_root is None:
        return {"available": False}
    root = Path(project_root).expanduser().resolve()
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            ).stdout.strip()
        )
        return {"available": True, "root": str(root), "revision": revision, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"available": False, "root": str(root)}


def _runtime_metadata() -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    try:
        import torch

        result["torch"] = str(torch.__version__)
        result["torch_cuda"] = str(torch.version.cuda) if torch.version.cuda else None
    except ImportError:
        result["torch"] = None
        result["torch_cuda"] = None
    return result


def _config_metadata(config: Mapping[str, Any], config_path: Optional[str]) -> Dict[str, Any]:
    sanitized = _redact_config(config)
    path = Path(config_path).expanduser().resolve() if config_path else None
    return {
        "path": str(path) if path else None,
        "file_sha256": _file_sha256(path) if path else None,
        "resolved_sha256": _sha256_bytes(_canonical_json(sanitized)),
        "resolved": sanitized,
    }


def _camera_metadata(config: Mapping[str, Any], arm: str) -> Sequence[Dict[str, Any]]:
    cameras = ((config.get("sensors") or {}).get("cameras") or {})
    result = []
    for role in sorted(cameras):
        camera = cameras[role] or {}
        if not isinstance(camera, Mapping):
            camera = {"value": camera}
        result.append(
            {
                "arm_config": arm,
                "role": str(role),
                "device_id": _json_safe(camera.get("device_id")),
                "enabled": bool(camera.get("enabled", True)),
            }
        )
    return result


def _can_metadata(config: Mapping[str, Any], arm: str) -> Dict[str, Any]:
    robot = config.get("robot") or {}
    if not isinstance(robot, Mapping):
        robot = {"value": robot}
    return {
        "arm_config": arm,
        "channel": _json_safe(robot.get("channel")),
        "robot_target": _json_safe(robot.get("_target_")),
    }


def policy_metadata(policy: Any) -> Dict[str, Any]:
    """Collect policy provenance without assuming a local-policy implementation."""
    result: Dict[str, Any] = {
        "class": type(policy).__name__,
        "module": type(policy).__module__,
    }
    describe = getattr(policy, "reproducibility_metadata", None)
    if callable(describe):
        try:
            extra = describe()
        except Exception as exc:  # provenance must not block a safe run
            result["metadata_error"] = f"{type(exc).__name__}: {exc}"
        else:
            if isinstance(extra, Mapping):
                result.update(_json_safe(extra))
            else:
                result["metadata_error"] = "policy returned non-mapping metadata"
    return result


def build_rollout_manifest(
    *,
    instruction: str,
    rollout_index: int,
    seed_plan: RolloutSeedPlan,
    policy: Any,
    primary_config: Mapping[str, Any],
    primary_config_path: Optional[str],
    secondary_config: Optional[Mapping[str, Any]] = None,
    secondary_config_path: Optional[str] = None,
    process_seed_metadata: Optional[Mapping[str, Any]] = None,
    project_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Build the durable, hardware-free provenance record for one rollout."""
    configs: Dict[str, Any] = {
        "primary": _config_metadata(primary_config, primary_config_path),
    }
    cameras = list(_camera_metadata(primary_config, "primary"))
    can_channels = [_can_metadata(primary_config, "primary")]
    if secondary_config is not None:
        configs["secondary"] = _config_metadata(secondary_config, secondary_config_path)
        cameras.extend(_camera_metadata(secondary_config, "secondary"))
        can_channels.append(_can_metadata(secondary_config, "secondary"))

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at": _utc_now(),
        "instruction": str(instruction),
        "reproducibility": {
            **seed_plan.metadata(rollout_index),
            "process_seed": _json_safe(process_seed_metadata or {}),
        },
        "policy": policy_metadata(policy),
        "hardware_configuration": {
            "cameras": cameras,
            "can": can_channels,
        },
        "configs": configs,
        "runtime": _runtime_metadata(),
        "source": _git_metadata(project_root),
    }


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically publish a JSON manifest for a detached postmortem reader."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with open(temporary, "w", encoding="utf-8") as file:
            json.dump(_json_safe(value), file, indent=2, sort_keys=True, allow_nan=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
