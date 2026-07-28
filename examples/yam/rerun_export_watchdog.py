"""Detached, hardware-free Rerun exporter for YAM rollout directories.

This process is deliberately separate from the policy/robot launcher.  It
opens neither CAN nor cameras, waits for the launcher to exit, then converts
every raw rollout it can find below ``--root`` into an atomically-published
``rollout.rrd``.  If a launcher dies before final HDF5 flush, it still creates
an explicitly camera-only (or marker-only) RRD instead of silently omitting a
replay.

The module imports only the offline ``rerun_rollout`` converter and standard
library modules.  It is safe to leave running after a controller failure.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from rerun_rollout import write_incomplete_rollout_rrd, write_rollout_rrd


CAMERA_DIRS = ("left_rgb", "front_rgb", "right_rgb")
STATUS_FILENAME = "rerun_export.status.json"
LOCK_FILENAME = ".rerun_export.lock"


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _atomic_write_json(path: Path, value: Dict[str, Any]) -> None:
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with open(temporary_path, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _write_status(rollout_dir: Path, state: str, **details: Any) -> None:
    _atomic_write_json(
        rollout_dir / STATUS_FILENAME,
        {
            "schema_version": 1,
            "state": state,
            "updated_at": _timestamp(),
            **details,
        },
    )


def _parent_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # It exists but belongs to another user; do not prematurely export.
        return True
    return True


def _is_nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _acquire_lock(rollout_dir: Path, stale_after_seconds: float = 600.0) -> Path | None:
    """Prevent two independently started watchdogs from writing one RRD."""
    lock_path = rollout_dir / LOCK_FILENAME
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o664)
    except FileExistsError:
        try:
            if time.time() - lock_path.stat().st_mtime > stale_after_seconds:
                lock_path.unlink()
                return _acquire_lock(rollout_dir, stale_after_seconds)
        except OSError:
            pass
        return None
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"pid": os.getpid(), "started_at": _timestamp()}, f)
            f.write("\n")
    except Exception:
        lock_path.unlink(missing_ok=True)
        raise
    return lock_path


def _release_lock(lock_path: Path | None) -> None:
    if lock_path is None:
        return
    try:
        lock_path.unlink(missing_ok=True)
    except OSError:
        pass


def discover_rollout_dirs(root: Path) -> list[Path]:
    """Find saved and partially-saved rollout roots without walking PNG trees."""
    candidates: set[Path] = set()
    for current, dirnames, filenames in os.walk(root):
        current_path = Path(current)
        if (
            "episode.h5" in filenames
            or "rollout.started.json" in filenames
            or any(camera_dir in dirnames for camera_dir in CAMERA_DIRS)
        ):
            candidates.add(current_path)
        # Each camera tree can contain thousands of images. Its parent is the
        # only possible rollout root, so descending is needless and expensive.
        dirnames[:] = [
            name
            for name in dirnames
            if name not in CAMERA_DIRS and not name.startswith(".")
        ]
    return sorted(candidates)


def _short_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def export_rollout(
    rollout_dir: Path,
    *,
    fps: float,
    image_stride: int,
    jpeg_quality: int,
    action_chunk_size: int,
) -> bool:
    """Export one directory, falling back to an honest incomplete RRD.

    Returns ``True`` only once a nonempty final ``rollout.rrd`` exists.  The
    caller can safely retry a ``False`` result after transient filesystem or
    Rerun errors.
    """
    output_path = rollout_dir / "rollout.rrd"
    if _is_nonempty_file(output_path):
        _write_status(
            rollout_dir,
            "complete",
            mode=("existing_full_or_legacy" if (rollout_dir / "episode.h5").is_file() else "partial_camera_only"),
            output=output_path.name,
        )
        return True

    lock_path = _acquire_lock(rollout_dir)
    if lock_path is None:
        return False
    try:
        _write_status(rollout_dir, "exporting")
        episode_path = rollout_dir / "episode.h5"
        if episode_path.is_file():
            try:
                output = write_rollout_rrd(
                    rollout_dir,
                    output_path,
                    fps=fps,
                    image_stride=image_stride,
                    jpeg_quality=jpeg_quality,
                    action_chunk_size=action_chunk_size,
                )
            except Exception as full_error:  # noqa: BLE001 - fallback is intentional
                fallback_reason = (
                    "The saved HDF5 could not be converted into a full replay: "
                    f"{_short_error(full_error)}. Camera-only recovery was used instead."
                )
                try:
                    output = write_incomplete_rollout_rrd(
                        rollout_dir,
                        output_path,
                        reason=fallback_reason,
                        fps=fps,
                        image_stride=image_stride,
                        jpeg_quality=jpeg_quality,
                    )
                except Exception as fallback_error:  # noqa: BLE001 - persisted below
                    _write_status(
                        rollout_dir,
                        "error",
                        full_export_error=_short_error(full_error),
                        fallback_error=_short_error(fallback_error),
                        traceback=traceback.format_exc(limit=8),
                    )
                    return False
                _write_status(
                    rollout_dir,
                    "complete",
                    mode="partial_camera_only",
                    output=output.name,
                    full_export_error=_short_error(full_error),
                )
                return True

            _write_status(rollout_dir, "complete", mode="full", output=output.name)
            return True

        # ``episode.h5`` is absent: the saver was interrupted before durable
        # telemetry flush.  Make a truthful no-telemetry RRD from whatever
        # images (or just the launch marker) reached disk.
        output = write_incomplete_rollout_rrd(
            rollout_dir,
            output_path,
            reason=(
                "The launcher exited before it atomically published `episode.h5`. "
                "This is a recovered incomplete rollout."
            ),
            fps=fps,
            image_stride=image_stride,
            jpeg_quality=jpeg_quality,
        )
        _write_status(
            rollout_dir,
            "complete",
            mode="partial_camera_only",
            output=output.name,
        )
        return True
    except Exception as exc:  # noqa: BLE001 - watchdog must keep retrying
        _write_status(
            rollout_dir,
            "error",
            error=_short_error(exc),
            traceback=traceback.format_exc(limit=8),
        )
        return False
    finally:
        _release_lock(lock_path)


def export_pending_rollouts(
    root: Path,
    *,
    fps: float,
    image_stride: int,
    jpeg_quality: int,
    action_chunk_size: int,
) -> Tuple[int, int]:
    """Run a single offline export pass, returning ``(complete, pending)``."""
    complete = 0
    pending = 0
    for rollout_dir in discover_rollout_dirs(root):
        if export_rollout(
            rollout_dir,
            fps=fps,
            image_stride=image_stride,
            jpeg_quality=jpeg_quality,
            action_chunk_size=action_chunk_size,
        ):
            complete += 1
        else:
            pending += 1
    return complete, pending


def run_watchdog(args: argparse.Namespace) -> int:
    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.ready_file is not None:
        _atomic_write_json(
            args.ready_file.expanduser().resolve(),
            {
                "schema_version": 1,
                "state": "ready",
                "pid": os.getpid(),
                "root": str(root),
                "started_at": _timestamp(),
            },
        )

    # Waiting for the launcher to exit eliminates races with labeling/moving a
    # finished rollout.  A normal and an abnormal launcher exit take the same
    # path; this child neither owns nor retains hardware resources.
    while _parent_is_alive(args.parent_pid):
        time.sleep(args.poll_seconds)

    time.sleep(args.settle_seconds)
    completed_idle_scans = 0
    while True:
        _complete, pending = export_pending_rollouts(
            root,
            fps=args.fps,
            image_stride=args.image_stride,
            jpeg_quality=args.jpeg_quality,
            action_chunk_size=args.action_chunk_size,
        )
        if pending:
            completed_idle_scans = 0
        else:
            completed_idle_scans += 1
            if completed_idle_scans >= args.post_exit_scans:
                return 0
        time.sleep(args.poll_seconds)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Session task root containing eval/success/failure rollouts")
    parser.add_argument("--parent-pid", type=int, required=True, help="Launcher PID to wait for")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--image-stride", type=int, default=6)
    parser.add_argument("--jpeg-quality", type=int, default=75)
    parser.add_argument("--action-chunk-size", type=int, default=30)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--settle-seconds", type=float, default=2.0)
    parser.add_argument("--post-exit-scans", type=int, default=3)
    parser.add_argument("--ready-file", type=Path, default=None)
    args = parser.parse_args()
    if args.parent_pid < 1:
        parser.error("--parent-pid must be positive")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.image_stride < 1:
        parser.error("--image-stride must be at least 1")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be in [1, 100]")
    if args.action_chunk_size < 1:
        parser.error("--action-chunk-size must be at least 1")
    if args.poll_seconds <= 0 or args.settle_seconds < 0:
        parser.error("poll/settle durations must be nonnegative (poll must be positive)")
    if args.post_exit_scans < 1:
        parser.error("--post-exit-scans must be at least 1")
    return args


def main() -> None:
    raise SystemExit(run_watchdog(_parse_args()))


if __name__ == "__main__":
    main()
