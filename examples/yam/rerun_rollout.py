"""Create an offline Rerun recording from one saved YAM rollout.

This module is deliberately post-hoc: it never opens a camera, a CAN device,
or a robot.  It reads the raw rollout saved by :class:`EvalRolloutSaver` and
writes a portable ``.rrd`` file.  Open the result after the rollout has
finished with::

    /home/npow/molmoact2-venv/bin/rerun path/to/rollout/rollout.rrd

The current on-disk rollout format records encoder observations (``state``
and ``next_state``) and RGB PNGs.  If an ``action``-like dataset is present
in ``episode.h5`` (or an ``actions.npy``/``policy_actions.npy`` sidecar is
present), it is logged as the policy target too.  This lets the recorder be
used for old rollouts immediately, while retaining target-vs-feedback plots
for new rollouts once the launcher persists policy actions.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, Optional
from uuid import uuid4

import h5py
import numpy as np
from PIL import Image
import rerun as rr
import rerun.blueprint as rrb


CAMERAS = ("left", "front", "right")
JOINT_NAMES = (
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "joint_6",
    "gripper",
)
ACTION_DATASETS = (
    "action",
    "actions",
    "target",
    "targets",
    "target_joints",
    "target_joint_positions",
    "command",
    "commands",
    "commanded_joint_positions",
)
ACTION_SIDECARS = ("policy_actions.npy", "actions.npy", "targets.npy")


def _blueprint() -> rrb.Blueprint:
    """A useful default layout, stored inside every generated RRD."""
    cameras = rrb.Horizontal(
        rrb.Spatial2DView(origin="/cameras/left", name="left camera"),
        rrb.Spatial2DView(origin="/cameras/front", name="front camera"),
        rrb.Spatial2DView(origin="/cameras/right", name="right camera"),
        column_shares=[1, 1, 1],
    )
    telemetry = rrb.Horizontal(
        rrb.TimeSeriesView(origin="/robot/actual", name="encoder feedback"),
        rrb.TimeSeriesView(origin="/policy/target", name="policy target"),
        rrb.TimeSeriesView(origin="/robot/tracking_error", name="target - feedback"),
        column_shares=[1, 1, 1],
    )
    return rrb.Blueprint(
        rrb.Vertical(
            cameras,
            telemetry,
            rrb.Horizontal(
                rrb.TextDocumentView(origin="/task/instruction", name="instruction"),
                rrb.TextDocumentView(origin="/policy/action_chunk", name="current action chunk"),
                column_shares=[1, 2],
            ),
            row_shares=[2, 1, 1],
        ),
        collapse_panels=True,
    )


def _as_2d(array: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError(f"{name} must have shape (steps, dofs), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _load_targets(rollout_dir: Path, h5: h5py.File, num_steps: int) -> Optional[np.ndarray]:
    """Load optional commanded targets without guessing from measured state."""
    for name in ACTION_DATASETS:
        if name in h5:
            actions = _as_2d(np.asarray(h5[name]), f"episode.h5:{name}")
            break
    else:
        actions = None
        for sidecar_name in ACTION_SIDECARS:
            sidecar = rollout_dir / sidecar_name
            if sidecar.exists():
                actions = _as_2d(np.load(sidecar), str(sidecar))
                break

    if actions is None:
        return None
    if len(actions) != num_steps:
        raise ValueError(
            f"target action count ({len(actions)}) does not match saved state count ({num_steps})"
        )
    return actions


def _joint_names(dofs: int) -> tuple[str, ...]:
    if dofs == len(JOINT_NAMES):
        return JOINT_NAMES
    return tuple(f"joint_{index + 1}" for index in range(dofs))


def _log_series_style(
    recording: rr.RecordingStream,
    path: str,
    names: Iterable[str],
    color: list[int],
) -> None:
    recording.log(path, rr.SeriesLines(names=list(names), colors=[color]), static=True)


def _load_rgb(path: Path) -> np.ndarray:
    # PIL's conversion is explicit: Rerun Image expects RGB rather than BGR.
    with Image.open(path) as image:
        return np.ascontiguousarray(np.asarray(image.convert("RGB")))


def _temporary_output_path(output_path: Path) -> Path:
    """Return a sibling path which cannot be mistaken for a finished RRD."""
    return output_path.with_name(f".{output_path.name}.{uuid4().hex}.partial")


def _finalize_output(temporary_path: Path, output_path: Path) -> Path:
    """Atomically publish a completely finalized RRD.

    Rerun writes its footer when the recording stream disconnects.  Writing to
    a temporary sibling and replacing only afterwards prevents a watcher (or a
    user double-clicking the file) from ever seeing a half-written ``.rrd``.
    """
    if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
        raise RuntimeError(f"Rerun did not create a nonempty recording at {temporary_path}")
    os.replace(temporary_path, output_path)
    return output_path


def _cleanup_temporary_output(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        # A failed best-effort cleanup must not hide the original export error.
        pass


def _incomplete_instruction(rollout_dir: Path) -> str:
    """Read the durable launch marker when no completed HDF5 is available."""
    marker = rollout_dir / "rollout.started.json"
    try:
        value = json.loads(marker.read_text())
    except (OSError, json.JSONDecodeError):
        return ""
    instruction = value.get("instruction", "")
    return instruction if isinstance(instruction, str) else ""


def _camera_frame_sets(rollout_dir: Path) -> dict[str, dict[int, Path]]:
    """Index numeric PNG frames without assuming every camera is perfect."""
    frames: dict[str, dict[int, Path]] = {}
    for camera in CAMERAS:
        camera_dir = rollout_dir / f"{camera}_rgb"
        if not camera_dir.is_dir():
            continue
        indexed: dict[int, Path] = {}
        for image_path in camera_dir.glob("*.png"):
            try:
                step = int(image_path.stem)
            except ValueError:
                continue
            indexed[step] = image_path
        if indexed:
            frames[camera] = indexed
    return frames


def _chunk_markdown(chunk_index: int, first_step: int, actions: np.ndarray) -> str:
    header = " | ".join(("step",) + _joint_names(actions.shape[1]))
    divider = " | ".join(["---"] * (actions.shape[1] + 1))
    rows = [f"{first_step + i} | " + " | ".join(f"{x:.4f}" for x in action) for i, action in enumerate(actions)]
    return "\n".join(
        (
            f"## Policy action chunk {chunk_index}",
            "",
            f"Absolute joint targets for rollout steps {first_step}–{first_step + len(actions) - 1}.",
            "",
            header,
            divider,
            *rows,
        )
    )


def _load_policy_chunks(h5: h5py.File) -> list[tuple[int, float, np.ndarray]]:
    """Load exact original policy plans when a newer rollout provides them."""
    if "policy_action_chunks" not in h5:
        return []

    group = h5["policy_action_chunks"]
    chunks: list[tuple[int, float, np.ndarray]] = []
    for name in sorted(group.keys()):
        dataset = group[name]
        actions = _as_2d(np.asarray(dataset), f"episode.h5:policy_action_chunks/{name}")
        start_step = int(dataset.attrs["start_step"])
        inference_sec = float(dataset.attrs.get("inference_sec", float("nan")))
        chunks.append((start_step, inference_sec, actions))
    return chunks


def _load_optional_step_values(
    h5: h5py.File, name: str, num_steps: int
) -> Optional[np.ndarray]:
    """Read an optional one-value-per-control-step diagnostic dataset."""
    if name not in h5:
        return None
    values = np.asarray(h5[name])
    if values.shape != (num_steps,):
        raise ValueError(
            f"episode.h5:{name} must have shape ({num_steps},), got {values.shape}"
        )
    return values


def write_rollout_rrd(
    rollout_dir: str | Path,
    output_path: str | Path | None = None,
    *,
    fps: float = 30.0,
    image_stride: int = 6,
    jpeg_quality: int = 75,
    action_chunk_size: int = 30,
) -> Path:
    """Write a self-contained Rerun ``.rrd`` from an already-finished rollout.

    ``image_stride`` only down-samples images; encoder feedback and targets are
    always logged at every saved control step. A stride of 6 gives a 5 Hz
    visual replay for a 30 Hz rollout, matching the current physical camera
    capture rate while avoiding six copies of each already-cached frame. Set
    it to 1 when every saved camera frame is needed.
    """
    rollout_dir = Path(rollout_dir).expanduser().resolve()
    episode_path = rollout_dir / "episode.h5"
    if not episode_path.is_file():
        raise FileNotFoundError(f"Missing {episode_path}")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if image_stride < 1:
        raise ValueError("image_stride must be at least 1")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be in [1, 100]")
    if action_chunk_size < 1:
        raise ValueError("action_chunk_size must be at least 1")

    output_path = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else rollout_dir / "rollout.rrd"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = _temporary_output_path(output_path)

    with h5py.File(episode_path, "r") as h5:
        state = _as_2d(np.asarray(h5["state"]), "episode.h5:state")
        next_state = _as_2d(np.asarray(h5["next_state"]), "episode.h5:next_state")
        if state.shape != next_state.shape:
            raise ValueError(
                f"state shape {state.shape} differs from next_state shape {next_state.shape}"
            )
        instruction = str(h5.attrs.get("language_instruction", ""))
        targets = _load_targets(rollout_dir, h5, len(state))
        policy_chunks = _load_policy_chunks(h5)
        policy_chunk_indices = _load_optional_step_values(h5, "policy_chunk_index", len(state))
        policy_action_indices = _load_optional_step_values(h5, "policy_action_index", len(state))
        policy_inference_times = _load_optional_step_values(h5, "policy_inference_sec", len(state))

    names = _joint_names(state.shape[1])
    # A local stream means multiple rollouts can be exported in one Python
    # process without their data or file sinks bleeding into each other.
    recording = rr.RecordingStream("yam_molmoact_rollout", recording_id=uuid4())
    try:
        with recording:
            # save() must happen before the first log. Context-manager exit
            # writes the footer, making this RRD immediately seekable.
            recording.save(temporary_path, default_blueprint=_blueprint())
            recording.log(
            "/task/instruction",
            rr.TextDocument(
                f"# Task instruction\n\n{instruction or '_No instruction saved._'}",
                media_type="text/markdown",
            ),
            static=True,
            )
            recording.log(
            "/task/recording_notes",
            rr.TextDocument(
                "# Rollout data\n\n"
                "`robot/actual/pre` is encoder feedback before a command; "
                "`robot/actual/post` is encoder feedback after it. "
                + (
                    "`policy/target` is the saved policy command."
                    if targets is not None
                    else "No policy target was saved in this historical rollout."
                ),
                media_type="text/markdown",
            ),
            static=True,
            )
            _log_series_style(recording, "/robot/actual/pre", names, [65, 150, 245])
            _log_series_style(recording, "/robot/actual/post", names, [65, 190, 115])
            if targets is not None:
                _log_series_style(recording, "/policy/target", names, [245, 170, 60])
                _log_series_style(recording, "/robot/tracking_error", names, [230, 75, 75])

            chunks_by_start_step = {
                start_step: (chunk_index, inference_sec, actions)
                for chunk_index, (start_step, inference_sec, actions) in enumerate(policy_chunks)
            }

            for step in range(len(state)):
                recording.set_time("step", sequence=step)
                recording.set_time("elapsed", duration=step / fps)
                recording.log("/robot/actual/pre", rr.Scalars(state[step]))
                recording.log("/robot/actual/post", rr.Scalars(next_state[step]))

                if targets is not None:
                    recording.log("/policy/target", rr.Scalars(targets[step]))
                    recording.log("/robot/tracking_error", rr.Scalars(targets[step] - next_state[step]))

                    if step in chunks_by_start_step:
                        chunk_index, inference_sec, chunk = chunks_by_start_step[step]
                        recording.log(
                            "/policy/action_chunk",
                            rr.TextDocument(
                                _chunk_markdown(chunk_index, step, chunk),
                                media_type="text/markdown",
                            ),
                        )
                        if policy_inference_times is None and np.isfinite(inference_sec):
                            recording.log("/policy/inference_sec", rr.Scalars(inference_sec))
                    elif not policy_chunks and step % action_chunk_size == 0:
                        # Historical rollouts recorded individual targets but not
                        # the original chunk boundaries. Preserve a useful view
                        # without claiming this reconstruction is authoritative.
                        chunk = targets[step : step + action_chunk_size]
                        recording.log(
                            "/policy/action_chunk",
                            rr.TextDocument(
                                _chunk_markdown(step // action_chunk_size, step, chunk),
                                media_type="text/markdown",
                            ),
                        )

                    if policy_chunk_indices is not None:
                        recording.log("/policy/chunk_index", rr.Scalars(policy_chunk_indices[step]))
                    if policy_action_indices is not None:
                        recording.log("/policy/action_index", rr.Scalars(policy_action_indices[step]))
                    if policy_inference_times is not None and np.isfinite(policy_inference_times[step]):
                        recording.log("/policy/inference_sec", rr.Scalars(policy_inference_times[step]))

                if step % image_stride == 0:
                    for camera in CAMERAS:
                        image_path = rollout_dir / f"{camera}_rgb" / f"{step:06d}.png"
                        if image_path.is_file():
                            # JPEG makes the RRD substantially smaller than reusing
                            # the raw decoded RGB tensor, while keeping each frame
                            # independently seekable in Rerun.
                            recording.log(
                                f"/cameras/{camera}",
                                rr.Image(_load_rgb(image_path)).compress(jpeg_quality=jpeg_quality),
                            )
    except Exception:
        _cleanup_temporary_output(temporary_path)
        raise
    finally:
        # ``RecordingStream.__exit__`` finalizes the .rrd, but it deliberately
        # leaves its background sink alive. Explicitly close it so a finished
        # rollout process releases CUDA/camera resources and can exit.
        recording.disconnect()

    return _finalize_output(temporary_path, output_path)


def write_incomplete_rollout_rrd(
    rollout_dir: str | Path,
    output_path: str | Path | None = None,
    *,
    reason: str,
    fps: float = 30.0,
    image_stride: int = 6,
    jpeg_quality: int = 75,
) -> Path:
    """Write an honest camera-only RRD when a rollout died before HDF5 flush.

    The normal saver keeps joint telemetry in memory until its final flush. If
    a process dies during that flush, inventing state/action arrays would make
    the replay look valid while concealing lost data. This fallback instead
    keeps the synchronized camera frames that reached disk and records a
    prominent explanation that telemetry was not persisted.

    Frames are synchronized by the intersection of camera frame indices. A
    single missing final camera frame therefore drops only that unmatched frame
    (the exact failure mode of rollout ``20260727_163712``).
    """
    rollout_dir = Path(rollout_dir).expanduser().resolve()
    if fps <= 0:
        raise ValueError("fps must be positive")
    if image_stride < 1:
        raise ValueError("image_stride must be at least 1")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be in [1, 100]")

    output_path = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else rollout_dir / "rollout.rrd"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    camera_frames = _camera_frame_sets(rollout_dir)
    shared_steps = (
        sorted(set.intersection(*(set(values) for values in camera_frames.values())))
        if camera_frames
        else []
    )
    frame_counts = (
        ", ".join(f"{camera}={len(frames)}" for camera, frames in sorted(camera_frames.items()))
        if camera_frames
        else "none"
    )
    instruction = _incomplete_instruction(rollout_dir)
    temporary_path = _temporary_output_path(output_path)
    recording = rr.RecordingStream("yam_molmoact_incomplete_rollout", recording_id=uuid4())
    try:
        with recording:
            recording.save(temporary_path, default_blueprint=_blueprint())
            recording.log(
                "/task/instruction",
                rr.TextDocument(
                    f"# Task instruction\n\n{instruction or '_Instruction was not persisted._'}",
                    media_type="text/markdown",
                ),
                static=True,
            )
            recording.log(
                "/task/recording_notes",
                rr.TextDocument(
                    "# Incomplete rollout — camera-only recovery\n\n"
                    f"{reason}\n\n"
                    "The control process ended before it durably wrote `episode.h5`, "
                    "so joint feedback, policy actions, and inference metadata cannot "
                    "be reconstructed honestly. This replay contains only synchronized "
                    "camera frames that reached disk.\n\n"
                    f"Camera frame counts before synchronization: `{frame_counts}`.  "
                    f"Using {len(shared_steps)} shared frame indices; image stride={image_stride}.",
                    media_type="text/markdown",
                ),
                static=True,
            )
            for sample_index, step in enumerate(shared_steps):
                if sample_index % image_stride:
                    continue
                recording.set_time("step", sequence=step)
                recording.set_time("elapsed", duration=step / fps)
                for camera, frames in camera_frames.items():
                    recording.log(
                        f"/cameras/{camera}",
                        rr.Image(_load_rgb(frames[step])).compress(jpeg_quality=jpeg_quality),
                    )
    except Exception:
        _cleanup_temporary_output(temporary_path)
        raise
    finally:
        recording.disconnect()

    return _finalize_output(temporary_path, output_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollout_dir", type=Path, help="Directory containing episode.h5 and *_rgb/")
    parser.add_argument("--output", type=Path, default=None, help="Output .rrd path (default: <rollout_dir>/rollout.rrd)")
    parser.add_argument("--fps", type=float, default=30.0, help="Saved control frequency used for the elapsed timeline")
    parser.add_argument("--image-stride", type=int, default=6, help="Log every Nth image frame (default: 6)")
    parser.add_argument("--jpeg-quality", type=int, default=75, help="JPEG quality for camera frames (default: 75)")
    parser.add_argument("--action-chunk-size", type=int, default=30, help="Actions per policy chunk (default: 30)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    path = write_rollout_rrd(
        args.rollout_dir,
        args.output,
        fps=args.fps,
        image_stride=args.image_stride,
        jpeg_quality=args.jpeg_quality,
        action_chunk_size=args.action_chunk_size,
    )
    print(f"[rerun] wrote {path}")
    print(f"[rerun] playback: /home/npow/molmoact2-venv/bin/rerun {path}")
    print(
        "[rerun] browser: /home/npow/molmoact2-venv/bin/rerun "
        f"--web-viewer --web-viewer-port 9090 {path}"
    )


if __name__ == "__main__":
    main()
