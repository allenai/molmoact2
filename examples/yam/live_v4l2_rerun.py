"""Camera-only live Rerun web viewer for the physical YAM setup.

This program intentionally imports no robot, motor, or CAN code. It opens
only the three V4L2 RGB devices configured in a physical YAM YAML file and
streams their latest frames to a local Rerun web viewer. It is for aiming and
checking cameras while the arms are idle.

Run from the molmoact2 checkout::

    PYTHONPATH=examples/yam /home/npow/molmoact2-venv/bin/python \
      examples/yam/live_v4l2_rerun.py \
      --config examples/yam/configs/yam_left_physical.yaml

Then open the URL printed at startup (normally http://127.0.0.1:9091/?url=...).
Use Ctrl-C to stop it and release all camera devices.
"""

from __future__ import annotations

import argparse
import logging
import signal
import threading
import time
from pathlib import Path
from typing import Dict, Sequence
from urllib.parse import quote

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from omegaconf import OmegaConf

from gello_min.v4l2_camera import V4L2Camera


CAMERA_NAMES = ("left_camera", "front_camera", "right_camera")
LOGGER = logging.getLogger("yam.live_v4l2_rerun")


def _blueprint() -> rrb.Blueprint:
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(origin="/cameras/left", name="left camera"),
            rrb.Spatial2DView(origin="/cameras/front", name="front camera"),
            rrb.Spatial2DView(origin="/cameras/right", name="right camera"),
            column_shares=[1, 1, 1],
        ),
        collapse_panels=True,
    )


def _camera_devices(config_path: Path) -> Dict[str, str]:
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    camera_config = config["sensors"]["cameras"]
    devices: Dict[str, str] = {}
    for camera_name in CAMERA_NAMES:
        spec = camera_config[camera_name]
        if not bool(spec.get("enabled", True)):
            LOGGER.warning("%s disabled in %s", camera_name, config_path)
            continue
        device = str(spec["device_id"])
        if not device.startswith("/dev/"):
            raise ValueError(
                f"{camera_name} has {device!r}; this camera-only viewer requires a V4L2 /dev/ path"
            )
        devices[camera_name.replace("_camera", "")] = device
    if not devices:
        raise ValueError(f"No enabled V4L2 cameras in {config_path}")
    return devices


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("examples/yam/configs/yam_left_physical.yaml"),
        help="Physical YAM config containing sensors.cameras device paths.",
    )
    parser.add_argument("--fps", type=float, default=5.0, help="Per-camera stream rate (default: 5).")
    parser.add_argument("--jpeg-quality", type=int, default=75)
    parser.add_argument("--grpc-port", type=int, default=9877)
    parser.add_argument("--web-port", type=int, default=9091)
    parser.add_argument("--server-memory-limit", default="256MiB")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.fps <= 0:
        raise SystemExit("--fps must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        raise SystemExit("--jpeg-quality must be in [1, 100]")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    devices = _camera_devices(args.config)
    LOGGER.info("Opening V4L2 cameras only: %s", devices)

    stop_event = threading.Event()

    def _request_stop(_signum: int, _frame: object) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    blueprint = _blueprint()
    rr.init("yam_live_v4l2_cameras", default_blueprint=blueprint)
    grpc_url = rr.serve_grpc(
        grpc_port=args.grpc_port,
        default_blueprint=blueprint,
        server_memory_limit=str(args.server_memory_limit),
        newest_first=True,
    )
    rr.serve_web_viewer(web_port=args.web_port, open_browser=False, connect_to=grpc_url)
    viewer_url = f"http://127.0.0.1:{args.web_port}/?url={quote(grpc_url, safe='')}"
    print(f"[live-camera-rerun] Open: {viewer_url}", flush=True)
    print("[live-camera-rerun] Camera-only process; Ctrl-C stops it and releases the devices.", flush=True)

    cameras: Dict[str, V4L2Camera] = {}
    try:
        for name, device in devices.items():
            cameras[name] = V4L2Camera(device, width=640, height=480, fps=int(args.fps))

        frame_index = 0
        period_sec = 1.0 / args.fps
        while not stop_event.is_set():
            started = time.monotonic()
            rr.set_time("frame", sequence=frame_index)
            for name, camera in cameras.items():
                try:
                    image, _ = camera.read()
                    rr.log(
                        f"/cameras/{name}",
                        rr.Image(np.ascontiguousarray(image)).compress(jpeg_quality=args.jpeg_quality),
                    )
                except Exception as exc:  # noqa: BLE001 -- leave the other feeds usable
                    LOGGER.warning("%s frame unavailable: %s", name, exc)
            frame_index += 1
            stop_event.wait(max(0.0, period_sec - (time.monotonic() - started)))
    finally:
        for camera in cameras.values():
            camera.close()
        rr.disconnect()
        LOGGER.info("Live camera viewer stopped; V4L2 devices released.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
