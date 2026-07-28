#!/usr/bin/env python3
"""Camera-only web dashboard for the physical YAM camera setup.

This process intentionally imports no I2RT robot or CAN code.  It opens the
three V4L2 cameras named by a physical YAM config and serves their latest RGB
frames in a simple browser page.  It is meant for checking and aiming cameras
while the arms are idle.

Example::

    cd /tmp/molmoact2
    PYTHONPATH=examples/yam /home/npow/molmoact2-venv/bin/python \
      examples/yam/live_camera_dashboard.py \
      --config examples/yam/configs/yam_left_physical.yaml

Then open http://127.0.0.1:9091 .  Press Ctrl-C in the process terminal to
release all V4L2 devices.
"""

from __future__ import annotations

import argparse
import html
import logging
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from omegaconf import OmegaConf

from gello_min.v4l2_camera import V4L2Camera


CAMERA_NAMES = ("left_camera", "front_camera", "right_camera")
LOGGER = logging.getLogger("yam.live_camera_dashboard")


@dataclass(frozen=True)
class Frame:
    jpeg: bytes
    captured_at: float


class CameraDashboard:
    """Own V4L2 devices and maintain lightweight browser-ready JPEG frames."""

    def __init__(self, devices: Dict[str, str], fps: float, jpeg_quality: int) -> None:
        self._devices = devices
        self._fps = fps
        self._jpeg_quality = jpeg_quality
        self._cameras: Dict[str, V4L2Camera] = {}
        self._frames: Dict[str, Frame] = {}
        self._errors: Dict[str, str] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._encode_thread: Optional[threading.Thread] = None

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._devices)

    def start(self) -> None:
        try:
            for name, device in self._devices.items():
                self._cameras[name] = V4L2Camera(device, width=640, height=480, fps=int(self._fps))
        except Exception:
            self.close()
            raise

        self._encode_thread = threading.Thread(
            target=self._encode_loop,
            name="yam_camera_dashboard_encoder",
            daemon=True,
        )
        self._encode_thread.start()

    def _encode_loop(self) -> None:
        period_sec = 1.0 / self._fps
        params = [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
        while not self._stop_event.is_set():
            started = time.monotonic()
            for name, camera in self._cameras.items():
                try:
                    rgb, _ = camera.read()
                    bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)
                    ok, encoded = cv2.imencode(".jpg", bgr, params)
                    if not ok:
                        raise RuntimeError("OpenCV JPEG encoding failed")
                    with self._lock:
                        self._frames[name] = Frame(bytes(encoded), time.time())
                        self._errors.pop(name, None)
                except Exception as exc:  # noqa: BLE001 -- keep other feeds live
                    with self._lock:
                        self._errors[name] = str(exc)
            self._stop_event.wait(max(0.0, period_sec - (time.monotonic() - started)))

    def frame(self, name: str) -> Frame:
        with self._lock:
            frame = self._frames.get(name)
            error = self._errors.get(name)
        if frame is None:
            detail = error or "waiting for the first frame"
            raise HTTPException(status_code=503, detail=f"{name} camera unavailable: {detail}")
        return frame

    def status(self) -> dict[str, dict[str, object]]:
        now = time.time()
        with self._lock:
            return {
                name: {
                    "frame_age_sec": None if name not in self._frames else round(now - self._frames[name].captured_at, 3),
                    "error": self._errors.get(name),
                }
                for name in self._devices
            }

    def close(self) -> None:
        self._stop_event.set()
        if self._encode_thread is not None:
            self._encode_thread.join(timeout=2.0)
            self._encode_thread = None
        for camera in self._cameras.values():
            camera.close()
        self._cameras = {}


def _camera_devices(config_path: Path) -> Dict[str, str]:
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    camera_config = config["sensors"]["cameras"]
    devices: Dict[str, str] = {}
    for camera_name in CAMERA_NAMES:
        spec = camera_config[camera_name]
        if not bool(spec.get("enabled", True)):
            continue
        device = str(spec["device_id"])
        if not device.startswith("/dev/"):
            raise ValueError(
                f"{camera_name} has {device!r}; this dashboard requires a V4L2 /dev/ path"
            )
        devices[camera_name.removesuffix("_camera")] = device
    if not devices:
        raise ValueError(f"No enabled V4L2 cameras in {config_path}")
    return devices


def _page(camera_names: tuple[str, ...]) -> str:
    cards = "\n".join(
        f'''<section class="card"><h2>{html.escape(name)}</h2><img id="{html.escape(name)}" alt="{html.escape(name)} live feed"><p id="{html.escape(name)}-status">starting…</p></section>'''
        for name in camera_names
    )
    names = ", ".join(repr(name) for name in camera_names)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>YAM live cameras</title><style>
body {{ margin: 0; background: #111827; color: #f9fafb; font: 16px system-ui, sans-serif; }}
header {{ padding: 16px 22px; border-bottom: 1px solid #374151; }}
h1 {{ margin: 0; font-size: 21px; }} header p {{ margin: 5px 0 0; color: #9ca3af; }}
main {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; padding: 16px; }}
.card {{ background: #1f2937; border-radius: 10px; padding: 12px; min-width: 0; }}
h2 {{ margin: 0 0 10px; text-transform: capitalize; font-size: 17px; }}
img {{ display: block; width: 100%; aspect-ratio: 4 / 3; object-fit: contain; background: #030712; border-radius: 6px; }}
.card p {{ margin: 8px 0 0; font: 13px ui-monospace, monospace; color: #9ca3af; }}
</style></head><body><header><h1>YAM live cameras</h1><p>Camera-only dashboard — no CAN or arm control.</p></header>
<main>{cards}</main><script>
const cameras = [{names}];
async function refresh() {{
  const stamp = Date.now();
  for (const name of cameras) document.getElementById(name).src = `/frame/${{name}}.jpg?t=${{stamp}}`;
  try {{
    const status = await (await fetch('/status', {{cache: 'no-store'}})).json();
    for (const name of cameras) {{
      const item = status[name];
      document.getElementById(`${{name}}-status`).textContent = item.error ? `ERROR: ${{item.error}}` : `frame age: ${{item.frame_age_sec}} s`;
    }}
  }} catch (err) {{ for (const name of cameras) document.getElementById(`${{name}}-status`).textContent = `status unavailable: ${{err}}`; }}
  setTimeout(refresh, 200);
}}
refresh();
</script></body></html>"""


def _app(dashboard: CameraDashboard) -> FastAPI:
    app = FastAPI(title="YAM live cameras", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _page(dashboard.names)

    @app.get("/frame/{name}.jpg")
    def frame(name: str) -> Response:
        if name not in dashboard.names:
            raise HTTPException(status_code=404, detail=f"unknown camera {name!r}")
        image = dashboard.frame(name)
        return Response(
            content=image.jpeg,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    @app.get("/status")
    def status() -> dict[str, dict[str, object]]:
        return dashboard.status()

    return app


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("examples/yam/configs/yam_left_physical.yaml"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9091)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--jpeg-quality", type=int, default=75)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.fps <= 0:
        raise SystemExit("--fps must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        raise SystemExit("--jpeg-quality must be in [1, 100]")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    dashboard = CameraDashboard(_camera_devices(args.config), fps=args.fps, jpeg_quality=args.jpeg_quality)

    def _request_stop(_signum: int, _frame: object) -> None:
        # Uvicorn observes SIGINT/SIGTERM too; this handler only makes a
        # second Ctrl-C harmless while its orderly shutdown is in progress.
        dashboard.close()

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    dashboard.start()
    print(f"[live-camera-dashboard] Open: http://{args.host}:{args.port}", flush=True)
    print("[live-camera-dashboard] Camera-only process; Ctrl-C releases the devices.", flush=True)
    try:
        uvicorn.run(_app(dashboard), host=args.host, port=args.port, log_level="warning")
    finally:
        dashboard.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
