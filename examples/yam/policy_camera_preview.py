#!/usr/bin/env python3
"""Camera-only MolmoAct2 input preview for the physical YAM setup.

This program deliberately imports neither ``i2rt`` nor any robot/CAN code.
It opens the three V4L2 devices from a physical YAM config, then displays them
in the *same semantic order* passed to MolmoAct2:

``[top_cam (front_camera), left_cam (left_camera), right_cam (right_camera)]``.

Use it before a rollout to check the two things a technical camera preflight
cannot infer: that ``top_cam`` visibly contains both the red lid and black
box, and that each wrist view is assigned to the intended physical arm.

Example (from the MolmoAct2 checkout)::

    PYTHONPATH=examples/yam /home/npow/molmoact2-venv/bin/python \
      examples/yam/policy_camera_preview.py \
      --config examples/yam/configs/yam_right_primary.yaml \
      --host 0.0.0.0 --port 9092

Open ``http://<Jetson-IP>:9092`` in a browser.  The process has no motor
handles; Ctrl-C only releases its V4L2 camera devices.
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
from typing import Any, Dict, Mapping, Optional, Sequence

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from omegaconf import OmegaConf

from gello_min.v4l2_camera import V4L2Camera


LOGGER = logging.getLogger("yam.policy_camera_preview")
EXPECTED_SHAPE = (360, 640, 3)
MIN_MEAN_LUMA = 15.0
MAX_MEAN_LUMA = 240.0
MAX_CLIPPED_WHITE_FRACTION = 0.15


@dataclass(frozen=True)
class PolicyCameraSlot:
    """One model image slot and the physical config field that supplies it."""

    model_key: str
    config_key: str
    title: str
    operator_check: str


# This is intentionally kept next to the UI rather than inferred from config
# iteration order.  It mirrors MolmoActLocal.inference exactly:
# ``top_cam=front_camera_rgb, left_cam=left_camera_rgb,
# right_cam=right_camera_rgb``.
POLICY_CAMERA_SLOTS: tuple[PolicyCameraSlot, ...] = (
    PolicyCameraSlot(
        model_key="top_cam",
        config_key="front_camera",
        title="TOP — static overhead scene",
        operator_check="Both the red lid and black box must be clearly visible and not washed out.",
    ),
    PolicyCameraSlot(
        model_key="left_cam",
        config_key="left_camera",
        title="LEFT — left wrist view",
        operator_check="Confirm this is the physical LEFT arm's wrist/workspace view.",
    ),
    PolicyCameraSlot(
        model_key="right_cam",
        config_key="right_camera",
        title="RIGHT — right wrist view",
        operator_check="Confirm this is the physical RIGHT arm's wrist/workspace view.",
    ),
)


@dataclass(frozen=True)
class CameraSpec:
    slot: PolicyCameraSlot
    device: str


@dataclass(frozen=True)
class EncodedFrame:
    jpeg: bytes
    captured_at: float
    quality: Dict[str, object]


def load_policy_camera_specs(config_path: Path) -> tuple[CameraSpec, ...]:
    """Load the fixed model-slot mapping from a physical YAML config only.

    This parser does not instantiate a robot.  Requiring all three camera
    fields is deliberate: the checkpoint always consumes three images, and a
    black placeholder is not a valid physical preflight result.
    """

    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(config, Mapping):
        raise ValueError(f"Expected mapping at root of {config_path}")
    try:
        cameras = config["sensors"]["cameras"]  # type: ignore[index]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"{config_path} has no sensors.cameras mapping") from exc
    if not isinstance(cameras, Mapping):
        raise ValueError(f"{config_path} has invalid sensors.cameras")

    specs = []
    for slot in POLICY_CAMERA_SLOTS:
        raw_spec = cameras.get(slot.config_key)
        if not isinstance(raw_spec, Mapping):
            raise ValueError(f"{slot.config_key} missing from {config_path}")
        if not bool(raw_spec.get("enabled", True)):
            raise ValueError(
                f"{slot.config_key} is disabled in {config_path}; all three MolmoAct2 image slots are required"
            )
        device = str(raw_spec.get("device_id", ""))
        if not device.startswith("/dev/"):
            raise ValueError(
                f"{slot.config_key}.device_id={device!r}; the preview requires a stable V4L2 /dev/ path"
            )
        specs.append(CameraSpec(slot=slot, device=device))
    return tuple(specs)


def frame_quality(image: np.ndarray) -> Dict[str, object]:
    """Return the same inexpensive technical checks used before motor enable.

    A pass says the frame is fresh-looking, correctly shaped RGB and not
    grossly over/under-exposed.  It cannot determine task semantics; the page
    explicitly leaves lid/box and wrist-side verification to the operator.
    """

    image = np.asarray(image)
    shape_ok = image.shape == EXPECTED_SHAPE and image.dtype == np.uint8
    if image.ndim == 3 and image.shape[-1] == 3:
        luma = 0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]
        mean_luma: Optional[float] = float(np.mean(luma))
        clipped_white: Optional[float] = float(np.mean(np.all(image >= 250, axis=-1)))
    else:
        mean_luma = None
        clipped_white = None
    exposure_ok = (
        mean_luma is not None
        and clipped_white is not None
        and MIN_MEAN_LUMA <= mean_luma <= MAX_MEAN_LUMA
        and clipped_white <= MAX_CLIPPED_WHITE_FRACTION
    )
    return {
        "shape": list(image.shape),
        "dtype": str(image.dtype),
        "mean_luma": None if mean_luma is None else round(mean_luma, 1),
        "clipped_white_fraction": None if clipped_white is None else round(clipped_white, 4),
        "technical_ready": bool(shape_ok and exposure_ok),
        "reason": (
            "ready"
            if shape_ok and exposure_ok
            else (
                f"expected uint8 {EXPECTED_SHAPE}, got {image.dtype} {image.shape}"
                if not shape_ok
                else "exposure outside preflight range"
            )
        ),
    }


def _annotate_frame(image: np.ndarray, spec: CameraSpec, quality: Mapping[str, object]) -> np.ndarray:
    """Make a screenshot self-describing even outside the web UI."""

    bgr = cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_RGB2BGR)
    color = (50, 205, 50) if quality["technical_ready"] else (0, 165, 255)
    cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 54), (20, 20, 20), thickness=-1)
    cv2.putText(bgr, spec.slot.model_key, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    cv2.putText(
        bgr,
        spec.slot.title,
        (12, 47),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return bgr


class PolicyCameraPreview:
    """Own only configured V4L2 cameras and make policy-slot frames available."""

    def __init__(self, specs: Sequence[CameraSpec], fps: float, jpeg_quality: int) -> None:
        self._specs = tuple(specs)
        self._fps = float(fps)
        self._jpeg_quality = int(jpeg_quality)
        self._cameras: Dict[str, V4L2Camera] = {}
        self._frames: Dict[str, EncodedFrame] = {}
        self._errors: Dict[str, str] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._encode_thread: Optional[threading.Thread] = None

    @property
    def specs(self) -> tuple[CameraSpec, ...]:
        return self._specs

    def start(self) -> None:
        try:
            for spec in self._specs:
                self._cameras[spec.slot.model_key] = V4L2Camera(
                    spec.device,
                    width=EXPECTED_SHAPE[1],
                    height=EXPECTED_SHAPE[0],
                    fps=max(1, min(5, round(self._fps))),
                )
        except Exception:
            self.close()
            raise
        self._encode_thread = threading.Thread(
            target=self._encode_loop,
            name="yam_policy_camera_preview_encoder",
            daemon=True,
        )
        self._encode_thread.start()

    def _encode_loop(self) -> None:
        period_sec = 1.0 / self._fps
        params = [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
        by_key = {spec.slot.model_key: spec for spec in self._specs}
        while not self._stop_event.is_set():
            started = time.monotonic()
            for model_key, camera in self._cameras.items():
                try:
                    rgb, _ = camera.read()
                    quality = frame_quality(rgb)
                    bgr = _annotate_frame(rgb, by_key[model_key], quality)
                    ok, encoded = cv2.imencode(".jpg", bgr, params)
                    if not ok:
                        raise RuntimeError("OpenCV JPEG encoding failed")
                    with self._lock:
                        self._frames[model_key] = EncodedFrame(
                            jpeg=bytes(encoded), captured_at=time.time(), quality=quality
                        )
                        self._errors.pop(model_key, None)
                except Exception as exc:  # noqa: BLE001 -- keep other feeds visible
                    with self._lock:
                        self._errors[model_key] = str(exc)
            self._stop_event.wait(max(0.0, period_sec - (time.monotonic() - started)))

    def frame(self, model_key: str) -> EncodedFrame:
        with self._lock:
            result = self._frames.get(model_key)
            error = self._errors.get(model_key)
        if result is None:
            raise HTTPException(status_code=503, detail=error or "waiting for first frame")
        return result

    def status(self) -> dict[str, dict[str, object]]:
        now = time.time()
        with self._lock:
            return {
                spec.slot.model_key: {
                    "config_key": spec.slot.config_key,
                    "device": spec.device,
                    "frame_age_sec": (
                        None
                        if spec.slot.model_key not in self._frames
                        else round(now - self._frames[spec.slot.model_key].captured_at, 3)
                    ),
                    "error": self._errors.get(spec.slot.model_key),
                    "quality": (
                        None
                        if spec.slot.model_key not in self._frames
                        else self._frames[spec.slot.model_key].quality
                    ),
                }
                for spec in self._specs
            }

    def close(self) -> None:
        self._stop_event.set()
        if self._encode_thread is not None:
            self._encode_thread.join(timeout=2.0)
            self._encode_thread = None
        for camera in self._cameras.values():
            camera.close()
        self._cameras = {}


def _page(specs: Sequence[CameraSpec]) -> str:
    cards = "\n".join(
        f'''<section class="card"><h2>{html.escape(spec.slot.model_key)} <span>{html.escape(spec.slot.title)}</span></h2>
<p class="source">config: <code>{html.escape(spec.slot.config_key)}</code><br><code>{html.escape(spec.device)}</code></p>
<img id="{html.escape(spec.slot.model_key)}" alt="{html.escape(spec.slot.model_key)} policy image">
<p class="quality" id="{html.escape(spec.slot.model_key)}-status">waiting for first frame…</p>
<p class="check">{html.escape(spec.slot.operator_check)}</p></section>'''
        for spec in specs
    )
    model_keys = ", ".join(repr(spec.slot.model_key) for spec in specs)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>YAM policy camera preflight</title><style>
body {{ margin: 0; background: #111827; color: #f9fafb; font: 16px system-ui, sans-serif; }}
header {{ padding: 16px 22px; border-bottom: 1px solid #374151; }}
h1 {{ margin: 0; font-size: 22px; }} header p {{ margin: 6px 0 0; color: #cbd5e1; max-width: 1100px; }}
main {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 16px; padding: 16px; }}
.card {{ background: #1f2937; border-radius: 10px; padding: 12px; min-width: 0; }}
h2 {{ margin: 0 0 7px; font: 700 18px ui-monospace, monospace; color: #67e8f9; }} h2 span {{ font: 500 14px system-ui; color: #e5e7eb; }}
.source, .quality {{ margin: 6px 0; color: #cbd5e1; font: 12px ui-monospace, monospace; overflow-wrap: anywhere; }}
img {{ display: block; width: 100%; aspect-ratio: 16 / 9; object-fit: contain; background: #030712; border-radius: 6px; }}
.check {{ margin: 9px 0 2px; color: #fef3c7; line-height: 1.35; }}
.ready {{ color: #86efac; }} .not-ready {{ color: #fbbf24; }}
#checklist {{ margin: 12px 22px 0; background: #172554; border: 1px solid #1d4ed8; border-radius: 8px; padding: 12px 16px; }}
#checklist b {{ color: #93c5fd; }} ul {{ margin: 7px 0 0; padding-left: 20px; }}
</style></head><body><header><h1>MolmoAct2 policy camera preflight</h1>
<p>Exact model image order: <code>[top_cam, left_cam, right_cam]</code>. This is camera-only: it never imports CAN or enables an arm.</p></header>
<section id="checklist"><b>Required visual check before a rollout:</b><ul><li><code>top_cam</code> contains the red lid and black box together.</li><li><code>left_cam</code> is physically the left wrist view.</li><li><code>right_cam</code> is physically the right wrist view.</li><li>Every card says technical ready; otherwise fix exposure/cabling before launching.</li></ul></section>
<main>{cards}</main><script>
const keys = [{model_keys}];
function statusText(item) {{
  if (item.error) return `ERROR: ${{item.error}}`;
  if (!item.quality) return 'waiting for first frame…';
  const q = item.quality;
  return `${{q.technical_ready ? 'TECHNICAL READY' : 'NOT READY'}} · ${{q.reason}} · ${{q.shape.join('×')}} · luma=${{q.mean_luma}} · white=${{q.clipped_white_fraction}} · age=${{item.frame_age_sec}}s`;
}}
async function refresh() {{
  const stamp = Date.now();
  for (const key of keys) document.getElementById(key).src = `/frame/${{key}}.jpg?t=${{stamp}}`;
  try {{
    const all = await (await fetch('/status', {{cache: 'no-store'}})).json();
    for (const key of keys) {{
      const el = document.getElementById(`${{key}}-status`); const item = all[key];
      el.textContent = statusText(item); el.className = 'quality ' + (item.quality?.technical_ready ? 'ready' : 'not-ready');
    }}
  }} catch (error) {{
    for (const key of keys) document.getElementById(`${{key}}-status`).textContent = `status unavailable: ${{error}}`;
  }}
  setTimeout(refresh, 250);
}}
refresh();
</script></body></html>"""


def make_app(preview: PolicyCameraPreview) -> FastAPI:
    app = FastAPI(title="YAM policy camera preflight", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _page(preview.specs)

    @app.get("/frame/{model_key}.jpg")
    def frame(model_key: str) -> Response:
        if model_key not in {spec.slot.model_key for spec in preview.specs}:
            raise HTTPException(status_code=404, detail=f"unknown policy image slot {model_key!r}")
        result = preview.frame(model_key)
        return Response(
            content=result.jpeg,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store, max-age=0"},
        )

    @app.get("/status")
    def status() -> dict[str, dict[str, object]]:
        return preview.status()

    return app


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("examples/yam/configs/yam_left_physical.yaml"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9092)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.fps <= 0:
        raise SystemExit("--fps must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        raise SystemExit("--jpeg-quality must be in [1, 100]")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    specs = load_policy_camera_specs(args.config)
    for spec in specs:
        print(
            f"[policy-camera-preview] {spec.slot.model_key} <- {spec.slot.config_key}: {spec.device}",
            flush=True,
        )
    preview = PolicyCameraPreview(specs, fps=args.fps, jpeg_quality=args.jpeg_quality)

    def _request_stop(_signum: int, _frame: object) -> None:
        preview.close()

    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    preview.start()
    print(f"[policy-camera-preview] Open: http://{args.host}:{args.port}", flush=True)
    print("[policy-camera-preview] Camera-only process; Ctrl-C releases V4L2 devices.", flush=True)
    try:
        uvicorn.run(make_app(preview), host=args.host, port=args.port, log_level="warning")
    finally:
        preview.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
