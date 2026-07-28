"""Minimal RGB camera driver for Jetson systems without pyrealsense2 wheels."""

from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np


class V4L2Camera:
    """Return the latest RGB frame from a V4L2 stream without blocking control.

    The MolmoAct2 YAM policy consumes RGB only; depth is returned as a zero
    placeholder so it matches the existing camera-driver interface.  A capture
    thread owns ``VideoCapture.read()`` because V4L2 waits for the next frame;
    having the 30 Hz robot loop wait for three 5 FPS cameras would otherwise
    collapse action execution to a few hertz.
    """

    def __init__(
        self,
        device: str,
        width: int = 640,
        height: int = 360,
        fps: int = 5,
        read_wait_timeout_sec: float = 2.0,
        max_frame_age_sec: float = 1.5,
    ):
        self.device = device
        self._cap = cv2.VideoCapture()
        deadline = time.time() + 10.0
        while time.time() < deadline:
            self._cap.open(device, cv2.CAP_V4L2)
            if self._cap.isOpened():
                break
            self._cap.release()
            time.sleep(0.25)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open V4L2 camera {device} after waiting 10s")
        # The YAM checkpoint was trained on 640x360 RGB frames. Request that
        # geometry explicitly. Some RealSense UVC endpoints do not expose
        # MJPG and will retain YUYV; the bounded 5 FPS capture cache remains
        # the bandwidth guard in that case.
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, min(fps, 5))
        # A one-frame kernel buffer prevents a delayed consumer from reading a
        # long queue of old images once control resumes.
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._cap_lock = threading.Lock()
        self._frame_lock = threading.Lock()
        self._frame_ready = threading.Event()
        self._stop_event = threading.Event()
        self._capture_thread: Optional[threading.Thread] = None

        self._capture_period_sec = 1.0 / max(1, min(int(fps), 5))
        self._read_wait_timeout_sec = float(read_wait_timeout_sec)
        self._max_frame_age_sec = float(max_frame_age_sec)
        self._latest_rgb: Optional[np.ndarray] = None
        self._latest_depth: Optional[np.ndarray] = None
        self._latest_frame_timestamp: Optional[float] = None
        self._last_capture_error: Optional[BaseException] = None

        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name=f"v4l2_capture_{device.rsplit('/', 1)[-1]}",
            daemon=True,
        )
        self._capture_thread.start()

    def _capture_loop(self) -> None:
        """Continuously drain the device and publish its newest RGB frame."""
        while not self._stop_event.is_set():
            started = time.monotonic()
            try:
                with self._cap_lock:
                    ok, bgr = self._cap.read()
                if not ok or bgr is None:
                    raise RuntimeError(f"No frame from V4L2 camera {self.device}")

                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                with self._frame_lock:
                    # ``cvtColor`` returns a fresh array, so consumers can
                    # safely retain the previous frame while this reference is
                    # atomically replaced on the next capture.
                    self._latest_rgb = rgb
                    self._latest_depth = np.zeros((*rgb.shape[:2], 1), dtype=np.uint16)
                    self._latest_frame_timestamp = time.time()
                    self._last_capture_error = None
                    self._frame_ready.set()
            except Exception as exc:  # noqa: BLE001 — surface through read()
                with self._frame_lock:
                    self._last_capture_error = exc
                self._stop_event.wait(0.05)
                continue

            # Some V4L2 drivers ignore CAP_PROP_FPS.  Keep capture bounded at
            # the configured 5 FPS so this cache does not increase USB load.
            remaining = self._capture_period_sec - (time.monotonic() - started)
            if remaining > 0:
                self._stop_event.wait(remaining)

    def read(self, img_size: Optional[Tuple[int, int]] = None):
        """Return a cached frame immediately after the initial capture.

        At most one initial call waits for a camera frame.  Subsequent calls
        do not wait for a new image, which lets the robot command loop remain
        at its configured rate while images stay at the USB-safe capture rate.
        """
        if not self._frame_ready.wait(timeout=self._read_wait_timeout_sec):
            raise RuntimeError(f"Timed out waiting for V4L2 camera {self.device}")

        with self._frame_lock:
            rgb = self._latest_rgb
            depth = self._latest_depth
            timestamp = self._latest_frame_timestamp
            last_error = self._last_capture_error

        if rgb is None or depth is None or timestamp is None:
            if last_error is not None:
                raise RuntimeError(f"V4L2 camera {self.device} failed to capture a frame") from last_error
            raise RuntimeError(f"V4L2 camera {self.device} has no frame yet")

        age = time.time() - timestamp
        if age > self._max_frame_age_sec:
            raise RuntimeError(
                f"V4L2 camera {self.device} frame is stale ({age:.3f}s old; "
                f"limit {self._max_frame_age_sec:.3f}s)"
            )

        if img_size is not None:
            rgb = cv2.resize(rgb, img_size)
            depth = cv2.resize(depth, img_size)
            if depth.ndim == 2:
                depth = depth[:, :, None]
        return rgb, depth

    def close(self) -> None:
        self._stop_event.set()
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2.0)
            if self._capture_thread.is_alive():
                # Do not call ``release`` concurrently with OpenCV's native
                # ``read``; several V4L2 backends can segfault in that race.
                # The daemon thread and descriptor are then left for process
                # teardown, while a normal 5 FPS capture exits within 200 ms.
                return
            self._capture_thread = None
        with self._cap_lock:
            self._cap.release()
