"""Smoke test for Servo pi0.5-yam endpoint using real rollout images."""

import base64
import json
import logging
from pathlib import Path
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("servo.smoke_test")

GRANT_PATH = Path("~/servo-demo/grants/pi05-molmoact-yam.txt").expanduser()
IMAGE_DIR = Path("examples/yam/yam_eval_runs/data/red_lid_bimanual_dryrun/eval/20260814_101810")


def image_to_data_uri(path: Path) -> str:
    with open(path, "rb") as f:
        data = f.read()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def main():
    if not GRANT_PATH.exists():
        log.error("Grant file not found at %s", GRANT_PATH)
        return 1

    top_img = IMAGE_DIR / "front_rgb" / "000000.png"
    left_img = IMAGE_DIR / "left_rgb" / "000000.png"
    right_img = IMAGE_DIR / "right_rgb" / "000000.png"

    log.info("Loading test images from %s...", IMAGE_DIR)
    top_uri = image_to_data_uri(top_img)
    left_uri = image_to_data_uri(left_img)
    right_uri = image_to_data_uri(right_img)

    log.info("Connecting to Servo endpoint via grant at %s...", GRANT_PATH)
    from servo.direct import DirectPolicy, EndpointGrant
    from servo_contracts.inference import Observation

    blob = GRANT_PATH.read_text().strip()
    grant = EndpointGrant.parse(blob)
    log.info("Parsed EndpointGrant for deployment: %s at %s", grant.name, grant.endpoint_url)

    # Attach policy handle via Servo SDK direct mode
    policy = DirectPolicy(grant)
    log.info("Attached to policy. Active binding: %s", policy.active_binding)

    obs = Observation(
        images={
            "top": top_uri,
            "left": left_uri,
            "right": right_uri,
        },
        state=[0.0] * 14,
        instruction="Using both arms, coordinate to pick up the red lid, place it fully on top of the black box, release it, then move both arms away.",
    )

    log.info("Sending test observation to Servo endpoint (/act)...")
    start = time.perf_counter()
    prediction = policy.act(obs)
    elapsed = (time.perf_counter() - start) * 1000

    actions = np.asarray(prediction.actions, dtype=np.float32)
    log.info("SMOKE TEST SUCCESSFUL!")
    log.info("Response received in %.2f ms", elapsed)
    log.info("Action shape: %s (expected (30, 14))", actions.shape)
    log.info("First step action [left(7), right(7)]:\n%s", np.round(actions[0], 4))
    return 0


if __name__ == "__main__":
    main()
