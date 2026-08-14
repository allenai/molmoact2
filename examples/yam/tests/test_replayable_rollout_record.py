"""The two holes that made a prefetched rollout unreplayable, closed.

Both were silent: the artifact looked complete, and a replay built on it would
have compared the model against inputs it never received. That is the same
failure class as the state-plumbing confound of 2026-08-14, which cost a
session and produced a retracted finding.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from eval_utils import EvalRolloutSaver


def _frame(value: int) -> np.ndarray:
    return np.full((8, 12, 3), value, dtype=np.uint8)


def _observation(value: int) -> dict:
    return {
        "left_camera_rgb": _frame(value),
        "front_camera_rgb": _frame(value + 1),
        "right_camera_rgb": _frame(value + 2),
        "joint_positions": np.zeros(14, dtype=np.float32),
    }


class PolicyObservationRecordTest(unittest.TestCase):
    def test_a_prefetched_query_saves_the_frames_it_actually_consumed(self):
        """The step's own PNG is a DIFFERENT capture; both must survive."""

        with tempfile.TemporaryDirectory() as directory:
            rollout = Path(directory) / "rollout"
            saver = EvalRolloutSaver(rollout, instruction="pick up the red lid")
            try:
                control = _observation(10)
                saver.add_step(obs_pre=control, obs_post=control)
                # The prefetch thread captures again, later, at the same step.
                saver.add_policy_observation(0, _observation(200))
            finally:
                saver.flush()

            control_png = rollout / "left_rgb" / "000000.png"
            policy_png = rollout / "left_rgb_policy" / "000000.png"
            self.assertTrue(control_png.is_file())
            self.assertTrue(
                policy_png.is_file(),
                "the prefetched query's own frames were not saved, so that act "
                "cannot be replayed against what the model saw",
            )
            with Image.open(control_png) as image:
                self.assertEqual(int(np.asarray(image)[0, 0, 0]), 10)
            with Image.open(policy_png) as image:
                self.assertEqual(int(np.asarray(image)[0, 0, 0]), 200)

    def test_the_control_record_is_not_overwritten_by_the_policy_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            rollout = Path(directory) / "rollout"
            saver = EvalRolloutSaver(rollout, instruction="x")
            try:
                for step in range(3):
                    observation = _observation(step)
                    saver.add_step(obs_pre=observation, obs_post=observation)
                    saver.add_policy_observation(step, _observation(100 + step))
            finally:
                saver.flush()
            self.assertEqual(
                sorted(path.name for path in (rollout / "front_rgb").glob("*.png")),
                ["000000.png", "000001.png", "000002.png"],
            )
            self.assertEqual(
                sorted(path.name for path in (rollout / "front_rgb_policy").glob("*.png")),
                ["000000.png", "000001.png", "000002.png"],
            )


if __name__ == "__main__":
    unittest.main()
