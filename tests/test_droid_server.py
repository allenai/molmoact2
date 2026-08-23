import threading
import unittest
from types import SimpleNamespace

import numpy as np

from examples.droid.host_server_droid import Policy


class _RecordingModel:
    def __init__(self) -> None:
        self.kwargs = None

    def predict_action(self, **kwargs):
        if "action_mode" in kwargs:
            raise TypeError("unexpected keyword argument 'action_mode'")
        self.kwargs = kwargs
        return SimpleNamespace(actions=np.zeros((1, 15, 8), dtype=np.float32))


class DroidServerTest(unittest.TestCase):
    def test_policy_selects_continuous_inference_mode(self) -> None:
        model = _RecordingModel()
        policy = object.__new__(Policy)
        policy.processor = object()
        policy.model = model
        policy._lock = threading.Lock()

        image = np.zeros((16, 16, 3), dtype=np.uint8)
        actions = policy.predict(
            external_cam=image,
            wrist_cam=image,
            instruction="pick up the object",
            state=np.zeros(8, dtype=np.float32),
        )

        self.assertEqual(actions.shape, (15, 8))
        self.assertEqual(model.kwargs["inference_action_mode"], "continuous")
        self.assertNotIn("action_mode", model.kwargs)


if __name__ == "__main__":
    unittest.main()
