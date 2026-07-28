"""Hardware-free checks for the camera-only policy-slot preflight."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

import policy_camera_preview as preview


class PolicyCameraPreviewTests(unittest.TestCase):
    def test_slot_mapping_matches_molmoact_input_order(self):
        """The human preview must not accidentally inherit YAML dict ordering."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "physical.yaml"
            OmegaConf.save(
                OmegaConf.create(
                    {
                        "sensors": {
                            "cameras": {
                                # Deliberately use a non-model order here.
                                "right_camera": {"device_id": "/dev/right"},
                                "front_camera": {"device_id": "/dev/front"},
                                "left_camera": {"device_id": "/dev/left"},
                            }
                        }
                    }
                ),
                config_path,
            )
            specs = preview.load_policy_camera_specs(config_path)

        self.assertEqual(
            [(spec.slot.model_key, spec.slot.config_key, spec.device) for spec in specs],
            [
                ("top_cam", "front_camera", "/dev/front"),
                ("left_cam", "left_camera", "/dev/left"),
                ("right_cam", "right_camera", "/dev/right"),
            ],
        )

    def test_disabled_camera_is_not_treated_as_a_valid_model_input(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "physical.yaml"
            OmegaConf.save(
                OmegaConf.create(
                    {
                        "sensors": {
                            "cameras": {
                                "front_camera": {"device_id": "/dev/front"},
                                "left_camera": {"device_id": "/dev/left", "enabled": False},
                                "right_camera": {"device_id": "/dev/right"},
                            }
                        }
                    }
                ),
                config_path,
            )
            with self.assertRaisesRegex(ValueError, "left_camera is disabled"):
                preview.load_policy_camera_specs(config_path)

    def test_quality_uses_the_motor_enable_technical_contract(self):
        ready = np.full(preview.EXPECTED_SHAPE, 100, dtype=np.uint8)
        quality = preview.frame_quality(ready)
        self.assertTrue(quality["technical_ready"])
        self.assertEqual(quality["reason"], "ready")

        wrong_shape = preview.frame_quality(np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertFalse(wrong_shape["technical_ready"])
        self.assertIn("expected uint8", str(wrong_shape["reason"]))

        overexposed = preview.frame_quality(np.full(preview.EXPECTED_SHAPE, 255, dtype=np.uint8))
        self.assertFalse(overexposed["technical_ready"])
        self.assertEqual(overexposed["reason"], "exposure outside preflight range")


if __name__ == "__main__":
    unittest.main()
