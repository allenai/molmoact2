"""Hardware-free regression coverage for YAM rollout provenance and RNGs."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from eval_utils import EvalRolloutSaver
from molmoact_client import MolmoActLocal
from rollout_manifest import RolloutSeedPlan, build_rollout_manifest


class _PolicyWithMetadata:
    def reproducibility_metadata(self):
        return {
            "backend": "fake",
            "repo_id": "fake/model",
            "snapshot_revision": "abc123",
        }


class ReproducibilityTests(unittest.TestCase):
    def test_seed_plan_is_stable_and_namespaces_rollouts_and_queries(self):
        plan = RolloutSeedPlan(1234)
        rollout_zero = plan.rollout_seed(0)
        self.assertEqual(rollout_zero, RolloutSeedPlan(1234).rollout_seed(0))
        self.assertNotEqual(rollout_zero, plan.rollout_seed(1))
        self.assertEqual(
            plan.query_seed(rollout_zero, 0),
            RolloutSeedPlan(1234).query_seed(rollout_zero, 0),
        )
        self.assertNotEqual(
            plan.query_seed(rollout_zero, 0), plan.query_seed(rollout_zero, 1)
        )

    def test_local_policy_query_generator_restarts_for_each_rollout(self):
        # Bypass model loading: only test generator plumbing on CPU.
        policy = object.__new__(MolmoActLocal)
        policy.policy = SimpleNamespace(device="cpu")
        policy._query_seed_plan = RolloutSeedPlan(None)
        policy._rollout_seed = None
        policy._inference_index = 0

        first_rollout = policy.begin_rollout(91)
        generator_one, metadata_one = policy._generator_for_next_query()
        self.assertTrue(first_rollout["deterministic_generator"])
        self.assertEqual(metadata_one["query_index"], 0)
        self.assertIsNotNone(generator_one)
        sampled_one = torch.randn(4, generator=generator_one)

        # The same rollout seed must reproduce query zero exactly, even after
        # a prior generator has been consumed.
        policy.begin_rollout(91)
        generator_two, metadata_two = policy._generator_for_next_query()
        sampled_two = torch.randn(4, generator=generator_two)
        self.assertEqual(metadata_one["query_seed"], metadata_two["query_seed"])
        torch.testing.assert_close(sampled_one, sampled_two)

    def test_local_inference_passes_the_recorded_generator_to_model(self):
        received = []

        def predict(**kwargs):
            received.append(kwargs["generator"].initial_seed())
            return np.zeros((1, 14), dtype=np.float32)

        policy = object.__new__(MolmoActLocal)
        policy.policy = SimpleNamespace(device="cpu", predict=predict)
        policy._query_seed_plan = RolloutSeedPlan(None)
        policy._rollout_seed = None
        policy._inference_index = 0
        policy.num_steps = 10
        policy.enable_cuda_graph = False
        policy.logger = SimpleNamespace(info=lambda *_args, **_kwargs: None)
        policy.begin_rollout(123)
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        result = policy.inference(
            {
                "front_camera_rgb": image,
                "left_camera_rgb": image,
                "right_camera_rgb": image,
                "instruction": "test",
                "state": np.zeros(14, dtype=np.float32),
            }
        )
        self.assertEqual(received, [result["reproducibility"]["query_seed"]])
        self.assertEqual(result["reproducibility"]["query_index"], 0)
        self.assertEqual(policy._inference_index, 1)

    def test_manifest_redacts_secret_and_records_camera_can_model_identity(self):
        config = {
            "sensors": {
                "cameras": {
                    "front_camera": {"device_id": "/dev/v4l/by-id/front"},
                    "left_camera": {"device_id": "/dev/v4l/by-id/left"},
                }
            },
            "robot": {"_target_": "fake.YAMRobot", "channel": "can_left"},
            "eval": {"local": {"hf_token": "must-not-leak"}},
        }
        manifest = build_rollout_manifest(
            instruction="move the lid",
            rollout_index=2,
            seed_plan=RolloutSeedPlan(7),
            policy=_PolicyWithMetadata(),
            primary_config=config,
            primary_config_path=None,
        )
        self.assertEqual(manifest["reproducibility"]["rollout_index"], 2)
        self.assertEqual(manifest["policy"]["repo_id"], "fake/model")
        self.assertEqual(
            manifest["hardware_configuration"]["can"][0]["channel"], "can_left"
        )
        self.assertEqual(
            {camera["role"] for camera in manifest["hardware_configuration"]["cameras"]},
            {"front_camera", "left_camera"},
        )
        self.assertEqual(
            manifest["configs"]["primary"]["resolved"]["eval"]["local"]["hf_token"],
            "<redacted>",
        )

    def test_saver_publishes_manifest_and_query_seed_metadata(self):
        manifest = build_rollout_manifest(
            instruction="move the lid",
            rollout_index=0,
            seed_plan=RolloutSeedPlan(42),
            policy=_PolicyWithMetadata(),
            primary_config={"robot": {"channel": "can_left"}},
            primary_config_path=None,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            rollout_dir = Path(temp_dir) / "rollout"
            saver = EvalRolloutSaver(
                rollout_dir,
                instruction="move the lid",
                rollout_manifest=manifest,
            )
            saver.add_policy_action_chunk(
                start_step=0,
                actions=np.zeros((1, 2), dtype=np.float32),
                inference_sec=0.1,
                metadata={"query_seed": 99, "query_index": 0},
            )
            saver.add_step(
                obs_pre={"joint_positions": np.zeros(2, dtype=np.float32)},
                obs_post={"joint_positions": np.ones(2, dtype=np.float32)},
                action=np.zeros(2, dtype=np.float32),
            )
            saver.flush()

            persisted = json.loads((rollout_dir / "rollout.manifest.json").read_text())
            self.assertEqual(persisted["reproducibility"]["base_seed"], 42)
            with h5py.File(rollout_dir / "episode.h5", "r") as h5:
                self.assertEqual(h5.attrs["rollout_manifest_file"], "rollout.manifest.json")
                chunk_metadata = json.loads(
                    h5["policy_action_chunks/000000"].attrs["metadata_json"]
                )
                self.assertEqual(chunk_metadata["query_seed"], 99)


if __name__ == "__main__":
    unittest.main()
