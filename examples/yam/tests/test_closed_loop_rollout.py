"""Hardware-free regressions for the physical YAM rollout adapter."""

import threading
import time
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
from PIL import Image

import launch_yaml_eval_molmoact as launcher
from eval_utils import EvalRolloutSaver
from gello_min.env import RobotEnv
from gello_min.yam import YAMRobot
from gello_min.v4l2_camera import V4L2Camera
from molmoact_client import require_bimanual_state
from rerun_export_watchdog import export_pending_rollouts
from rerun_rollout import write_rollout_rrd


class _OneDofRobot:
    def num_dofs(self):
        return 1


class _OneDofEnv:
    def __init__(self):
        self.position = np.array([0.0], dtype=np.float32)
        self._robot = _OneDofRobot()

    def robot(self):
        return self._robot

    def get_obs(self):
        return {
            "joint_positions": self.position.copy(),
            "joint_velocities": np.zeros(1, dtype=np.float32),
            "ee_pos_quat": np.zeros(7, dtype=np.float32),
            "gripper_position": np.zeros(1, dtype=np.float32),
        }


class _Policy:
    def __init__(self):
        self.plan_positions = []

    def prepare_input(self, obs, _instruction):
        return obs

    def inference(self, obs):
        position = float(obs["joint_positions"][0])
        self.plan_positions.append(position)
        # Absolute targets: the output has to be based on the live pose at a
        # chunk boundary, rather than on a prefetch from the prior chunk.
        return {"actions": np.array([[position + 1.0], [position + 2.0]], dtype=np.float32)}


class _Saver:
    def __init__(self):
        self.steps = []
        self.policy_chunks = []

    def add_step(self, **kwargs):
        self.steps.append(kwargs)

    def add_policy_action_chunk(self, **kwargs):
        self.policy_chunks.append(kwargs)


class _LiveView:
    def update(self, **_kwargs):
        return None


class _FourteenDofRobot:
    def num_dofs(self):
        return 14


class _FourteenDofEnv:
    def __init__(self):
        self.position = np.arange(14, dtype=np.float32)
        self._robot = _FourteenDofRobot()

    def robot(self):
        return self._robot

    def get_obs(self):
        return {
            "joint_positions": self.position.copy(),
            "joint_velocities": np.zeros(14, dtype=np.float32),
            "ee_pos_quat": np.zeros(14, dtype=np.float32),
            "gripper_position": np.array([self.position[6], self.position[13]]),
        }


class _FourteenDofPolicy:
    def __init__(self):
        self.states = []

    def prepare_input(self, obs, _instruction):
        state = np.asarray(obs["joint_positions"], dtype=np.float32)
        self.states.append(state.copy())
        return {"state": state}

    def inference(self, _input):
        # Each half is deliberately distinct. The test can prove the inactive
        # half was masked from execution instead of merely being ignored by a
        # seven-DoF crop.
        return {
            "actions": np.array(
                [
                    np.concatenate((np.full(7, 100.0), np.full(7, 200.0))),
                    np.concatenate((np.full(7, 101.0), np.full(7, 201.0))),
                ],
                dtype=np.float32,
            )
        }


class ClosedLoopRolloutTests(unittest.TestCase):
    def test_single_arm_prompt_is_explicit_and_rejects_other_side(self):
        self.assertEqual(
            launcher.with_active_single_arm_instruction("pick up the lid", "right"),
            "pick up the lid using the right arm.",
        )
        self.assertEqual(
            launcher.with_active_single_arm_instruction("Use the left arm.", "left"),
            "Use the left arm.",
        )
        with self.assertRaisesRegex(ValueError, "only the right arm"):
            launcher.with_active_single_arm_instruction("Use the left arm.", "right")

    def test_absolute_action_chunk_is_replanned_from_live_state(self):
        env = _OneDofEnv()
        policy = _Policy()
        saver = _Saver()
        applied = []

        def apply_action(fake_env, action):
            fake_env.position = np.asarray(action, dtype=np.float32).copy()
            applied.append(float(fake_env.position[0]))
            return fake_env.get_obs()

        with patch.object(launcher, "dynamic_smoothing", side_effect=apply_action):
            launcher.run_one_rollout(
                env=env,
                policy=policy,
                saver=saver,
                instruction="test",
                rollout_idx=0,
                num_rollouts=1,
                max_steps=4,
                live_view=_LiveView(),
            )

        # The old async-prefetch implementation yielded [1, 2, 1, 2] by
        # planning the second chunk from q=0.  A correct absolute-pose loop
        # replans at q=2 and yields a monotonic [1, 2, 3, 4].
        self.assertEqual(policy.plan_positions, [0.0, 2.0])
        self.assertEqual(applied, [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(len(saver.steps), 4)
        self.assertEqual(len(saver.policy_chunks), 2)
        np.testing.assert_array_equal(saver.steps[0]["action"], [1.0])
        self.assertEqual(saver.steps[2]["policy_chunk_index"], 1)

    def test_bimanual_right_active_mask_uses_live_feedback_for_left_hold(self):
        env = _FourteenDofEnv()
        policy = _FourteenDofPolicy()
        saver = _Saver()
        applied = []
        mask = launcher.BimanualActiveArmHoldMask(active_arm_side="right")

        def apply_action(fake_env, action):
            fake_env.position = np.asarray(action, dtype=np.float32).copy()
            applied.append(fake_env.position.copy())
            return fake_env.get_obs()

        with patch.object(launcher, "dynamic_smoothing", side_effect=apply_action):
            launcher.run_one_rollout(
                env=env,
                policy=policy,
                saver=saver,
                instruction="use the right arm",
                rollout_idx=0,
                num_rollouts=1,
                max_steps=2,
                live_view=_LiveView(),
                execution_mask=mask,
            )

        # The policy saw an actual fourteen-field state, not a fake padded
        # seven-field input. Its raw left-half predictions remain recorded,
        # while the applied left target is fresh encoder feedback at each tick.
        self.assertEqual(policy.states[0].shape, (14,))
        np.testing.assert_array_equal(
            saver.policy_chunks[0]["actions"][0, :7], np.full(7, 100.0)
        )
        np.testing.assert_array_equal(applied[0][:7], np.arange(7, dtype=np.float32))
        np.testing.assert_array_equal(applied[1][:7], np.arange(7, dtype=np.float32))
        np.testing.assert_array_equal(applied[0][7:], np.full(7, 200.0))
        np.testing.assert_array_equal(applied[1][7:], np.full(7, 201.0))
        np.testing.assert_array_equal(saver.steps[0]["action"], applied[0])

    def test_bimanual_shadow_mask_commands_only_live_feedback(self):
        mask = launcher.BimanualActiveArmHoldMask(
            active_arm_side="left", execution_mode="shadow"
        )
        measured = np.arange(14, dtype=np.float32)
        model_action = np.full(14, 123.0, dtype=np.float32)
        np.testing.assert_array_equal(mask.command_target(model_action, measured), measured)

    def test_bimanual_execution_requires_explicit_side_and_order(self):
        with self.assertRaisesRegex(ValueError, "requires --active-arm-side"):
            launcher.resolve_bimanual_execution_mask(
                bimanual=True, active_arm_side=None, execution_mode=None
            )
        with self.assertRaisesRegex(ValueError, "requires --right-config-path"):
            launcher.resolve_bimanual_execution_mask(
                bimanual=False, active_arm_side=None, execution_mode=None
            )
        launcher.validate_bimanual_model_arm_order(
            {"model_arm_side": "left"}, {"model_arm_side": "right"}
        )
        with self.assertRaisesRegex(ValueError, "model_arm_side"):
            launcher.validate_bimanual_model_arm_order(
                {"model_arm_side": "right"}, {"model_arm_side": "left"}
            )

    def test_policy_state_gate_rejects_unsafe_single_arm_padding(self):
        with self.assertRaisesRegex(ValueError, "unsupported 7-D pad/crop"):
            require_bimanual_state(np.zeros(7, dtype=np.float32), source="test")
        state = np.arange(14, dtype=np.float32)
        np.testing.assert_array_equal(require_bimanual_state(state, source="test"), state)


class RolloutRecordingTests(unittest.TestCase):
    def test_saved_actions_and_offline_rerun_recording(self):
        """Rerun conversion is post-hoc and has the exact applied commands."""
        with tempfile.TemporaryDirectory() as temp_dir:
            rollout_dir = Path(temp_dir) / "rollout"
            saver = EvalRolloutSaver(rollout_dir, instruction="move the red lid")
            action_chunk = np.array([[0.1] * 7, [0.2] * 7], dtype=np.float32)
            saver.add_policy_action_chunk(
                start_step=0,
                actions=action_chunk,
                inference_sec=0.75,
            )
            image = np.full((8, 10, 3), 42, dtype=np.uint8)
            for step, action in enumerate(action_chunk):
                state = np.full(7, float(step), dtype=np.float32)
                next_state = state + 0.05
                saver.add_step(
                    obs_pre={"joint_positions": state, "left_camera_rgb": image},
                    obs_post={"joint_positions": next_state},
                    action=action,
                    policy_chunk_index=0,
                    policy_action_index=step,
                    policy_inference_sec=0.75 if step == 0 else float("nan"),
                )
            saver.flush()

            with h5py.File(rollout_dir / "episode.h5", "r") as h5:
                np.testing.assert_allclose(h5["action"][:], action_chunk)
                self.assertEqual(h5["policy_action_chunks/000000"].attrs["start_step"], 0)
                self.assertAlmostEqual(
                    float(h5["policy_action_chunks/000000"].attrs["inference_sec"]),
                    0.75,
                )

            rrd_path = write_rollout_rrd(rollout_dir, image_stride=1, jpeg_quality=70)
            self.assertTrue(rrd_path.is_file())
            self.assertGreater(rrd_path.stat().st_size, 0)

    def test_missing_last_camera_frame_preserves_hdf5_and_full_replay(self):
        """One camera hiccup must not discard every other saved artifact."""
        with tempfile.TemporaryDirectory() as temp_dir:
            rollout_dir = Path(temp_dir) / "rollout"
            saver = EvalRolloutSaver(rollout_dir, instruction="sparse camera regression")
            image = np.full((8, 10, 3), 77, dtype=np.uint8)
            for step in range(2):
                pre = {
                    "joint_positions": np.full(7, step, dtype=np.float32),
                    "left_camera_rgb": image,
                    "front_camera_rgb": image,
                }
                if step == 0:
                    pre["right_camera_rgb"] = image
                saver.add_step(
                    obs_pre=pre,
                    obs_post={"joint_positions": np.full(7, step + 0.1, dtype=np.float32)},
                    action=np.full(7, step, dtype=np.float32),
                )
            saver.flush()

            self.assertTrue((rollout_dir / "episode.h5").is_file())
            self.assertTrue((rollout_dir / "rollout.raw_complete.json").is_file())
            self.assertTrue((rollout_dir / "left_rgb" / "000001.png").is_file())
            self.assertTrue((rollout_dir / "front_rgb" / "000001.png").is_file())
            self.assertFalse((rollout_dir / "right_rgb" / "000001.png").exists())
            with h5py.File(rollout_dir / "episode.h5", "r") as h5:
                self.assertEqual(h5["state"].shape, (2, 7))
                self.assertEqual(h5["action"].shape, (2, 7))

            rrd_path = write_rollout_rrd(rollout_dir, image_stride=1, jpeg_quality=70)
            self.assertTrue(rrd_path.is_file())
            self.assertGreater(rrd_path.stat().st_size, 0)

    def test_hdf5_is_published_before_png_compression_failure(self):
        """An interrupted/failed image flush must retain full action telemetry."""
        with tempfile.TemporaryDirectory() as temp_dir:
            rollout_dir = Path(temp_dir) / "rollout"
            saver = EvalRolloutSaver(rollout_dir, instruction="publish telemetry first")
            image = np.full((8, 10, 3), 61, dtype=np.uint8)
            saver.add_step(
                obs_pre={
                    "joint_positions": np.zeros(7, dtype=np.float32),
                    "left_camera_rgb": image,
                },
                obs_post={"joint_positions": np.ones(7, dtype=np.float32)},
                action=np.full(7, 0.25, dtype=np.float32),
            )

            with patch("eval_utils._save_png", side_effect=RuntimeError("simulated png failure")):
                with self.assertRaisesRegex(RuntimeError, "simulated png failure"):
                    saver.flush()

            # ``raw_complete`` is intentionally absent because the camera
            # payload did not finish, but the detached exporter can now use
            # this atomically published file for the complete numeric replay.
            self.assertTrue((rollout_dir / "episode.h5").is_file())
            self.assertFalse((rollout_dir / "rollout.raw_complete.json").exists())
            with h5py.File(rollout_dir / "episode.h5", "r") as h5:
                np.testing.assert_allclose(h5["action"][:], [[0.25] * 7])

    def test_camera_only_recovery_drops_unmatched_frame(self):
        """A pre-HDF5 crash still gets a portable, explicitly incomplete RRD."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rollout_dir = root / "eval" / "failed"
            rollout_dir.mkdir(parents=True)
            (rollout_dir / "rollout.started.json").write_text(
                '{"instruction": "recover camera frames"}\n', encoding="utf-8"
            )
            image = np.full((8, 10, 3), 99, dtype=np.uint8)
            for camera, steps in (("left", (0, 1)), ("front", (0, 1)), ("right", (0,))):
                camera_dir = rollout_dir / f"{camera}_rgb"
                camera_dir.mkdir()
                for step in steps:
                    Image.fromarray(image).save(camera_dir / f"{step:06d}.png")

            # Exercise the detached export path. It must produce a replay from
            # sparse raw PNGs without importing a camera/CAN/robot runtime.
            complete, pending = export_pending_rollouts(
                root,
                fps=30,
                image_stride=1,
                jpeg_quality=70,
                action_chunk_size=30,
            )
            self.assertEqual((complete, pending), (1, 0))
            rrd_path = rollout_dir / "rollout.rrd"
            self.assertTrue(rrd_path.is_file())
            self.assertGreater(rrd_path.stat().st_size, 0)

    def test_marker_only_recovery_still_writes_a_failure_rrd(self):
        """Even a crash before the first camera frame must not omit an RRD."""
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            rollout_dir = root / "eval" / "failed_before_first_step"
            rollout_dir.mkdir(parents=True)
            (rollout_dir / "rollout.started.json").write_text(
                '{"instruction": "marker-only recovery"}\n', encoding="utf-8"
            )

            complete, pending = export_pending_rollouts(
                root,
                fps=30,
                image_stride=1,
                jpeg_quality=70,
                action_chunk_size=30,
            )
            self.assertEqual((complete, pending), (1, 0))
            self.assertTrue((rollout_dir / "rollout.rrd").is_file())


class _I2RTFeedbackRobot:
    def __init__(self):
        self.arm_position = np.arange(6, dtype=np.float32)
        self.arm_velocity = np.arange(6, dtype=np.float32) / 10.0
        self.gripper_position = np.array([0.6], dtype=np.float32)
        self.gripper_velocity = np.array([0.6], dtype=np.float32)
        self.commands = []

    def get_observations(self):
        return {
            "joint_pos": self.arm_position.copy(),
            "joint_vel": self.arm_velocity.copy(),
            "gripper_pos": self.gripper_position.copy(),
            "gripper_vel": self.gripper_velocity.copy(),
        }

    def command_joint_pos(self, command):
        self.commands.append(np.asarray(command).copy())


class YAMFeedbackTests(unittest.TestCase):
    def test_observation_uses_encoder_feedback_not_last_command(self):
        wrapper = object.__new__(YAMRobot)
        wrapper.robot = _I2RTFeedbackRobot()
        wrapper._joint_state = np.zeros(7, dtype=np.float32)
        wrapper._joint_velocities = np.zeros(7, dtype=np.float32)
        wrapper._gripper_state = 0.0
        wrapper._last_commanded_joint_state = np.zeros(7, dtype=np.float32)

        command = np.array([99.0, 99.0, 99.0, 99.0, 99.0, 99.0, 0.75], dtype=np.float32)
        wrapper.command_joint_state(command)
        obs = wrapper.get_observations()

        np.testing.assert_array_equal(wrapper.robot.commands[-1], command)
        np.testing.assert_array_equal(
            obs["joint_positions"],
            np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.6], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            obs["joint_velocities"], np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
        )


class V4L2CacheTests(unittest.TestCase):
    @staticmethod
    def _camera_with_cached_frame(timestamp):
        camera = object.__new__(V4L2Camera)
        camera.device = "fake-v4l2"
        camera._frame_ready = threading.Event()
        camera._frame_ready.set()
        camera._frame_lock = threading.Lock()
        camera._latest_rgb = np.full((3, 4, 3), 17, dtype=np.uint8)
        camera._latest_depth = np.zeros((3, 4, 1), dtype=np.uint16)
        camera._latest_frame_timestamp = timestamp
        camera._last_capture_error = None
        camera._read_wait_timeout_sec = 0.1
        camera._max_frame_age_sec = 1.0
        return camera

    def test_read_returns_cached_frame_without_waiting_for_a_new_capture(self):
        camera = self._camera_with_cached_frame(time.time())
        started = time.monotonic()
        rgb, depth = camera.read()

        self.assertLess(time.monotonic() - started, 0.05)
        np.testing.assert_array_equal(rgb, np.full((3, 4, 3), 17, dtype=np.uint8))
        self.assertEqual(depth.shape, (3, 4, 1))

    def test_read_rejects_a_stalled_capture(self):
        camera = self._camera_with_cached_frame(time.time() - 2.0)
        with self.assertRaisesRegex(RuntimeError, "stale"):
            camera.read()


class CameraPreflightTests(unittest.TestCase):
    class _Camera:
        def __init__(self, frames):
            self.frames = iter(frames)
            self.last = None

        def read(self):
            try:
                self.last = next(self.frames)
            except StopIteration:
                pass
            return self.last, None

    def test_waits_for_settled_well_exposed_frames(self):
        blown = np.full((360, 640, 3), 255, dtype=np.uint8)
        usable = np.full((360, 640, 3), 100, dtype=np.uint8)
        launcher.wait_for_camera_visual_preflight(
            {"top": self._Camera([blown, usable, usable])},
            timeout_sec=1,
            required_consecutive_frames=2,
            min_warmup_sec=0,
            poll_sec=0,
        )

    def test_rejects_wrong_policy_image_shape(self):
        wrong_shape = np.full((480, 640, 3), 100, dtype=np.uint8)
        with self.assertRaisesRegex(RuntimeError, "Camera visual preflight failed"):
            launcher.wait_for_camera_visual_preflight(
                {"top": self._Camera([wrong_shape])},
                timeout_sec=0.01,
                required_consecutive_frames=1,
                min_warmup_sec=0,
                poll_sec=0,
            )


class _ClosableRobot:
    def __init__(self):
        self.closed = 0

    def num_dofs(self):
        return 1

    def close(self):
        self.closed += 1


class _ClosableCamera:
    def __init__(self):
        self.closed = 0

    def close(self):
        self.closed += 1


class RuntimeCleanupTests(unittest.TestCase):
    def test_env_close_releases_robot_and_every_camera(self):
        robot = _ClosableRobot()
        left = _ClosableCamera()
        front = _ClosableCamera()
        env = RobotEnv(robot, camera_dict={"left": left, "front": front})

        env.close()

        self.assertEqual(robot.closed, 1)
        self.assertEqual(left.closed, 1)
        self.assertEqual(front.closed, 1)
        self.assertEqual(env._camera_dict, {})


if __name__ == "__main__":
    unittest.main()
