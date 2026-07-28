from typing import Dict, Tuple

import numpy as np

from gello_min.robot import Robot
from i2rt.robots.utils import GripperType


class YAMRobot(Robot):
    """A class representing a simulated YAM robot."""

    def __init__(self, channel="can0"):
        from i2rt.robots.get_robot import get_yam_robot

        self.robot = get_yam_robot(channel=channel, gripper_type=GripperType.LINEAR_4310)

        # YAM has 7 joints (6 arm joints + 1 gripper)
        self._joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "gripper",
        ]
        self._joint_state = np.zeros(self.num_dofs())
        self._joint_velocities = np.zeros(self.num_dofs())
        self._gripper_state = 0.0
        self._last_commanded_joint_state = self._joint_state.copy()
        # Initialize from feedback.  Never initialize to zeros: that would
        # make the first command attempt to drive the physical arm home.
        self._joint_state = self.get_joint_state()

    def num_dofs(self) -> int:
        return 7  # YAM has 7 DOFs

    def _assert_control_healthy(self) -> None:
        """Fail rather than returning cached feedback after a CAN-loop fault."""
        motor_chain = getattr(self.robot, "motor_chain", None)
        if motor_chain is None or getattr(motor_chain, "running", True):
            return
        control_error = getattr(motor_chain, "_control_error", None)
        raise RuntimeError(
            "YAM motor control is no longer running; refusing to use stale encoder feedback. "
            f"Control error: {control_error!r}"
        )

    def _read_feedback(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return coherent 7-DoF encoder feedback from the I2RT robot.

        ``MotorChainRobot.get_observations`` is guarded by its state lock and
        reports real motor feedback.  Do not use the wrapper's last command as
        an observation: absolute-pose policies must condition on where the arm
        actually is, not where it was asked to go.
        """
        self._assert_control_healthy()
        feedback = self.robot.get_observations()
        arm_pos = np.asarray(feedback["joint_pos"], dtype=np.float32).reshape(-1)
        arm_vel = np.asarray(feedback["joint_vel"], dtype=np.float32).reshape(-1)
        gripper_pos = np.asarray(feedback.get("gripper_pos", ()), dtype=np.float32).reshape(-1)
        gripper_vel = np.asarray(feedback.get("gripper_vel", ()), dtype=np.float32).reshape(-1)
        joint_pos = np.concatenate((arm_pos, gripper_pos))
        joint_vel = np.concatenate((arm_vel, gripper_vel))

        if joint_pos.size != self.num_dofs() or joint_vel.size != self.num_dofs():
            raise RuntimeError(
                "Expected 7 YAM joint-feedback values (6 arm + 1 gripper), "
                f"got positions={joint_pos.size}, velocities={joint_vel.size}"
            )
        # MolmoAct's YAM checkpoint represents gripper aperture directly in
        # [0, 1]: 0 is closed and 1 is open.  Never feed a bad motor-coordinate
        # branch into the policy as if it were an aperture.
        aperture = float(gripper_pos[0])
        if not -0.05 <= aperture <= 1.05:
            raise RuntimeError(
                "YAM gripper feedback is outside normalized policy aperture [0, 1]: "
                f"{aperture:.4f}. Check the gripper endpoint/2π branch mapping."
            )
        gripper_pos = np.clip(gripper_pos, 0.0, 1.0)
        joint_pos[-1] = gripper_pos[0]
        return joint_pos.copy(), joint_vel.copy(), gripper_pos.copy()

    def get_joint_state(self) -> np.ndarray:
        joint_pos, joint_vel, gripper_pos = self._read_feedback()
        self._joint_state = joint_pos
        self._joint_velocities = joint_vel
        self._gripper_state = float(gripper_pos[0])
        return self._joint_state.copy()

    def command_joint_state(self, joint_state: np.ndarray) -> None:
        assert (
            len(joint_state) == self.num_dofs()
        ), f"Expected {self.num_dofs()} joint values, got {len(joint_state)}"

        # Command the I2RT robot with all 7 joints (6 arm + 1 gripper)
        self._last_commanded_joint_state = np.asarray(joint_state, dtype=np.float32).copy()
        self.command_joint_pos(self._last_commanded_joint_state)

    def get_observations(self) -> Dict[str, np.ndarray]:
        joint_pos, joint_vel, gripper_pos = self._read_feedback()
        self._joint_state = joint_pos
        self._joint_velocities = joint_vel
        self._gripper_state = float(gripper_pos[0])
        ee_pos_quat = np.zeros(7)  # Placeholder for FK
        return {
            "joint_positions": joint_pos.copy(),
            "joint_velocities": joint_vel.copy(),
            "ee_pos_quat": ee_pos_quat,
            "gripper_position": gripper_pos.copy(),
        }

    def get_joint_pos(self):
        return self.get_joint_state()

    def command_joint_pos(self, target_pos):
        self._assert_control_healthy()
        # Ensure we send exactly 7 joints to the I2RT robot
        if len(target_pos) > 7:
            target_pos = target_pos[:7]
        elif len(target_pos) < 7:
            # Pad with zeros if we have fewer than 7 joints
            target_pos = np.pad(target_pos, (0, 7 - len(target_pos)), "constant")
        target_pos = np.asarray(target_pos, dtype=np.float32)
        aperture = float(target_pos[-1])
        if not -0.05 <= aperture <= 1.05:
            raise ValueError(
                "YAM policy gripper target must be normalized aperture in [0, 1], "
                f"got {aperture:.4f}"
            )
        target_pos[-1] = np.clip(target_pos[-1], 0.0, 1.0)
        self.robot.command_joint_pos(target_pos)

    def close(self) -> None:
        """Stop the I2RT control thread and disable the physical motors."""
        self.robot.close()


def main():
    robot = YAMRobot()
    print(robot.get_observations())


if __name__ == "__main__":
    main()
