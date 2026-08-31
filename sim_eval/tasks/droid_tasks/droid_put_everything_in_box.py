"""
Put-Everything-In-Box Task for the DROID (Franka FR3 + Robotiq) Robot.

Two fixed YCB objects (lego duplo on the left, tennis ball on the right)
sit in front of the robot beside an open-top box. The goal is to pick up
both objects and drop them inside the box.

Prompt: "put everything into the box"
"""

from typing import Any, Dict

import numpy as np
import sapien
import sapien.physx as physx
import torch
from transforms3d.euler import euler2quat

import mani_skill.envs.utils.randomization as randomization
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose

from ...robots.franka_droid import FrankaDROID  # noqa: F401 (registers agent)


def _build_open_box(scene, name, inner_half, height, wall_thickness,
                    color=None, body_type="static", initial_pose=None):
    """Open-top box: square floor + 4 side walls, no lid."""
    builder = scene.create_actor_builder()
    ih, wt, h = inner_half, wall_thickness, height
    outer = ih + wt
    mat = sapien.render.RenderMaterial(
        base_color=color if color is not None else [0.55, 0.35, 0.18, 1.0]
    )
    floor_pose = sapien.Pose(p=[0, 0, wt / 2])
    builder.add_box_visual(pose=floor_pose, half_size=[outer, outer, wt / 2], material=mat)
    builder.add_box_collision(pose=floor_pose, half_size=[outer, outer, wt / 2])
    wall_z = wt + h / 2
    for sx in (+1.0, -1.0):
        pose = sapien.Pose(p=[sx * (ih + wt / 2), 0, wall_z])
        builder.add_box_visual(pose=pose, half_size=[wt / 2, outer, h / 2], material=mat)
        builder.add_box_collision(pose=pose, half_size=[wt / 2, outer, h / 2])
    for sy in (+1.0, -1.0):
        pose = sapien.Pose(p=[0, sy * (ih + wt / 2), wall_z])
        builder.add_box_visual(pose=pose, half_size=[ih, wt / 2, h / 2], material=mat)
        builder.add_box_collision(pose=pose, half_size=[ih, wt / 2, h / 2])
    if initial_pose is not None:
        builder.initial_pose = initial_pose
    return builder.build_static(name=name) if body_type == "static" else builder.build(name=name)


@register_env("DroidPutEverythingInBox-v1", max_episode_steps=400, asset_download_ids=["ycb"])
class DroidPutEverythingInBoxEnv(BaseEnv):
    """
    Put-Everything-In-Box task for the DROID robot.

    A lego duplo sits on the left, a tennis ball on the right. The robot
    must place both inside the open-top box in the center.

    Success: both objects' centers are within the box interior.
    """

    SUPPORTED_ROBOTS = ["franka_droid"]
    agent: FrankaDROID

    # Fixed objects: (ycb_id, anchor_xy)
    _objects_cfg = [
        ("073-a_lego_duplo", (-0.40,  0.20)),   # left  (+y)
        ("056_tennis_ball",  (-0.40, -0.20)),   # right (-y)
    ]
    spawn_noise = 0.02
    object_friction = 1.0

    box_inner_half = 0.09
    box_height = 0.06
    box_wall = 0.008
    box_pos = (-0.40, 0.0)

    robot_base_p = [-0.85, 0.0, -0.2]
    robot_base_rpy = (0.0, 0.0, 0.0)

    def __init__(
        self,
        *args,
        robot_uids="franka_droid",
        robot_init_qpos_noise=0.02,
        reset_robot_qpos=True,
        reconfiguration_freq=1,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.reset_robot_qpos = reset_robot_qpos
        super().__init__(
            *args,
            robot_uids=robot_uids,
            reconfiguration_freq=reconfiguration_freq,
            **kwargs,
        )

    @property
    def _robot_base_pose(self) -> sapien.Pose:
        return sapien.Pose(p=self.robot_base_p, q=euler2quat(*self.robot_base_rpy))

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.1, 0.0, 0.5], target=[-0.4, 0.0, 0.05])
        return [CameraConfig("base_camera", pose, 640, 480, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.2, 0.2, 0.5], target=[-0.4, 0.0, 0.05])
        return CameraConfig("render_camera", pose=pose, width=1280, height=720,
                            fov=1.2, near=0.01, far=100, shader_pack="rt")

    def _load_agent(self, options: dict):
        super()._load_agent(options, self._robot_base_pose)

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.objects = []
        for ycb_id, _ in self._objects_cfg:
            builder = actors.get_actor_builder(self.scene, id=f"ycb:{ycb_id}")
            builder.initial_pose = sapien.Pose()
            obj = builder.build(name=f"obj_{ycb_id}")
            for body in obj._bodies:
                for cs in body.get_collision_shapes():
                    cs.physical_material = physx.PhysxMaterial(
                        static_friction=self.object_friction,
                        dynamic_friction=self.object_friction,
                        restitution=0.0,
                    )
            self.objects.append(obj)

        bx, by = self.box_pos
        self.box = _build_open_box(
            self.scene, name="open_box",
            inner_half=self.box_inner_half, height=self.box_height,
            wall_thickness=self.box_wall, color=[0.55, 0.35, 0.18, 1.0],
            body_type="static", initial_pose=sapien.Pose(p=[bx, by, 0.0]),
        )

    def _after_reconfigure(self, options: dict):
        object_zs = []
        for obj in self.objects:
            collision_mesh = obj.get_first_collision_mesh()
            object_zs.append(-collision_mesh.bounding_box.bounds[0, 2])
        self.object_zs = common.to_tensor(object_zs, device=self.device)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            self.agent.robot.set_pose(self._robot_base_pose)
            if self.reset_robot_qpos:
                self.agent.robot.set_qpos(self.agent.keyframes["rest"].qpos)

            for i, (_, (ax, ay)) in enumerate(self._objects_cfg):
                xyz = torch.zeros((b, 3))
                xyz[:, 0] = ax + (torch.rand((b,)) * 2 - 1) * self.spawn_noise
                xyz[:, 1] = ay + (torch.rand((b,)) * 2 - 1) * self.spawn_noise
                xyz[:, 2] = self.object_zs[i]
                qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
                self.objects[i].set_pose(Pose.create_from_pq(xyz, qs))

    def _get_obs_extra(self, info: dict) -> Dict[str, Any]:
        bx, by = self.box_pos
        box_center = torch.tensor(
            [bx, by, self.box_wall], device=self.device
        ).reshape(1, 3).expand(self.num_envs, 3)
        return dict(tcp_pose=self.agent.tcp.pose.raw_pose, box_center=box_center)

    def evaluate(self) -> Dict[str, Any]:
        bx, by = self.box_pos
        per_obj = []
        for obj in self.objects:
            p = obj.pose.p
            inside_xy = (torch.abs(p[:, 0] - bx) < self.box_inner_half) & \
                        (torch.abs(p[:, 1] - by) < self.box_inner_half)
            z_ok = (p[:, 2] > self.box_wall - 0.01) & \
                   (p[:, 2] < self.box_wall + self.box_height + 0.05)
            per_obj.append(inside_xy & z_ok)
        in_box = torch.stack(per_obj, dim=0)
        n_in_box = in_box.float().sum(dim=0)
        return {
            "success": in_box.all(dim=0),
            "n_in_box": n_in_box,
            "n_total": torch.full_like(n_in_box, len(self.objects)),
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict) -> torch.Tensor:
        reward = info["n_in_box"] / float(len(self.objects)) * 5.0
        reward[info["success"]] = 5.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict) -> torch.Tensor:
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5.0
