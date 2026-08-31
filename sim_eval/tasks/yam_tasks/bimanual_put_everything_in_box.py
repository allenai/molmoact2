"""
Put-Everything-In-Box Task for the YAM Bimanual Robot.

Two fixed YCB objects (lego duplo left, tennis ball right) sit beside an
open-top box in the middle of the table. The robot must place both inside.

Prompt: "put everything into the box"
"""

from typing import Any, Dict

import numpy as np
import sapien
import torch

import mani_skill.envs.utils.randomization as randomization
import sapien.physx as physx
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose

from ...robots.bimanual_yam import BimanualYAM  # noqa: F401 (registers agent)


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


@register_env("BimanualYAMPutEverythingInBox-v1", max_episode_steps=400, asset_download_ids=["ycb"])
class BimanualYAMPutEverythingInBoxEnv(BaseEnv):
    """
    Put-Everything-In-Box task for the YAM bimanual robot.

    Two fixed objects (lego duplo on the left, tennis ball on the right) start
    beside an open-top box. The robot must place both inside the box.

    Success: every object's center is within the box interior (X/Y) and below the rim (Z).
    """

    SUPPORTED_ROBOTS = ["bimanual_yam"]
    agent: BimanualYAM

    # Fixed objects: (ycb_id, anchor_xy)  left = +y, right = -y
    _objects_cfg = [
        ("073-a_lego_duplo", (-0.30,  0.22)),   # left
        ("056_tennis_ball",  (-0.30, -0.22)),   # right
    ]

    spawn_noise = 0.02
    object_friction = 1.0

    box_inner_half = 0.09
    box_height = 0.06
    box_wall = 0.008
    box_pos = (-0.15, 0.0)

    def __init__(
        self,
        *args,
        robot_uids="bimanual_yam",
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
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[-0.65, 0.0, 0.5], target=[-0.4, 0.0, 0.1])
        return [CameraConfig("base_camera", pose, 640, 480, np.pi / 2, 0.01, 100)]

    # @property
    # def _default_human_render_camera_configs(self):
    #     w, h = self.agent.cam_width, self.agent.cam_height
    #     K = self.agent._intrinsic_from_hfov(w, h, self.agent.top_cam_hfov_deg)
    #     return CameraConfig(
    #         uid="render_camera",
    #         pose=sapien.Pose(
    #             p=[0.15, 0, 0.8],
    #             q=[0.7660444431189782, 0, 0.6427876096865391, 0],
    #         ),
    #         width=w, height=h, intrinsic=K, near=0.01, far=100,
    #         mount=self.agent.robot.links_map["bimanual_base"],
    #     )
    
    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(eye=[0.2, 0.2, 0.5], target=[-0.4, 0.0, 0.05])
        return CameraConfig("render_camera", pose=pose, width=1280, height=720,
                            fov=1.2, near=0.01, far=100, shader_pack="rt")

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
            self.scene,
            name="open_box",
            inner_half=self.box_inner_half,
            height=self.box_height,
            wall_thickness=self.box_wall,
            color=[0.55, 0.35, 0.18, 1.0],
            body_type="static",
            initial_pose=sapien.Pose(p=[bx, by, 0.0]),
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

            self.agent.robot.set_pose(sapien.Pose(p=[-0.65, 0, 0.01]))
            if self.reset_robot_qpos:
                self.agent.robot.set_qpos(self.agent.keyframes["home"].qpos)

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
        obs = dict(box_center=box_center)
        for i, (ycb_id, _) in enumerate(self._objects_cfg):
            obs[f"{ycb_id}_pose"] = self.objects[i].pose.raw_pose
        return obs

    def evaluate(self) -> Dict[str, Any]:
        bx, by = self.box_pos
        per_obj = []
        for obj in self.objects:
            p = obj.pose.p
            dx = torch.abs(p[:, 0] - bx)
            dy = torch.abs(p[:, 1] - by)
            inside_xy = (dx < self.box_inner_half) & (dy < self.box_inner_half)
            z_ok = (p[:, 2] > self.box_wall - 0.01) & (p[:, 2] < self.box_wall + self.box_height + 0.05)
            per_obj.append(inside_xy & z_ok)
        in_box = torch.stack(per_obj, dim=0)   # (N, b)
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
