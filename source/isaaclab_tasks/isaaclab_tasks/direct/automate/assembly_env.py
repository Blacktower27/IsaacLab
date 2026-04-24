# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import json
import os

import numpy as np
import torch
import warp as wp

import carb
import isaacsim.core.utils.torch as torch_utils
from pxr import UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, check_file_path, retrieve_file_path
from isaaclab.utils.math import axis_angle_from_quat
from isaaclab_tasks.direct.factory import factory_utils

from . import automate_algo_utils as automate_algo
from . import automate_log_utils as automate_log
from . import factory_control as fc
from . import industreal_algo_utils as industreal_algo
from .assembly_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG, AssemblyEnvCfg
from .automate_forge_bridge import FORGE_TASK_NAMES, get_curr_successes_forge
from .automate_forge_reset import randomize_initial_state_forge, reset_forge_keypoints_after_randomize
from .soft_dtw_cuda import SoftDTW


def _validate_automate_usd_paths(cfg: AssemblyEnvCfg, cfg_task) -> None:
    """Raise a clear error if spawn USDs are missing (Nucleus or local).

    Missing RJ45/BNC part USDs can otherwise surface later as a cryptic
    ``Failed to create articulation ... Robot/root_joint`` from PhysX.
    """
    paths = [
        ("Robot (franka_mimic)", getattr(cfg.robot.spawn, "usd_path", None)),
        ("Fixed asset", getattr(cfg_task.fixed_asset.spawn, "usd_path", None)),
        ("Held asset", getattr(cfg_task.held_asset.spawn, "usd_path", None)),
    ]
    for label, path in paths:
        if not path:
            continue
        if check_file_path(path) == 0:
            raise FileNotFoundError(
                f"AssemblyEnv: {label} USD not found or unreachable:\n  {path}\n"
                "For default AutoMate tasks, sync Isaac Nucleus assets (Isaac/IsaacLab/AutoMate). "
                "For RJ45/BNC/BoxLid, ensure custom_assets USDs exist under isaaclab_assets/... "
                "(see assembly_tasks_cfg.py paths)."
            )


class AssemblyEnv(DirectRLEnv):
    cfg: AssemblyEnvCfg

    def __init__(self, cfg: AssemblyEnvCfg, render_mode: str | None = None, **kwargs):
        # Single-step dims + prev_actions; full policy/critic input = dim * window (matches FactoryEnv).
        self._single_obs_dim = sum([OBS_DIM_CFG[obs] for obs in cfg.obs_order]) + cfg.action_space
        self._single_state_dim = sum([STATE_DIM_CFG[state] for state in cfg.state_order]) + cfg.action_space
        cfg.observation_space = self._single_obs_dim * cfg.obs_window_size
        cfg.state_space = self._single_state_dim * cfg.state_window_size
        self.cfg_task = cfg.tasks[cfg.task_name]
        # Forge-named tasks: Factory-style success checks, held-base/keypoints, reset — not Forge reward shaping.
        self._is_forge_task = self.cfg_task.name in FORGE_TASK_NAMES

        _validate_automate_usd_paths(cfg, self.cfg_task)
        super().__init__(cfg, render_mode, **kwargs)

        self._has_sdf = (
            hasattr(self.cfg_task.held_asset_cfg, "obj_path")
            and hasattr(self.cfg_task.fixed_asset_cfg, "obj_path")
            and bool(getattr(self.cfg_task, "assembly_dir", ""))
        )
        self._has_dtw = self.cfg_task.imitation_rwd_scale > 0 and bool(getattr(self.cfg_task, "disassembly_path_json", ""))

        self._set_body_inertias()
        self._init_tensors()
        self._set_default_dynamics_parameters()
        self._compute_intermediate_values(dt=self.physics_dt)

        if self._has_sdf:
            wp.init()
            self.wp_device = wp.get_preferred_device()
            self.plug_mesh, self.plug_sample_points, self.socket_mesh = industreal_algo.load_asset_mesh_in_warp(
                self.cfg_task.assembly_dir + self.cfg_task.held_asset_cfg.obj_path,
                self.cfg_task.assembly_dir + self.cfg_task.fixed_asset_cfg.obj_path,
                self.cfg_task.num_mesh_sample_points,
                self.wp_device,
            )

        if hasattr(self.cfg_task.held_asset_cfg, "obj_path") and self.cfg_task.assembly_dir:
            self.gripper_open_width = automate_algo.get_gripper_open_width(
                self.cfg_task.assembly_dir + self.cfg_task.held_asset_cfg.obj_path
            )
        else:
            self.gripper_open_width = getattr(self.cfg_task, "gripper_open_width_override", 0.02)

        if self._has_dtw:
            cuda_version = automate_algo.get_cuda_version()
            if (cuda_version is not None) and (cuda_version < (13, 0, 0)):
                self.soft_dtw_criterion = SoftDTW(use_cuda=True, device=self.device, gamma=self.cfg_task.soft_dtw_gamma)
            else:
                self.soft_dtw_criterion = SoftDTW(use_cuda=False, device=self.device, gamma=self.cfg_task.soft_dtw_gamma)

        # Evaluate
        if self.cfg_task.if_logging_eval:
            self._init_eval_logging()

    def _init_eval_logging(self):
        self.held_asset_pose_log = torch.empty(
            (0, 7), dtype=torch.float32, device=self.device
        )  # (position, quaternion)
        self.fixed_asset_pose_log = torch.empty((0, 7), dtype=torch.float32, device=self.device)
        self.success_log = torch.empty((0, 1), dtype=torch.float32, device=self.device)

        # Turn off SBC during evaluation so all plugs are initialized outside of the socket
        self.cfg_task.if_sbc = False

    def _set_body_inertias(self):
        """Note: this is to account for the asset_options.armature parameter in IGE."""
        inertias = self._robot.root_physx_view.get_inertias()
        offset = torch.zeros_like(inertias)
        offset[:, :, [0, 4, 8]] += 0.01
        new_inertias = inertias + offset
        self._robot.root_physx_view.set_inertias(new_inertias, torch.arange(self.num_envs))

    def _set_default_dynamics_parameters(self):
        """Set parameters defining dynamic interactions."""
        self.default_gains = torch.tensor(self.cfg.ctrl.default_task_prop_gains, device=self.device).repeat(
            (self.num_envs, 1)
        )

        self.pos_threshold = torch.tensor(self.cfg.ctrl.pos_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.rot_threshold = torch.tensor(self.cfg.ctrl.rot_action_threshold, device=self.device).repeat(
            (self.num_envs, 1)
        )

        # Set masses and frictions.
        self._set_friction(self._held_asset, self.cfg_task.held_asset_cfg.friction)
        self._set_friction(self._fixed_asset, self.cfg_task.fixed_asset_cfg.friction)
        self._set_friction(self._robot, self.cfg_task.robot_cfg.friction)

    def _set_friction(self, asset, value):
        """Update material properties for a given asset."""
        materials = asset.root_physx_view.get_material_properties()
        materials[..., 0] = value  # Static friction.
        materials[..., 1] = value  # Dynamic friction.
        env_ids = torch.arange(self.scene.num_envs, device="cpu")
        asset.root_physx_view.set_material_properties(materials, env_ids)

    def _init_tensors(self):
        """Initialize tensors once."""
        self.identity_quat = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        )

        # Control targets.
        self.ctrl_target_joint_pos = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.ctrl_target_fingertip_midpoint_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.ctrl_target_fingertip_midpoint_quat = torch.zeros((self.num_envs, 4), device=self.device)

        # Fixed asset.
        self.fixed_pos_action_frame = torch.zeros((self.num_envs, 3), device=self.device)
        self.fixed_pos_obs_frame = torch.zeros((self.num_envs, 3), device=self.device)
        self.init_fixed_pos_obs_noise = torch.zeros((self.num_envs, 3), device=self.device)

        # Held asset (Forge tasks: geometric tip from factory_utils)
        if self._is_forge_task:
            self.held_base_pos_local = factory_utils.get_held_base_pos_local(
                self.cfg_task.name, self.cfg_task.fixed_asset_cfg, self.num_envs, self.device
            )
        else:
            held_base_x_offset = 0.0
            held_base_z_offset = 0.0
            self.held_base_pos_local = torch.tensor([0.0, 0.0, 0.0], device=self.device).repeat((self.num_envs, 1))
            self.held_base_pos_local[:, 0] = held_base_x_offset
            self.held_base_pos_local[:, 2] = held_base_z_offset
        self.held_base_quat_local = self.identity_quat.clone().detach()

        self.held_base_pos = torch.zeros_like(self.held_base_pos_local)
        self.held_base_quat = self.identity_quat.clone().detach()

        self._has_grasp_json = bool(getattr(self.cfg_task, "plug_grasp_json", ""))
        if self._has_grasp_json:
            self.plug_grasps, self.disassembly_dists = self._load_assembly_info()
        else:
            self.plug_grasps, self.disassembly_dists = self._create_default_assembly_info()
        self.curriculum_height_bound, self.curriculum_height_step = self._get_curriculum_info(self.disassembly_dists)
        if self._has_dtw:
            self._load_disassembly_data()
            self.prev_ee_traj = torch.zeros(
                (self.num_envs, self.cfg_task.num_point_robot_traj, self._dtw_dim),
                device=self.device,
            )
        else:
            self.eef_pos_traj = None
            self.eef_pose_traj = None
            self._dtw_dim = 3
            self._dtw_pose_mode = False
            self.prev_ee_traj = None

        # Load grasp pose from json files given assembly ID
        # Grasp pose tensors
        self.palm_to_finger_center = (
            torch.tensor([0.0, 0.0, -self.cfg_task.palm_to_finger_dist], device=self.device)
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )
        self.robot_to_gripper_quat = (
            torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        )
        self.plug_grasp_pos_local = self.plug_grasps[: self.num_envs, :3]
        self.plug_grasp_quat_local = torch.roll(self.plug_grasps[: self.num_envs, 3:], -1, 1)

        # Computer body indices.
        self.left_finger_body_idx = self._robot.body_names.index("panda_leftfinger")
        self.right_finger_body_idx = self._robot.body_names.index("panda_rightfinger")
        self.fingertip_body_idx = self._robot.body_names.index("panda_fingertip_centered")

        # Tensors for finite-differencing.
        self.last_update_timestamp = 0.0  # Note: This is for finite differencing body velocities.
        self.prev_fingertip_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_fingertip_quat = self.identity_quat.clone()
        self.prev_joint_pos = torch.zeros((self.num_envs, 7), device=self.device)

        self.obs_history_buf = torch.zeros(
            (self.num_envs, self.cfg.obs_window_size, self._single_obs_dim), device=self.device
        )
        self.state_history_buf = torch.zeros(
            (self.num_envs, self.cfg.state_window_size, self._single_state_dim), device=self.device
        )

        # Keypoint tensors.
        self.target_held_base_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_held_base_quat = self.identity_quat.clone().detach()

        if self._is_forge_task:
            self._forge_kp_n = self._forge_keypoint_count()
            self.keypoint_offsets = None
            self.keypoints_held = torch.zeros((self.num_envs, self._forge_kp_n, 3), device=self.device)
            self.keypoints_fixed = torch.zeros_like(self.keypoints_held)
            self._allocate_forge_kp_buffers()
        else:
            offsets = self._get_keypoint_offsets(self.cfg_task.num_keypoints)
            self.keypoint_offsets = offsets * self.cfg_task.keypoint_scale
            self.keypoints_held = torch.zeros((self.num_envs, self.cfg_task.num_keypoints, 3), device=self.device)
            self.keypoints_fixed = torch.zeros_like(self.keypoints_held, device=self.device)

        # Used to compute target poses.
        self.fixed_success_pos_local = torch.zeros((self.num_envs, 3), device=self.device)
        self.fixed_success_pos_local[:, 2] = 0.0

        self.ep_succeeded = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.ep_success_times = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)

        # SBC
        if self.cfg_task.if_sbc:
            self.curr_max_disp = self.curriculum_height_bound[:, 0]
        else:
            self.curr_max_disp = self.curriculum_height_bound[:, 1]

        self.prev_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)

    def _forge_keypoint_count(self) -> int:
        if self.cfg_task.name == "box_lid_insert":
            return 2 + max(self.cfg_task.num_reset_extra_kp, self.cfg_task.num_success_extra_kp)
        if self.cfg_task.name == "rj45_insert":
            return self.cfg_task.num_reset_kp
        if self.cfg_task.name == "bnc_insert":
            return max(self.cfg_task.num_reset_kp, self.cfg_task.num_success_kp)
        return 4

    def _allocate_forge_kp_buffers(self):
        if self.cfg_task.name == "box_lid_insert":
            n = self._forge_kp_n
            self.kp_lid_local = torch.zeros((self.num_envs, n, 3), device=self.device)
            self.kp_box_local = torch.zeros((self.num_envs, n, 3), device=self.device)
            self.kp_box_y_target = torch.zeros((self.num_envs, n), device=self.device)
        elif self.cfg_task.name == "rj45_insert":
            self.kp_rj45_female_z_init = self.cfg_task.female_rear_edge_z_local
            n = self.cfg_task.num_reset_kp
            self.kp_rj45_male_local = torch.zeros((self.num_envs, n, 3), device=self.device)
            self.kp_rj45_female_local = torch.zeros((self.num_envs, n, 3), device=self.device)
        elif self.cfg_task.name == "bnc_insert":
            n = max(self.cfg_task.num_reset_kp, self.cfg_task.num_success_kp)
            self.kp_bnc_local = torch.zeros((self.num_envs, n, 3), device=self.device)
            self.kp_bnc_fixed_local = torch.zeros((self.num_envs, n, 3), device=self.device)

    def _update_forge_keypoints_and_dist(self):
        ident = self.identity_quat
        if self.cfg_task.name == "box_lid_insert":
            n = self.kp_lid_local.shape[1]
            for i in range(n):
                _, self.keypoints_held[:, i] = torch_utils.tf_combine(
                    self.held_quat, self.held_pos, ident, self.kp_lid_local[:, i]
                )
                _, self.keypoints_fixed[:, i] = torch_utils.tf_combine(
                    self.fixed_quat, self.fixed_pos, ident, self.kp_box_local[:, i]
                )
        elif self.cfg_task.name == "rj45_insert":
            n_kp = self.kp_rj45_male_local.shape[1]
            for i in range(n_kp):
                _, self.keypoints_held[:, i] = torch_utils.tf_combine(
                    self.held_quat, self.held_pos, ident, self.kp_rj45_male_local[:, i]
                )
                _, self.keypoints_fixed[:, i] = torch_utils.tf_combine(
                    self.fixed_quat, self.fixed_pos, ident, self.kp_rj45_female_local[:, i]
                )
        elif self.cfg_task.name == "bnc_insert":
            held_base_pos_kp, held_base_quat_kp = factory_utils.get_held_base_pose(
                self.held_pos,
                self.held_quat,
                self.cfg_task.name,
                self.cfg_task.fixed_asset_cfg,
                self.num_envs,
                self.device,
            )
            target_base_pos_kp, target_base_quat_kp = factory_utils.get_target_held_base_pose(
                self.fixed_pos,
                self.fixed_quat,
                self.cfg_task.name,
                self.cfg_task.fixed_asset_cfg,
                self.num_envs,
                self.device,
                task_cfg=self.cfg_task,
            )
            n_kp = self.kp_bnc_local.shape[1]
            for i in range(n_kp):
                _, self.keypoints_held[:, i] = torch_utils.tf_combine(
                    held_base_quat_kp, held_base_pos_kp, ident, self.kp_bnc_local[:, i]
                )
                _, self.keypoints_fixed[:, i] = torch_utils.tf_combine(
                    target_base_quat_kp, target_base_pos_kp, ident, self.kp_bnc_fixed_local[:, i]
                )

        self.keypoint_dist = torch.norm(self.keypoints_held - self.keypoints_fixed, p=2, dim=-1).mean(-1)

        if self.cfg_task.name == "rj45_insert":
            close = self.keypoint_dist < self.cfg_task.kp_advance_threshold
            if close.any():
                self.kp_rj45_female_local[close, :, 2] -= self.cfg_task.kp_advance_step
                self.kp_rj45_female_local[:, :, 2].clamp_(min=self.cfg_task.kp_advance_z_limit)

        if self.cfg_task.name == "box_lid_insert":
            close = self.keypoint_dist < self.cfg_task.kp_advance_threshold
            if close.any():
                step = self.cfg_task.kp_advance_y_step
                cur = self.kp_box_local[:, :, 1]
                tgt = self.kp_box_y_target
                diff = tgt - cur
                delta = torch.sign(diff) * torch.minimum(diff.abs(), torch.full_like(diff, step))
                self.kp_box_local[close, :, 1] = cur[close] + delta[close]

        if self.cfg_task.name == "bnc_insert":
            nr = self.cfg_task.num_reset_kp
            thr = float(getattr(self.cfg_task, "bnc_kp_advance_threshold", 0.005))
            step = float(getattr(self.cfg_task, "bnc_kp_advance_step", 0.0002))
            close = self.keypoint_dist < thr
            if close.any():
                self.kp_bnc_fixed_local[close, :nr, 2] -= step
                self.kp_bnc_fixed_local[:, :nr, 2] = torch.maximum(
                    self.kp_bnc_fixed_local[:, :nr, 2], self.kp_bnc_local[:, :nr, 2]
                )

    def _load_assembly_info(self):
        """Load grasp pose and disassembly distance for plugs in each environment."""

        retrieve_file_path(self.cfg_task.plug_grasp_json, download_dir="./")
        with open(os.path.basename(self.cfg_task.plug_grasp_json)) as f:
            plug_grasp_dict = json.load(f)
        plug_grasps = [plug_grasp_dict[f"asset_{self.cfg_task.assembly_id}"] for i in range(self.num_envs)]

        retrieve_file_path(self.cfg_task.disassembly_dist_json, download_dir="./")
        with open(os.path.basename(self.cfg_task.disassembly_dist_json)) as f:
            disassembly_dist_dict = json.load(f)
        disassembly_dists = [disassembly_dist_dict[f"asset_{self.cfg_task.assembly_id}"] for i in range(self.num_envs)]

        return torch.as_tensor(plug_grasps).to(self.device), torch.as_tensor(disassembly_dists).to(self.device)

    def _create_default_assembly_info(self):
        """Create fallback grasp pose and disassembly distance from task config fields."""
        pos = self.cfg_task.simple_grasp_pos_local
        quat_xyzw = self.cfg_task.simple_grasp_quat_local
        single = pos + quat_xyzw  # [x, y, z, qx, qy, qz, qw]
        plug_grasps = torch.tensor([single] * self.num_envs, dtype=torch.float32, device=self.device)

        dist = getattr(self.cfg_task, "default_disassembly_dist", 0.01)
        disassembly_dists = torch.full((self.num_envs,), dist, dtype=torch.float32, device=self.device)
        return plug_grasps, disassembly_dists

    def _get_curriculum_info(self, disassembly_dists):
        """Calculate the ranges and step sizes for Sampling-based Curriculum (SBC) in each environment."""

        curriculum_height_bound = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)
        curriculum_height_step = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)

        curriculum_height_bound[:, 1] = disassembly_dists + self.cfg_task.curriculum_freespace_range

        curriculum_height_step[:, 0] = curriculum_height_bound[:, 1] / self.cfg_task.num_curriculum_step
        curriculum_height_step[:, 1] = -curriculum_height_step[:, 0] / 2.0

        return curriculum_height_bound, curriculum_height_step

    def _load_disassembly_data(self):
        """Load pre-collected EE trajectories for Soft-DTW imitation.

        **Position-only** JSON: each dict has ``fingertip_centered_pos`` ``[[x,y,z], ...]`` (env frame).
        **Pose** JSON: each dict has ``fingertip_centered_pose`` ``[[x,y,z,qw,qx,qy,qz], ...]`` (env position + world wxyz).
        Pose trajectories are start-normalized via ``preprocess_reference_pose_trajectory``; runtime uses
        goal-relative position and ``inv(gripper_goal_quat) * fingertip_quat`` to match.

        Time order: index 0 ≈ start, last ≈ goal (init→insert). Variable-length demos are padded by repeating the last row.
        """
        resolved_path = retrieve_file_path(self.cfg_task.disassembly_path_json, download_dir="./")
        with open(resolved_path, encoding="utf-8") as f:
            disassembly_traj = json.load(f)

        if not disassembly_traj:
            raise ValueError("disassembly_path_json loaded an empty list; need at least one trajectory.")

        use_pose = "fingertip_centered_pose" in disassembly_traj[0]
        if use_pose:
            self._dtw_pose_mode = True
            self._dtw_dim = 7
            self.eef_pos_traj = None
            shifted: list[np.ndarray] = []
            for i in range(len(disassembly_traj)):
                if "fingertip_centered_pose" not in disassembly_traj[i]:
                    raise KeyError(
                        f"Trajectory {i}: mixed formats — all trajectories must include 'fingertip_centered_pose'."
                    )
                pose_np = np.asarray(disassembly_traj[i]["fingertip_centered_pose"], dtype=np.float32).reshape((-1, 7))
                pose_t = torch.as_tensor(pose_np, device=self.device, dtype=torch.float32)
                shifted.append(automate_algo.preprocess_reference_pose_trajectory(pose_t).cpu().numpy())
        else:
            self._dtw_pose_mode = False
            self._dtw_dim = 3
            self.eef_pose_traj = None
            shifted = []
            for i in range(len(disassembly_traj)):
                if "fingertip_centered_pos" not in disassembly_traj[i]:
                    raise KeyError(
                        f"Trajectory {i} missing 'fingertip_centered_pos'. "
                        "Use Automate format or add 'fingertip_centered_pose' for pose DTW "
                        "(see collect_rj45_trajectories.py --output_automate)."
                    )
                curr_ee_traj = np.asarray(disassembly_traj[i]["fingertip_centered_pos"], dtype=np.float32).reshape((-1, 3))
                curr_ee_goal = curr_ee_traj[0, :].copy()
                shifted.append(curr_ee_traj - curr_ee_goal)

        max_len = max(len(a) for a in shifted)
        padded: list[np.ndarray] = []
        for arr in shifted:
            if arr.shape[0] < max_len:
                pad = np.repeat(arr[-1:, :], max_len - arr.shape[0], axis=0)
                arr = np.concatenate([arr, pad], axis=0)
            padded.append(arr)

        stacked = torch.as_tensor(np.stack(padded, axis=0), dtype=torch.float32, device=self.device)
        if use_pose:
            self.eef_pose_traj = stacked
        else:
            self.eef_pos_traj = stacked

    def _get_keypoint_offsets(self, num_keypoints):
        """Get uniformly-spaced keypoints along a line of unit length, centered at 0."""
        keypoint_offsets = torch.zeros((num_keypoints, 3), device=self.device)
        keypoint_offsets[:, -1] = torch.linspace(0.0, 1.0, num_keypoints, device=self.device) - 0.5

        return keypoint_offsets

    def _apply_robot_articulation_props_on_source_env(self):
        """Match FactoryEnv: patch real articulation root on env_0 before cloning (Mimic Franka)."""
        spawn_cfg = self.cfg.robot.spawn
        if spawn_cfg is None or spawn_cfg.articulation_props is None:
            return

        source_robot_path = self.cfg.robot.prim_path.replace(".*", "0")
        articulation_root_prims = sim_utils.get_all_matching_child_prims(
            source_robot_path,
            predicate=lambda prim: prim.HasAPI(UsdPhysics.ArticulationRootAPI),
            traverse_instance_prims=False,
        )
        if len(articulation_root_prims) != 1:
            return

        articulation_root_path = articulation_root_prims[0].GetPath().pathString
        sim_utils.modify_articulation_root_properties(articulation_root_path, spawn_cfg.articulation_props)

    def _setup_scene(self):
        """Initialize simulation scene."""
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -0.4))

        # spawn a usd file of a table into the scene
        cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
        cfg.func(
            "/World/envs/env_.*/Table", cfg, translation=(0.55, 0.0, 0.0), orientation=(0.70711, 0.0, 0.0, 0.70711)
        )

        self._robot = Articulation(self.cfg.robot)
        if isinstance(self.cfg_task.fixed_asset, RigidObjectCfg):
            self._fixed_asset = RigidObject(self.cfg_task.fixed_asset)
        else:
            self._fixed_asset = Articulation(self.cfg_task.fixed_asset)
        if isinstance(self.cfg_task.held_asset, RigidObjectCfg):
            self._held_asset = RigidObject(self.cfg_task.held_asset)
        else:
            self._held_asset = Articulation(self.cfg_task.held_asset)

        self._apply_robot_articulation_props_on_source_env()

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions()

        self.scene.articulations["robot"] = self._robot
        if isinstance(self._fixed_asset, RigidObject):
            self.scene.rigid_objects["fixed_asset"] = self._fixed_asset
        else:
            self.scene.articulations["fixed_asset"] = self._fixed_asset
        if isinstance(self._held_asset, RigidObject):
            self.scene.rigid_objects["held_asset"] = self._held_asset
        else:
            self.scene.articulations["held_asset"] = self._held_asset

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _compute_intermediate_values(self, dt):
        """Get values computed from raw tensors. This includes adding noise."""
        # TODO: A lot of these can probably only be set once?
        self.fixed_pos = self._fixed_asset.data.root_pos_w - self.scene.env_origins
        self.fixed_quat = self._fixed_asset.data.root_quat_w

        self.held_pos = self._held_asset.data.root_pos_w - self.scene.env_origins
        self.held_quat = self._held_asset.data.root_quat_w

        self.fingertip_midpoint_pos = self._robot.data.body_pos_w[:, self.fingertip_body_idx] - self.scene.env_origins
        self.fingertip_midpoint_quat = self._robot.data.body_quat_w[:, self.fingertip_body_idx]
        self.fingertip_midpoint_linvel = self._robot.data.body_lin_vel_w[:, self.fingertip_body_idx]
        self.fingertip_midpoint_angvel = self._robot.data.body_ang_vel_w[:, self.fingertip_body_idx]

        jacobians = self._robot.root_physx_view.get_jacobians()

        self.left_finger_jacobian = jacobians[:, self.left_finger_body_idx - 1, 0:6, 0:7]
        self.right_finger_jacobian = jacobians[:, self.right_finger_body_idx - 1, 0:6, 0:7]
        self.fingertip_midpoint_jacobian = (self.left_finger_jacobian + self.right_finger_jacobian) * 0.5
        self.arm_mass_matrix = self._robot.root_physx_view.get_generalized_mass_matrices()[:, 0:7, 0:7]
        self.joint_pos = self._robot.data.joint_pos.clone()
        self.joint_vel = self._robot.data.joint_vel.clone()

        # Compute pose of gripper goal and top of socket in socket frame
        self.gripper_goal_quat, self.gripper_goal_pos = torch_utils.tf_combine(
            self.fixed_quat,
            self.fixed_pos,
            self.plug_grasp_quat_local,
            self.plug_grasp_pos_local,
        )

        self.gripper_goal_quat, self.gripper_goal_pos = torch_utils.tf_combine(
            self.gripper_goal_quat,
            self.gripper_goal_pos,
            self.robot_to_gripper_quat,
            self.palm_to_finger_center,
        )

        # Finite-differencing results in more reliable velocity estimates.
        self.ee_linvel_fd = (self.fingertip_midpoint_pos - self.prev_fingertip_pos) / dt
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()

        # Add state differences if velocity isn't being added.
        rot_diff_quat = torch_utils.quat_mul(
            self.fingertip_midpoint_quat, torch_utils.quat_conjugate(self.prev_fingertip_quat)
        )
        rot_diff_quat *= torch.sign(rot_diff_quat[:, 0]).unsqueeze(-1)
        rot_diff_aa = axis_angle_from_quat(rot_diff_quat)
        self.ee_angvel_fd = rot_diff_aa / dt
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()

        joint_diff = self.joint_pos[:, 0:7] - self.prev_joint_pos
        self.joint_vel_fd = joint_diff / dt
        self.prev_joint_pos = self.joint_pos[:, 0:7].clone()

        # Keypoint tensors & held-base geometry (Forge matches factory_utils).
        if self._is_forge_task:
            # get_held_base_pose / get_target_held_base_pose return (pos, quat) — order matches BNC branch below.
            self.held_base_pos[:], self.held_base_quat[:] = factory_utils.get_held_base_pose(
                self.held_pos,
                self.held_quat,
                self.cfg_task.name,
                self.cfg_task.fixed_asset_cfg,
                self.num_envs,
                self.device,
            )
            self.target_held_base_pos[:], self.target_held_base_quat[:] = factory_utils.get_target_held_base_pose(
                self.fixed_pos,
                self.fixed_quat,
                self.cfg_task.name,
                self.cfg_task.fixed_asset_cfg,
                self.num_envs,
                self.device,
                task_cfg=self.cfg_task,
            )
            self._update_forge_keypoints_and_dist()
        else:
            self.held_base_quat[:], self.held_base_pos[:] = torch_utils.tf_combine(
                self.held_quat, self.held_pos, self.held_base_quat_local, self.held_base_pos_local
            )
            self.target_held_base_quat[:], self.target_held_base_pos[:] = torch_utils.tf_combine(
                self.fixed_quat, self.fixed_pos, self.identity_quat, self.fixed_success_pos_local
            )

            # Compute pos of keypoints on held asset, and fixed asset in world frame
            for idx, keypoint_offset in enumerate(self.keypoint_offsets):
                self.keypoints_held[:, idx] = torch_utils.tf_combine(
                    self.held_base_quat, self.held_base_pos, self.identity_quat, keypoint_offset.repeat(self.num_envs, 1)
                )[1]
                self.keypoints_fixed[:, idx] = torch_utils.tf_combine(
                    self.target_held_base_quat,
                    self.target_held_base_pos,
                    self.identity_quat,
                    keypoint_offset.repeat(self.num_envs, 1),
                )[1]

            self.keypoint_dist = torch.norm(self.keypoints_held - self.keypoints_fixed, p=2, dim=-1).mean(-1)
        self.last_update_timestamp = self._robot._data._sim_timestamp

    def _update_obs_state_history(self, obs_tensors: torch.Tensor, state_tensors: torch.Tensor):
        """Shift history left and append latest step; flatten to (num_envs, window * dim)."""
        if self.cfg.obs_window_size > 1:
            self.obs_history_buf[:, :-1] = self.obs_history_buf[:, 1:].clone()
        self.obs_history_buf[:, -1] = obs_tensors

        if self.cfg.state_window_size > 1:
            self.state_history_buf[:, :-1] = self.state_history_buf[:, 1:].clone()
        self.state_history_buf[:, -1] = state_tensors

        obs_out = self.obs_history_buf.view(self.num_envs, -1)
        state_out = self.state_history_buf.view(self.num_envs, -1)
        return obs_out, state_out

    def _get_observations(self):
        """Get actor/critic inputs using asymmetric critic."""

        prev_actions = self.actions.clone()

        obs_dict = {
            "joint_pos": self.joint_pos[:, 0:7],
            "fingertip_pos": self.fingertip_midpoint_pos,
            "fingertip_quat": self.fingertip_midpoint_quat,
            "fingertip_goal_pos": self.gripper_goal_pos,
            "fingertip_goal_quat": self.gripper_goal_quat,
            "delta_pos": self.gripper_goal_pos - self.fingertip_midpoint_pos,
            "prev_actions": prev_actions,
        }

        state_dict = {
            "joint_pos": self.joint_pos[:, 0:7],
            "joint_vel": self.joint_vel[:, 0:7],
            "fingertip_pos": self.fingertip_midpoint_pos,
            "fingertip_quat": self.fingertip_midpoint_quat,
            "ee_linvel": self.fingertip_midpoint_linvel,
            "ee_angvel": self.fingertip_midpoint_angvel,
            "fingertip_goal_pos": self.gripper_goal_pos,
            "fingertip_goal_quat": self.gripper_goal_quat,
            "held_pos": self.held_pos,
            "held_quat": self.held_quat,
            "delta_pos": self.gripper_goal_pos - self.fingertip_midpoint_pos,
            "prev_actions": prev_actions,
        }

        obs_tensors = torch.cat([obs_dict[k] for k in self.cfg.obs_order + ["prev_actions"]], dim=-1)
        state_tensors = torch.cat([state_dict[k] for k in self.cfg.state_order + ["prev_actions"]], dim=-1)
        obs_out, state_out = self._update_obs_state_history(obs_tensors, state_tensors)
        return {"policy": obs_out, "critic": state_out}

    def _reset_buffers(self, env_ids):
        """Reset buffers."""
        self.ep_succeeded[env_ids] = 0

    def _pre_physics_step(self, action):
        """Apply policy actions with smoothing."""
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self._reset_buffers(env_ids)

        self.actions = (
            self.cfg.ctrl.ema_factor * action.clone().to(self.device) + (1 - self.cfg.ctrl.ema_factor) * self.actions
        )

    def move_gripper_in_place(self, ctrl_target_gripper_dof_pos):
        """Keep gripper in current position as gripper closes."""
        actions = torch.zeros((self.num_envs, 6), device=self.device)
        ctrl_target_gripper_dof_pos = 0.0

        # Interpret actions as target pos displacements and set pos target
        pos_actions = actions[:, 0:3] * self.pos_threshold
        self.ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = actions[:, 3:6]

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)

        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)

        rot_actions_quat = torch.where(
            angle.unsqueeze(-1).repeat(1, 4) > 1.0e-6,
            rot_actions_quat,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1),
        )
        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)

        target_euler_xyz = torch.stack(torch_utils.get_euler_xyz(self.ctrl_target_fingertip_midpoint_quat), dim=1)
        target_euler_xyz[:, 0] = 3.14159
        target_euler_xyz[:, 1] = 0.0

        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            roll=target_euler_xyz[:, 0], pitch=target_euler_xyz[:, 1], yaw=target_euler_xyz[:, 2]
        )

        self.ctrl_target_gripper_dof_pos = ctrl_target_gripper_dof_pos
        self.generate_ctrl_signals()

    def _apply_action(self):
        """Apply actions for policy as delta targets from current position."""
        # Get current yaw for success checking.
        _, _, curr_yaw = torch_utils.get_euler_xyz(self.fingertip_midpoint_quat)
        self.curr_yaw = torch.where(curr_yaw > np.deg2rad(235), curr_yaw - 2 * np.pi, curr_yaw)

        # Note: We use finite-differenced velocities for control and observations.
        # Check if we need to re-compute velocities within the decimation loop.
        if self.last_update_timestamp < self._robot._data._sim_timestamp:
            self._compute_intermediate_values(dt=self.physics_dt)

        # Interpret actions as target pos displacements and set pos target
        pos_actions = self.actions[:, 0:3] * self.pos_threshold

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = self.actions[:, 3:6]
        if self.cfg_task.unidirectional_rot:
            rot_actions[:, 2] = -(rot_actions[:, 2] + 1.0) * 0.5  # [-1, 0]
        rot_actions = rot_actions * self.rot_threshold

        self.ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions
        # To speed up learning, never allow the policy to move more than 5cm away from the base.
        delta_pos = self.ctrl_target_fingertip_midpoint_pos - self.fixed_pos_action_frame
        pos_error_clipped = torch.clip(
            delta_pos, -self.cfg.ctrl.pos_action_bounds[0], self.cfg.ctrl.pos_action_bounds[1]
        )
        self.ctrl_target_fingertip_midpoint_pos = self.fixed_pos_action_frame + pos_error_clipped

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)

        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        rot_actions_quat = torch.where(
            angle.unsqueeze(-1).repeat(1, 4) > 1e-6,
            rot_actions_quat,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1),
        )
        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)

        target_euler_xyz = torch.stack(torch_utils.get_euler_xyz(self.ctrl_target_fingertip_midpoint_quat), dim=1)
        target_euler_xyz[:, 0] = 3.14159  # Restrict actions to be upright.
        target_euler_xyz[:, 1] = 0.0

        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            roll=target_euler_xyz[:, 0], pitch=target_euler_xyz[:, 1], yaw=target_euler_xyz[:, 2]
        )

        self.ctrl_target_gripper_dof_pos = 0.0
        self.generate_ctrl_signals()

    def _set_gains(self, prop_gains, rot_deriv_scale=1.0):
        """Set robot gains using critical damping."""
        self.task_prop_gains = prop_gains
        self.task_deriv_gains = 2 * torch.sqrt(prop_gains)
        self.task_deriv_gains[:, 3:6] /= rot_deriv_scale

    def generate_ctrl_signals(self):
        """Get Jacobian. Set Franka DOF position targets (fingers) or DOF torques (arm)."""
        self.joint_torque, self.applied_wrench = fc.compute_dof_torque(
            cfg=self.cfg,
            dof_pos=self.joint_pos,
            dof_vel=self.joint_vel,  # _fd,
            fingertip_midpoint_pos=self.fingertip_midpoint_pos,
            fingertip_midpoint_quat=self.fingertip_midpoint_quat,
            fingertip_midpoint_linvel=self.ee_linvel_fd,
            fingertip_midpoint_angvel=self.ee_angvel_fd,
            jacobian=self.fingertip_midpoint_jacobian,
            arm_mass_matrix=self.arm_mass_matrix,
            ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat,
            task_prop_gains=self.task_prop_gains,
            task_deriv_gains=self.task_deriv_gains,
            device=self.device,
        )

        # set target for gripper joints to use GYM's PD controller
        self.ctrl_target_joint_pos[:, 7:9] = self.ctrl_target_gripper_dof_pos
        self.joint_torque[:, 7:9] = 0.0

        self._robot.set_joint_position_target(self.ctrl_target_joint_pos)
        self._robot.set_joint_effort_target(self.joint_torque)

    def _get_dones(self):
        """Update intermediate values used for rewards and observations."""
        self._compute_intermediate_values(dt=self.physics_dt)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self._is_forge_task and bool(getattr(self.cfg_task, "terminate_on_success", False)):
            st = float(self.cfg_task.success_threshold)
            success_now = get_curr_successes_forge(
                self.cfg_task,
                self.held_pos,
                self.held_quat,
                self.fixed_pos,
                self.fixed_quat,
                self.held_base_pos,
                self.ep_succeeded,
                self.num_envs,
                self.device,
                st,
            )
            if self.cfg_task.name == "bnc_insert" and not bool(
                getattr(self.cfg_task, "bnc_apply_success_criteria", True)
            ):
                success_now = torch.zeros(
                    (self.num_envs,), dtype=torch.bool, device=self.device
                )
            return success_now, time_out & ~success_now
        return time_out, time_out

    def _get_curr_successes(self, success_threshold, check_rot=False):
        """Per-env geometric success (same logic as reward). Used by ``rl_games/play.py`` and viz scripts."""
        del check_rot  # Automate / Forge bridge does not use Factory nut_thread rotation branch.
        if self._is_forge_task:
            # Same threshold semantics as factory_env.FactoryEnv._get_curr_successes (Forge).
            st = float(success_threshold)
            return get_curr_successes_forge(
                self.cfg_task,
                self.held_pos,
                self.held_quat,
                self.fixed_pos,
                self.fixed_quat,
                self.held_base_pos,
                self.ep_succeeded,
                self.num_envs,
                self.device,
                st,
            )
        return automate_algo.check_plug_inserted_in_socket(
            self.held_pos,
            self.fixed_pos,
            self.disassembly_dists,
            self.keypoints_held,
            self.keypoints_fixed,
            self.cfg_task.close_error_thresh,
            self.episode_length_buf,
        )

    def _get_rewards(self):
        """Update rewards and compute success statistics."""
        if self._is_forge_task:
            # Match Factory/Forge: for rj45/bnc, success_threshold > 1.0 → ENGAGE branch; else → SUCCESS (depth).
            st = float(self.cfg_task.success_threshold)
            curr_successes = get_curr_successes_forge(
                self.cfg_task,
                self.held_pos,
                self.held_quat,
                self.fixed_pos,
                self.fixed_quat,
                self.held_base_pos,
                self.ep_succeeded,
                self.num_envs,
                self.device,
                st,
            )
            if self.cfg_task.name == "bnc_insert" and not bool(
                getattr(self.cfg_task, "bnc_apply_success_criteria", True)
            ):
                curr_successes = torch.zeros(
                    (self.num_envs,), dtype=torch.bool, device=self.device
                )
            first_success = torch.logical_and(curr_successes, self.ep_succeeded == 0)
            first_success_ids = first_success.nonzero(as_tuple=False).squeeze(-1)
            if self.cfg_task.name == "bnc_insert" and curr_successes.any():
                self.kp_bnc_fixed_local[curr_successes] = self.kp_bnc_local[curr_successes].clone()
            if self.cfg_task.name == "box_lid_insert" and len(first_success_ids) > 0:
                ns = self.cfg_task.num_success_extra_kp
                kp = torch.rand((len(first_success_ids), ns, 3), device=self.device)
                kp[:, :, 0] = kp[:, :, 0] * 0.1046 - 0.0523
                kp[:, :, 1] = kp[:, :, 1] * 0.0838 - 0.0444
                kp[:, :, 2] = kp[:, :, 2] * 0.0112 + 0.0188
                self.kp_lid_local[first_success_ids, 2 : 2 + ns] = kp
                self.kp_box_local[first_success_ids, 2 : 2 + ns] = kp
                self.kp_box_y_target[first_success_ids, 2 : 2 + ns] = kp[:, :, 1].clone()
                y0 = self.cfg_task.kp_advance_y_start
                self.kp_box_local[first_success_ids, 2 : 2 + ns, 1] = y0
        else:
            curr_successes = automate_algo.check_plug_inserted_in_socket(
                self.held_pos,
                self.fixed_pos,
                self.disassembly_dists,
                self.keypoints_held,
                self.keypoints_fixed,
                self.cfg_task.close_error_thresh,
                self.episode_length_buf,
            )

        rew_buf = self._update_rew_buf(curr_successes)
        self.ep_succeeded = torch.maximum(self.ep_succeeded, curr_successes.long())

        # Only log episode success rates at the end of an episode.
        if torch.any(self.reset_buf):
            self.extras["successes"] = torch.count_nonzero(self.ep_succeeded) / self.num_envs

            if self.cfg_task.if_sbc:
                sbc_rwd_scale = automate_algo.get_curriculum_reward_scale(
                    curr_max_disp=self.curr_max_disp,
                    curriculum_height_bound=self.curriculum_height_bound,
                )

                rew_buf *= sbc_rwd_scale

            if self.cfg_task.if_sbc:
                self.curr_max_disp = automate_algo.get_new_max_disp(
                    curr_success=torch.count_nonzero(self.ep_succeeded) / self.num_envs,
                    cfg_task=self.cfg_task,
                    curriculum_height_bound=self.curriculum_height_bound,
                    curriculum_height_step=self.curriculum_height_step,
                    curr_max_disp=self.curr_max_disp,
                )

            self.extras["curr_max_disp"] = self.curr_max_disp

            if self.cfg_task.if_logging_eval:
                self.success_log = torch.cat([self.success_log, self.ep_succeeded.reshape((self.num_envs, 1))], dim=0)

                if self.success_log.shape[0] >= self.cfg_task.num_eval_trials:
                    automate_log.write_log_to_hdf5(
                        self.held_asset_pose_log,
                        self.fixed_asset_pose_log,
                        self.success_log,
                        self.cfg_task.eval_filename,
                    )
                    exit(0)

        self.prev_actions = self.actions.clone()
        return rew_buf

    def _update_rew_buf(self, curr_successes):
        """Compute reward at current timestep (native AutoMate: SDF or keypoint exp + optional DTW + success)."""
        rew_dict = dict({})

        if self._has_sdf:
            rew_dict["sdf"] = industreal_algo.get_sdf_reward(
                self.plug_mesh,
                self.plug_sample_points,
                self.held_pos,
                self.held_quat,
                self.fixed_pos,
                self.fixed_quat,
                self.wp_device,
                self.device,
            )
        else:
            rew_dict["sdf"] = torch.exp(-10.0 * self.keypoint_dist)

        rew_dict["curr_successes"] = curr_successes.clone().float()

        curr_eef_pos = (self.fingertip_midpoint_pos - self.gripper_goal_pos).reshape(-1, 3)

        if self._has_dtw:
            if self._dtw_pose_mode:
                curr_quat_rel = torch_utils.quat_mul(
                    torch_utils.quat_conjugate(self.gripper_goal_quat),
                    self.fingertip_midpoint_quat,
                )
                curr_pose = torch.cat([curr_eef_pos, curr_quat_rel], dim=-1)
                rew_dict["imitation"] = automate_algo.get_imitation_reward_from_dtw_pose_traj(
                    self.eef_pose_traj,
                    curr_pose,
                    self.prev_ee_traj,
                    self.soft_dtw_criterion,
                    self.device,
                    pos_w=getattr(self.cfg_task, "imitation_pose_pos_w", 1.0),
                    rot_w=getattr(self.cfg_task, "imitation_pose_rot_w", 0.35),
                )
                self.prev_ee_traj = torch.cat(
                    (self.prev_ee_traj[:, 1:, :], curr_pose.unsqueeze(1).clone().detach()), dim=1
                )
            else:
                rew_dict["imitation"] = automate_algo.get_imitation_reward_from_dtw(
                    self.eef_pos_traj, curr_eef_pos, self.prev_ee_traj, self.soft_dtw_criterion, self.device
                )
                self.prev_ee_traj = torch.cat(
                    (self.prev_ee_traj[:, 1:, :], curr_eef_pos.unsqueeze(1).clone().detach()), dim=1
                )
        else:
            rew_dict["imitation"] = torch.zeros((self.num_envs,), device=self.device)

        rew_buf = (
            self.cfg_task.sdf_rwd_scale * rew_dict["sdf"]
            + self.cfg_task.imitation_rwd_scale * rew_dict["imitation"]
            + rew_dict["curr_successes"]
        )

        for rew_name, rew in rew_dict.items():
            self.extras[f"logs_rew_{rew_name}"] = rew.mean()

        return rew_buf

    def _reset_idx(self, env_ids):
        """
        We assume all envs will always be reset at the same time.
        """
        super()._reset_idx(env_ids)

        self.obs_history_buf[env_ids] = 0.0
        self.state_history_buf[env_ids] = 0.0

        self._set_assets_to_default_pose(env_ids)
        self._set_franka_to_default_pose(joints=self.cfg.ctrl.reset_joints, env_ids=env_ids)
        self.step_sim_no_action()

        self.randomize_initial_state(env_ids)

        if self._is_forge_task:
            reset_forge_keypoints_after_randomize(self, env_ids)

        if self.cfg_task.if_logging_eval:
            self.held_asset_pose_log = torch.cat(
                [self.held_asset_pose_log, torch.cat([self.held_pos, self.held_quat], dim=1)], dim=0
            )
            self.fixed_asset_pose_log = torch.cat(
                [self.fixed_asset_pose_log, torch.cat([self.fixed_pos, self.fixed_quat], dim=1)], dim=0
            )

        if self._has_dtw:
            pos_rel = (self.fingertip_midpoint_pos - self.gripper_goal_pos).unsqueeze(1)
            if self._dtw_pose_mode:
                q_rel = torch_utils.quat_mul(
                    torch_utils.quat_conjugate(self.gripper_goal_quat),
                    self.fingertip_midpoint_quat,
                )
                prev_slice = torch.cat([pos_rel, q_rel.unsqueeze(1)], dim=-1)
            else:
                prev_slice = pos_rel
            self.prev_ee_traj = torch.repeat_interleave(
                prev_slice, self.cfg_task.num_point_robot_traj, dim=1
            )

    def _set_assets_to_default_pose(self, env_ids):
        """Move assets to default pose before randomization."""
        held_state = self._held_asset.data.default_root_state.clone()[env_ids]
        held_state[:, 0:3] += self.scene.env_origins[env_ids]
        held_state[:, 7:] = 0.0
        self._held_asset.write_root_pose_to_sim(held_state[:, 0:7], env_ids=env_ids)
        self._held_asset.write_root_velocity_to_sim(held_state[:, 7:], env_ids=env_ids)
        self._held_asset.reset()

        fixed_state = self._fixed_asset.data.default_root_state.clone()[env_ids]
        fixed_state[:, 0:3] += self.scene.env_origins[env_ids]
        fixed_state[:, 7:] = 0.0
        self._fixed_asset.write_root_pose_to_sim(fixed_state[:, 0:7], env_ids=env_ids)
        self._fixed_asset.write_root_velocity_to_sim(fixed_state[:, 7:], env_ids=env_ids)
        self._fixed_asset.reset()

    def _move_gripper_to_grasp_pose(self, env_ids):
        """Define grasp pose for plug and move gripper to pose."""

        gripper_goal_quat, gripper_goal_pos = torch_utils.tf_combine(
            self.held_quat,
            self.held_pos,
            self.plug_grasp_quat_local,
            self.plug_grasp_pos_local,
        )

        gripper_goal_quat, gripper_goal_pos = torch_utils.tf_combine(
            gripper_goal_quat,
            gripper_goal_pos,
            self.robot_to_gripper_quat,
            self.palm_to_finger_center,
        )

        # Set target_pos
        self.ctrl_target_fingertip_midpoint_pos = gripper_goal_pos.clone()

        # Set target rot
        self.ctrl_target_fingertip_midpoint_quat = gripper_goal_quat.clone()

        self.set_pos_inverse_kinematics(env_ids)
        self.step_sim_no_action()

    def set_pos_inverse_kinematics(self, env_ids):
        """Set robot joint position using DLS IK."""
        ik_time = 0.0
        while ik_time < 0.50:
            # Compute error to target.
            pos_error, axis_angle_error = fc.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos[env_ids],
                fingertip_midpoint_quat=self.fingertip_midpoint_quat[env_ids],
                ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos[env_ids],
                ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat[env_ids],
                jacobian_type="geometric",
                rot_error_type="axis_angle",
            )

            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)

            # Solve DLS problem.
            delta_dof_pos = fc._get_delta_dof_pos(
                delta_pose=delta_hand_pose,
                ik_method="dls",
                jacobian=self.fingertip_midpoint_jacobian[env_ids],
                device=self.device,
            )
            self.joint_pos[env_ids, 0:7] += delta_dof_pos[:, 0:7]
            self.joint_vel[env_ids, :] = torch.zeros_like(self.joint_pos[env_ids,])

            self.ctrl_target_joint_pos[env_ids, 0:7] = self.joint_pos[env_ids, 0:7]
            # Update dof state.
            self._robot.write_joint_state_to_sim(self.joint_pos, self.joint_vel)
            self._robot.reset()
            self._robot.set_joint_position_target(self.ctrl_target_joint_pos)

            # Simulate and update tensors.
            self.step_sim_no_action()
            ik_time += self.physics_dt

        return pos_error, axis_angle_error

    def set_pos_inverse_kinematics_to_targets(
        self,
        ctrl_target_fingertip_midpoint_pos: torch.Tensor,
        ctrl_target_fingertip_midpoint_quat: torch.Tensor,
        env_ids: torch.Tensor,
    ):
        """DLS IK toward explicit fingertip targets (Forge reset path; matches Factory)."""
        ik_time = 0.0
        while ik_time < 0.25:
            pos_error, axis_angle_error = fc.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos[env_ids],
                fingertip_midpoint_quat=self.fingertip_midpoint_quat[env_ids],
                ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos[env_ids],
                ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat[env_ids],
                jacobian_type="geometric",
                rot_error_type="axis_angle",
            )

            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)

            delta_dof_pos = fc._get_delta_dof_pos(
                delta_pose=delta_hand_pose,
                ik_method="dls",
                jacobian=self.fingertip_midpoint_jacobian[env_ids],
                device=self.device,
            )
            self.joint_pos[env_ids, 0:7] += delta_dof_pos[:, 0:7]
            self.joint_vel[env_ids, :] = torch.zeros_like(self.joint_pos[env_ids,])

            self.ctrl_target_joint_pos[env_ids, 0:7] = self.joint_pos[env_ids, 0:7]
            self._robot.write_joint_state_to_sim(self.joint_pos, self.joint_vel)
            self._robot.reset()
            self._robot.set_joint_position_target(self.ctrl_target_joint_pos)

            self.step_sim_no_action()
            ik_time += self.physics_dt

        return pos_error, axis_angle_error

    def get_handheld_asset_relative_pose(self):
        """Default held asset pose in fingertip frame (Forge box_lid / RJ45 / BNC)."""
        cfg = self.cfg_task
        rc = cfg.robot_cfg
        if cfg.name == "box_lid_insert":
            held_asset_relative_pos = torch.zeros((self.num_envs, 3), device=self.device)
            held_asset_relative_pos[:, 2] = cfg.held_asset_cfg.height - rc.franka_fingerpad_length
            pos_offset = torch.tensor(cfg.held_asset_pos_offset, device=self.device)
            held_asset_relative_pos += pos_offset.unsqueeze(0)
        elif cfg.name == "rj45_insert":
            held_asset_relative_pos = torch.zeros((self.num_envs, 3), device=self.device)
            held_asset_relative_pos[:, 2] = cfg.held_asset_cfg.height - rc.franka_fingerpad_length
            pos_offset = torch.tensor(cfg.held_asset_pos_offset, device=self.device)
            held_asset_relative_pos += pos_offset.unsqueeze(0)
        elif cfg.name == "bnc_insert":
            held_asset_relative_pos = torch.zeros((self.num_envs, 3), device=self.device)
            held_asset_relative_pos[:, 2] = cfg.held_asset_cfg.height - rc.franka_fingerpad_length
            pos_offset = torch.tensor(cfg.held_asset_pos_offset, device=self.device)
            held_asset_relative_pos += pos_offset.unsqueeze(0)
        else:
            raise NotImplementedError(f"get_handheld_asset_relative_pose not implemented for {cfg.name}")

        held_asset_relative_quat = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        )
        if cfg.name in ("box_lid_insert", "rj45_insert"):
            initial_rot_deg = cfg.held_asset_rot_init
            rot_offset = getattr(cfg, "held_asset_rot_offset", [0.0, 0.0, 0.0])
            rot_euler = torch.tensor(
                [
                    rot_offset[0] * np.pi / 180.0,
                    rot_offset[1] * np.pi / 180.0,
                    (initial_rot_deg + rot_offset[2]) * np.pi / 180.0,
                ],
                device=self.device,
            ).repeat(self.num_envs, 1)
            held_asset_relative_quat = torch_utils.quat_from_euler_xyz(
                roll=rot_euler[:, 0], pitch=rot_euler[:, 1], yaw=rot_euler[:, 2]
            )

        return held_asset_relative_pos, held_asset_relative_quat

    def _set_franka_to_default_pose(self, joints, env_ids):
        """Return Franka to its default joint position."""
        gripper_width = self.gripper_open_width
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_pos[:, 7:] = gripper_width  # MIMIC
        joint_pos[:, :7] = torch.tensor(joints, device=self.device)[None, :]
        joint_vel = torch.zeros_like(joint_pos)
        joint_effort = torch.zeros_like(joint_pos)
        self.ctrl_target_joint_pos[env_ids, :] = joint_pos
        self._robot.set_joint_position_target(self.ctrl_target_joint_pos[env_ids], env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._robot.reset()
        self._robot.set_joint_effort_target(joint_effort, env_ids=env_ids)

        self.step_sim_no_action()

    def step_sim_no_action(self):
        """Step the simulation without an action. Used for resets."""
        self.scene.write_data_to_sim()
        self.sim.step(render=True)
        self.scene.update(dt=self.physics_dt)
        self._compute_intermediate_values(dt=self.physics_dt)

    def randomize_fixed_initial_state(self, env_ids):
        # (1.) Randomize fixed asset pose.
        fixed_state = self._fixed_asset.data.default_root_state.clone()[env_ids]
        # (1.a.) Position
        rand_sample = torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_pos_init_rand = 2 * (rand_sample - 0.5)  # [-1, 1]
        fixed_asset_init_pos_rand = torch.tensor(
            self.cfg_task.fixed_asset_init_pos_noise, dtype=torch.float32, device=self.device
        )
        fixed_pos_init_rand = fixed_pos_init_rand @ torch.diag(fixed_asset_init_pos_rand)
        fixed_state[:, 0:3] += fixed_pos_init_rand + self.scene.env_origins[env_ids]
        fixed_state[:, 2] += getattr(self.cfg_task, "fixed_asset_z_offset", 0.0)

        # (1.b.) Orientation
        fixed_orn_init_yaw = np.deg2rad(self.cfg_task.fixed_asset_init_orn_deg)
        fixed_orn_yaw_range = np.deg2rad(self.cfg_task.fixed_asset_init_orn_range_deg)
        rand_sample = torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_orn_euler = fixed_orn_init_yaw + fixed_orn_yaw_range * rand_sample
        fixed_orn_euler[:, 0:2] = 0.0  # Only change yaw.
        fixed_orn_quat = torch_utils.quat_from_euler_xyz(
            fixed_orn_euler[:, 0], fixed_orn_euler[:, 1], fixed_orn_euler[:, 2]
        )
        fixed_state[:, 3:7] = fixed_orn_quat
        # (1.c.) Velocity
        fixed_state[:, 7:] = 0.0  # vel
        # (1.d.) Update values.
        self._fixed_asset.write_root_state_to_sim(fixed_state, env_ids=env_ids)
        self._fixed_asset.reset()

        # (1.e.) Noisy position observation.
        fixed_asset_pos_noise = torch.randn((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_asset_pos_rand = torch.tensor(self.cfg.obs_rand.fixed_asset_pos, dtype=torch.float32, device=self.device)
        fixed_asset_pos_noise = fixed_asset_pos_noise @ torch.diag(fixed_asset_pos_rand)
        self.init_fixed_pos_obs_noise[:] = fixed_asset_pos_noise

        self.step_sim_no_action()

    def randomize_held_initial_state(self, env_ids, pre_grasp):
        curr_curriculum_disp_range = self.curriculum_height_bound[:, 1] - self.curr_max_disp
        if pre_grasp:
            self.curriculum_disp = self.curr_max_disp + curr_curriculum_disp_range * (
                torch.rand((self.num_envs,), dtype=torch.float32, device=self.device)
            )

            rand_sample = torch.rand((len(env_ids), 3), dtype=torch.float32, device=self.device)
            held_pos_init_rand = 2 * (rand_sample - 0.5)  # [-1, 1]
            held_asset_init_pos_rand = torch.tensor(
                self.cfg_task.held_asset_init_pos_noise, dtype=torch.float32, device=self.device
            )
            self.held_pos_init_rand = held_pos_init_rand @ torch.diag(held_asset_init_pos_rand)

        # Set plug pos to assembled state, but offset plug Z-coordinate by height of socket,
        # minus curriculum displacement
        held_state = self._held_asset.data.default_root_state.clone()
        held_state[env_ids, 0:3] = self.fixed_pos[env_ids].clone() + self.scene.env_origins[env_ids]
        held_state[env_ids, 3:7] = self.fixed_quat[env_ids].clone()
        held_state[env_ids, 7:] = 0.0

        held_state[env_ids, 2] += self.curriculum_disp

        plug_in_freespace_idx = torch.argwhere(self.curriculum_disp > self.disassembly_dists)
        held_state[plug_in_freespace_idx, :2] += self.held_pos_init_rand[plug_in_freespace_idx, :2]

        self._held_asset.write_root_state_to_sim(held_state)
        self._held_asset.reset()

        self.step_sim_no_action()

    def randomize_initial_state(self, env_ids):
        """Randomize initial state and perform any episode-level randomization."""
        if self._is_forge_task:
            return randomize_initial_state_forge(self, env_ids)

        # Disable gravity.
        physics_sim_view = sim_utils.SimulationContext.instance().physics_sim_view
        physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, 0.0))

        self.randomize_fixed_initial_state(env_ids)

        # Compute the frame on the bolt that would be used as observation: fixed_pos_obs_frame
        # For example, the tip of the bolt can be used as the observation frame
        fixed_tip_pos_local = torch.zeros_like(self.fixed_pos)
        fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.height
        fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.base_height

        _, fixed_tip_pos = torch_utils.tf_combine(
            self.fixed_quat, self.fixed_pos, self.identity_quat, fixed_tip_pos_local
        )
        self.fixed_pos_obs_frame[:] = fixed_tip_pos

        self.randomize_held_initial_state(env_ids, pre_grasp=True)

        self._move_gripper_to_grasp_pose(env_ids)

        self.randomize_held_initial_state(env_ids, pre_grasp=False)
        
        # DEBUG: pause before closing gripper so you can inspect object placement.
        # print("Debug observe...")
        # _DEBUG_OBSERVE_S = 20.0
        # _t = 0.0
        # while _t < _DEBUG_OBSERVE_S:
        #     self.scene.write_data_to_sim()
        #     self.sim.step(render=True)  # render=True prevents Fabric clone failure
        #     self.scene.update(dt=self.physics_dt)
        #     self._compute_intermediate_values(dt=self.physics_dt)
        #     _t += self.sim.get_physics_dt()
        # print("Done observing, closing gripper...")

        # Close hand
        # Set gains to use for quick resets.
        reset_task_prop_gains = torch.tensor(self.cfg.ctrl.reset_task_prop_gains, device=self.device).repeat(
            (self.num_envs, 1)
        )
        reset_rot_deriv_scale = self.cfg.ctrl.reset_rot_deriv_scale
        self._set_gains(reset_task_prop_gains, reset_rot_deriv_scale)

        self.step_sim_no_action()

        grasp_time = 0.0
        while grasp_time < 0.25:
            self.ctrl_target_joint_pos[env_ids, 7:] = 0.0  # Close gripper.
            self.ctrl_target_gripper_dof_pos = 0.0
            self.move_gripper_in_place(ctrl_target_gripper_dof_pos=0.0)
            self.step_sim_no_action()
            grasp_time += self.sim.get_physics_dt()


        self.prev_joint_pos = self.joint_pos[:, 0:7].clone()
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()

        # Set initial actions to involve no-movement. Needed for EMA/correct penalties.
        self.actions = torch.zeros_like(self.actions)
        self.prev_actions = torch.zeros_like(self.actions)
        self.fixed_pos_action_frame[:] = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise

        # Zero initial velocity.
        self.ee_angvel_fd[:, :] = 0.0
        self.ee_linvel_fd[:, :] = 0.0

        # Set initial gains for the episode.
        self._set_gains(self.default_gains)

        physics_sim_view.set_gravity(carb.Float3(*self.cfg.sim.gravity))
