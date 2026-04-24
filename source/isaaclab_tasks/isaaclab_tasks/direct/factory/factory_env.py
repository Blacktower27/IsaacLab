# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch

import carb
import isaacsim.core.utils.torch as torch_utils
from pxr import UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_apply_inverse

from . import factory_control, factory_utils
from .factory_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG, FactoryEnvCfg


class FactoryEnv(DirectRLEnv):
    cfg: FactoryEnvCfg

    def __init__(self, cfg: FactoryEnvCfg, render_mode: str | None = None, **kwargs):
        # Update number of obs/states (single-timestep dims, then multiply by window size).
        cfg.observation_space = sum([OBS_DIM_CFG[obs] for obs in cfg.obs_order])
        cfg.state_space = sum([STATE_DIM_CFG[state] for state in cfg.state_order])
        cfg.observation_space += cfg.action_space
        cfg.state_space += cfg.action_space
        cfg.observation_space *= cfg.obs_window_size
        cfg.state_space *= cfg.state_window_size
        self.cfg_task = cfg.task

        super().__init__(cfg, render_mode, **kwargs)

        factory_utils.set_body_inertias(self._robot, self.scene.num_envs)
        self._init_tensors()
        self._set_default_dynamics_parameters()

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
        if hasattr(self, "_held_asset"):
            factory_utils.set_friction(self._held_asset, self.cfg_task.held_asset_cfg.friction, self.scene.num_envs)
        factory_utils.set_friction(self._fixed_asset, self.cfg_task.fixed_asset_cfg.friction, self.scene.num_envs)
        factory_utils.set_friction(self._robot, self.cfg_task.robot_cfg.friction, self.scene.num_envs)

    def _init_tensors(self):
        """Initialize tensors once."""
        # Control targets.
        self.ctrl_target_joint_pos = torch.zeros((self.num_envs, self._robot.num_joints), device=self.device)
        self.ema_factor = self.cfg.ctrl.ema_factor
        self.dead_zone_thresholds = None

        # Fixed asset.
        self.fixed_pos_obs_frame = torch.zeros((self.num_envs, 3), device=self.device)
        self.init_fixed_pos_obs_noise = torch.zeros((self.num_envs, 3), device=self.device)

        # Computer body indices.
        self.left_finger_body_idx = self._robot.body_names.index(self.cfg.ctrl.left_finger_body_name)
        self.right_finger_body_idx = self._robot.body_names.index(self.cfg.ctrl.right_finger_body_name)
        self.fingertip_body_idx = self._robot.body_names.index(self.cfg.ctrl.fingertip_body_name)
        if self.cfg.ctrl.held_body_name:
            self._held_body_idx = self._robot.body_names.index(self.cfg.ctrl.held_body_name)

        # Tensors for finite-differencing.
        self.last_update_timestamp = 0.0  # Note: This is for finite differencing body velocities.
        self.prev_fingertip_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_fingertip_quat = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        )
        self.prev_joint_pos = torch.zeros((self.num_envs, 7), device=self.device)

        self.ep_succeeded = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.ep_success_times = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)

        # Observation/state history buffers (supports window_size > 1).
        single_obs_dim = sum([OBS_DIM_CFG[obs] for obs in self.cfg.obs_order]) + self.cfg.action_space
        single_state_dim = sum([STATE_DIM_CFG[state] for state in self.cfg.state_order]) + self.cfg.action_space
        self.obs_history_buf = torch.zeros(
            (self.num_envs, self.cfg.obs_window_size, single_obs_dim), device=self.device
        )
        self.state_history_buf = torch.zeros(
            (self.num_envs, self.cfg.state_window_size, single_state_dim), device=self.device
        )

        # [CUSTOM] Per-episode keypoints for box_lid_insert (sampled once per reset).
        if self.cfg_task.name == "box_lid_insert":
            n = 2 + max(self.cfg_task.num_reset_extra_kp, self.cfg_task.num_success_extra_kp)
            self.kp_lid_local = torch.zeros((self.num_envs, n, 3), device=self.device)
            self.kp_box_local = torch.zeros((self.num_envs, n, 3), device=self.device)
            self.kp_box_y_target = torch.zeros((self.num_envs, n), device=self.device)

        # [CUSTOM] Per-episode keypoints for rj45_insert — uniformly sampled on the
        # male bottom patch XY plane.  Buffers are zero-initialised here and filled
        # at each episode reset (see _reset_buffers).  Z is fixed per frame:
        #   male  → male_bottom_patch_z_local  (in held/male frame)
        #   female → female_rear_edge_z_local   (in fixed/female frame, decremented by progressive descent)
        if self.cfg_task.name == "rj45_insert":
            _N_RJ45_KP = self.cfg_task.num_reset_kp
            self.kp_rj45_female_z_init = self.cfg_task.female_rear_edge_z_local
            self.kp_rj45_male_local  = torch.zeros((self.num_envs, _N_RJ45_KP, 3), device=self.device)
            self.kp_rj45_female_local = torch.zeros((self.num_envs, _N_RJ45_KP, 3), device=self.device)
        # [CUSTOM] OLD: Fixed triangle keypoints for rj45_insert (kept for reference).
        # if self.cfg_task.name == "rj45_insert":
        #     x_lo, x_hi = self.cfg_task.male_bottom_patch_x_range_local
        #     y_lo, y_hi = self.cfg_task.male_bottom_patch_y_range_local
        #     z = self.cfg_task.male_bottom_patch_z_local
        #     self.kp_rj45_male_local = torch.tensor([
        #         [0.0,  y_hi, z],   # front centre
        #         [x_lo, y_lo, z],   # back left
        #         [x_hi, y_lo, z],   # back right
        #     ], dtype=torch.float32, device=self.device).unsqueeze(0).expand(self.num_envs, -1, -1).clone()
        #     z_f = self.cfg_task.female_rear_edge_z_local
        #     plug_dy = self.cfg_task.socket_target_y_local
        #     self.kp_rj45_female_z_init = z_f
        #     self.kp_rj45_female_local = torch.tensor([
        #         [0.0,   y_hi + plug_dy, z_f],   # front centre
        #         [x_lo,  y_lo + plug_dy, z_f],   # back left
        #         [x_hi,  y_lo + plug_dy, z_f],   # back right
        #     ], dtype=torch.float32, device=self.device).unsqueeze(0).expand(self.num_envs, -1, -1).clone()

        # [CUSTOM] Per-episode keypoints for bnc_insert: Z-only on plug; socket-side Z eases down
        # when close. Buffer size = max(num_reset_kp, num_success_kp).
        if self.cfg_task.name == "bnc_insert":
            _N_BNC_KP = max(self.cfg_task.num_reset_kp, self.cfg_task.num_success_kp)
            self.kp_bnc_local = torch.zeros((self.num_envs, _N_BNC_KP, 3), device=self.device)
            self.kp_bnc_fixed_local = torch.zeros((self.num_envs, _N_BNC_KP, 3), device=self.device)

    def _setup_scene(self):
        """Initialize simulation scene."""
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(), translation=(0.0, 0.0, -1.05))

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
        if not self.cfg.ctrl.held_body_name:
            self._held_asset = Articulation(self.cfg_task.held_asset)
        if self.cfg_task.name == "gear_mesh":
            self._small_gear_asset = Articulation(self.cfg_task.small_gear_cfg)
            self._large_gear_asset = Articulation(self.cfg_task.large_gear_cfg)

        self._apply_robot_articulation_props_on_source_env()

        self.scene.clone_environments(copy_from_source=True)
        if self.device == "cpu":
            # we need to explicitly filter collisions for CPU simulation
            self.scene.filter_collisions()

        self.scene.articulations["robot"] = self._robot
        if isinstance(self._fixed_asset, RigidObject):
            self.scene.rigid_objects["fixed_asset"] = self._fixed_asset
        else:
            self.scene.articulations["fixed_asset"] = self._fixed_asset
        if hasattr(self, "_held_asset"):
            self.scene.articulations["held_asset"] = self._held_asset
        if self.cfg_task.name == "gear_mesh":
            self.scene.articulations["small_gear"] = self._small_gear_asset
            self.scene.articulations["large_gear"] = self._large_gear_asset

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _apply_robot_articulation_props_on_source_env(self):
        """Apply robot articulation properties on the actual source articulation root.

        Some custom robot USDs place the articulation root on a child prim
        (for example ``/Robot/link_0``) rather than on ``/Robot``. Since the
        scene clones ``env_0`` after spawning, we need to patch the source env's
        real articulation root before cloning so every cloned environment inherits
        the correct fixed-base articulation setup.
        """
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

    def _compute_intermediate_values(self, dt):
        """Get values computed from raw tensors. This includes adding noise."""
        # TODO: A lot of these can probably only be set once?
        self.fixed_pos = self._fixed_asset.data.root_pos_w - self.scene.env_origins
        self.fixed_quat = self._fixed_asset.data.root_quat_w

        if hasattr(self, "_held_body_idx"):
            self.held_pos = self._robot.data.body_pos_w[:, self._held_body_idx] - self.scene.env_origins
            self.held_quat = self._robot.data.body_quat_w[:, self._held_body_idx]
        else:
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

        self.last_update_timestamp = self._robot._data._sim_timestamp

    def _update_obs_state_history(self, obs_tensors, state_tensors):
        """Shift history buffers left and insert the latest obs/state at the end.

        Returns flattened tensors of shape (num_envs, obs_window_size * single_obs_dim)
        and (num_envs, state_window_size * single_state_dim).
        """
        if self.cfg.obs_window_size > 1:
            self.obs_history_buf[:, :-1] = self.obs_history_buf[:, 1:].clone()
        self.obs_history_buf[:, -1] = obs_tensors

        if self.cfg.state_window_size > 1:
            self.state_history_buf[:, :-1] = self.state_history_buf[:, 1:].clone()
        self.state_history_buf[:, -1] = state_tensors

        obs_out = self.obs_history_buf.view(self.num_envs, -1)
        state_out = self.state_history_buf.view(self.num_envs, -1)
        if not hasattr(self, "_obs_shape_printed"):
            print(f"[DEBUG] obs_out shape: {obs_out.shape}  (window={self.cfg.obs_window_size})")
            print(f"[DEBUG] state_out shape: {state_out.shape}  (window={self.cfg.state_window_size})")
            self._obs_shape_printed = True
        return obs_out, state_out

    def _get_factory_obs_state_dict(self):
        """Populate dictionaries for the policy and critic."""
        noisy_fixed_pos = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise

        prev_actions = self.actions.clone()

        obs_dict = {
            "fingertip_pos": self.fingertip_midpoint_pos,
            "fingertip_pos_rel_fixed": self.fingertip_midpoint_pos - noisy_fixed_pos,
            "fingertip_quat": self.fingertip_midpoint_quat,
            "ee_linvel": self.ee_linvel_fd,
            "ee_angvel": self.ee_angvel_fd,
            "prev_actions": prev_actions,
        }

        state_dict = {
            "fingertip_pos": self.fingertip_midpoint_pos,
            "fingertip_pos_rel_fixed": self.fingertip_midpoint_pos - self.fixed_pos_obs_frame,
            "fingertip_quat": self.fingertip_midpoint_quat,
            "ee_linvel": self.fingertip_midpoint_linvel,
            "ee_angvel": self.fingertip_midpoint_angvel,
            "joint_pos": self.joint_pos[:, 0:7],
            "held_pos": self.held_pos,
            "held_pos_rel_fixed": self.held_pos - self.fixed_pos_obs_frame,
            "held_quat": self.held_quat,
            "fixed_pos": self.fixed_pos,
            "fixed_quat": self.fixed_quat,
            "task_prop_gains": self.task_prop_gains,
            "pos_threshold": self.pos_threshold,
            "rot_threshold": self.rot_threshold,
            "prev_actions": prev_actions,
        }
        return obs_dict, state_dict

    def _get_observations(self):
        """Get actor/critic inputs using asymmetric critic."""
        obs_dict, state_dict = self._get_factory_obs_state_dict()

        obs_tensors = factory_utils.collapse_obs_dict(obs_dict, self.cfg.obs_order + ["prev_actions"])
        state_tensors = factory_utils.collapse_obs_dict(state_dict, self.cfg.state_order + ["prev_actions"])
        obs_out, state_out = self._update_obs_state_history(obs_tensors, state_tensors)
        return {"policy": obs_out, "critic": state_out}

    def _reset_buffers(self, env_ids):
        """Reset buffers."""
        self.ep_succeeded[env_ids] = 0
        self.ep_success_times[env_ids] = 0

    def _pre_physics_step(self, action):
        """Apply policy actions with smoothing."""
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self._reset_buffers(env_ids)

        self.actions = self.ema_factor * action.clone().to(self.device) + (1 - self.ema_factor) * self.actions

    def close_gripper_in_place(self):
        """Keep gripper in current position as gripper closes."""
        actions = torch.zeros((self.num_envs, 6), device=self.device)

        # Interpret actions as target pos displacements and set pos target
        pos_actions = actions[:, 0:3] * self.pos_threshold
        ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions

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
        ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)

        target_euler_xyz = torch.stack(torch_utils.get_euler_xyz(ctrl_target_fingertip_midpoint_quat), dim=1)
        target_euler_xyz[:, 0] = 3.14159
        target_euler_xyz[:, 1] = 0.0

        ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            roll=target_euler_xyz[:, 0], pitch=target_euler_xyz[:, 1], yaw=target_euler_xyz[:, 2]
        )

        self.generate_ctrl_signals(
            ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
            ctrl_target_gripper_dof_pos=0.0,
        )

    def _apply_action(self):
        """Apply actions for policy as delta targets from current position."""
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

        ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions
        # To speed up learning, never allow the policy to move more than 5cm away from the base.
        fixed_pos_action_frame = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        delta_pos = ctrl_target_fingertip_midpoint_pos - fixed_pos_action_frame
        pos_error_clipped = torch.clip(
            delta_pos, -self.cfg.ctrl.pos_action_bounds[0], self.cfg.ctrl.pos_action_bounds[1]
        )
        ctrl_target_fingertip_midpoint_pos = fixed_pos_action_frame + pos_error_clipped

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)

        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        rot_actions_quat = torch.where(
            angle.unsqueeze(-1).repeat(1, 4) > 1e-6,
            rot_actions_quat,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1),
        )
        ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)

        target_euler_xyz = torch.stack(torch_utils.get_euler_xyz(ctrl_target_fingertip_midpoint_quat), dim=1)
        target_euler_xyz[:, 0] = 3.14159  # Restrict actions to be upright.
        target_euler_xyz[:, 1] = 0.0

        ctrl_target_fingertip_midpoint_quat = torch_utils.quat_from_euler_xyz(
            roll=target_euler_xyz[:, 0], pitch=target_euler_xyz[:, 1], yaw=target_euler_xyz[:, 2]
        )

        self.generate_ctrl_signals(
            ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
            ctrl_target_gripper_dof_pos=0.0,
        )

    def generate_ctrl_signals(
        self, ctrl_target_fingertip_midpoint_pos, ctrl_target_fingertip_midpoint_quat, ctrl_target_gripper_dof_pos
    ):
        """Get Jacobian. Set Franka DOF position targets (fingers) or DOF torques (arm)."""
        self.joint_torque, self.applied_wrench = factory_control.compute_dof_torque(
            cfg=self.cfg,
            dof_pos=self.joint_pos,
            dof_vel=self.joint_vel,
            fingertip_midpoint_pos=self.fingertip_midpoint_pos,
            fingertip_midpoint_quat=self.fingertip_midpoint_quat,
            fingertip_midpoint_linvel=self.fingertip_midpoint_linvel,
            fingertip_midpoint_angvel=self.fingertip_midpoint_angvel,
            jacobian=self.fingertip_midpoint_jacobian,
            arm_mass_matrix=self.arm_mass_matrix,
            ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat,
            task_prop_gains=self.task_prop_gains,
            task_deriv_gains=self.task_deriv_gains,
            device=self.device,
            dead_zone_thresholds=self.dead_zone_thresholds,
        )

        # set target for gripper joints to use physx's PD controller
        self.ctrl_target_joint_pos[:, 7:9] = ctrl_target_gripper_dof_pos
        self.joint_torque[:, 7:9] = 0.0

        self._robot.set_joint_position_target(self.ctrl_target_joint_pos)
        self._robot.set_joint_effort_target(self.joint_torque)

    def _get_dones(self):
        """Check which environments are terminated.

        For Factory reset logic, it is important that all environments
        stay in sync (i.e., _get_dones should return all true or all false).
        """
        self._compute_intermediate_values(dt=self.physics_dt)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if bool(getattr(self.cfg_task, "terminate_on_success", False)):
            check_rot = self.cfg_task.name == "nut_thread"
            success_now = self._get_curr_successes(
                self.cfg_task.success_threshold, check_rot=check_rot
            )
            return success_now, time_out & ~success_now
        return time_out, time_out

    def _get_curr_successes(self, success_threshold, check_rot=False):
        """Get success mask at current timestep."""
        curr_successes = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        held_base_pos, held_base_quat = factory_utils.get_held_base_pose(
            self.held_pos, self.held_quat, self.cfg_task.name, self.cfg_task.fixed_asset_cfg, self.num_envs, self.device
        )
        target_held_base_pos, target_held_base_quat = factory_utils.get_target_held_base_pose(
            self.fixed_pos,
            self.fixed_quat,
            self.cfg_task.name,
            self.cfg_task.fixed_asset_cfg,
            self.num_envs,
            self.device,
            task_cfg=self.cfg_task,
        )

        xy_dist = torch.linalg.vector_norm(target_held_base_pos[:, 0:2] - held_base_pos[:, 0:2], dim=1)
        z_disp = held_base_pos[:, 2] - target_held_base_pos[:, 2]

        is_centered = torch.where(xy_dist < 0.0025, torch.ones_like(curr_successes), torch.zeros_like(curr_successes))
        fixed_cfg = self.cfg_task.fixed_asset_cfg
        if self.cfg_task.name in ("peg_insert", "gear_mesh"):
            height_threshold = fixed_cfg.height * success_threshold
            is_close_or_below = torch.where(
                z_disp < height_threshold, torch.ones_like(curr_successes), torch.zeros_like(curr_successes)
            )
            curr_successes = torch.logical_and(is_centered, is_close_or_below)
        elif self.cfg_task.name == "nut_thread":
            height_threshold = fixed_cfg.thread_pitch * success_threshold
            is_close_or_below = torch.where(
                z_disp < height_threshold, torch.ones_like(curr_successes), torch.zeros_like(curr_successes)
            )
            curr_successes = torch.logical_and(is_centered, is_close_or_below)
        elif self.cfg_task.name == "box_lid_insert":
            # Box STL pocket geometry (box local frame, metres):
            #   Left  pocket centre X = -0.025218, Right pocket centre X = +0.024250
            #   Pocket width (X) = 11 mm, height (Z) = [0.021, 0.029], depth (Y) = [-0.0495, -0.0375]
            #   Groove mouth at Y = -0.0375, back wall at Y = -0.0495
            #   45° chamfers at groove mouth guide clip entry
            # Lid STL clip geometry (lid local frame, metres):
            #   Clip X centres match pocket centres; tooth top Z = 0.0289, outer face Y = -0.0444
            #
            # Two distinct checks dispatched by success_threshold value:
            #   engage_threshold  = 0.9  → engaged:  X aligned, clip inside groove (Y), Z reasonable
            #   success_threshold = 0.04 → success:  X + Y at front wall + Z in pocket window
            _LEFT_HOLE_X  = -0.025218
            _RIGHT_HOLE_X =  0.024250

            ident_q = (
                torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
                .unsqueeze(0)
                .expand(self.num_envs, -1)
            )

            # Clip tooth reference points in lid local frame (from lid STL).
            left_clip_local = torch.zeros((self.num_envs, 3), device=self.device)
            left_clip_local[:, 0] = -0.025218
            left_clip_local[:, 1] = -0.0444
            left_clip_local[:, 2] = 0.0289

            right_clip_local = torch.zeros((self.num_envs, 3), device=self.device)
            right_clip_local[:, 0] = 0.024250
            right_clip_local[:, 1] = -0.0444
            right_clip_local[:, 2] = 0.0289

            # Lid local → world → box local frame.
            _, left_clip_w  = torch_utils.tf_combine(self.held_quat, self.held_pos, ident_q, left_clip_local)
            _, right_clip_w = torch_utils.tf_combine(self.held_quat, self.held_pos, ident_q, right_clip_local)
            left_clip_box  = quat_apply_inverse(self.fixed_quat, left_clip_w  - self.fixed_pos)
            right_clip_box = quat_apply_inverse(self.fixed_quat, right_clip_w - self.fixed_pos)

            if success_threshold >= 0.5:
                # Engaged: clips are X-aligned AND inside the groove (Y) AND at a
                # reasonable height (Z). Z upper bound is relaxed above the box top so
                # the policy gets a signal while the lid is still descending from above.
                #   X: ±2 mm — groove entry needs correct alignment (chamfers only 1 mm wide)
                #   Y: [-49.5, -37.5] mm — clip is inside the groove (mouth → back wall)
                #   Z: [18, 35] mm — groove floor (21 mm) with tolerance, up to 5 mm above box
                _X_TOL = 0.002
                _Y_MIN, _Y_MAX = -0.0495,  0.035
                _Z_MIN, _Z_MAX =  0.018,   0.030

                left_x_ok  = (left_clip_box[:,  0] - _LEFT_HOLE_X).abs()  < _X_TOL
                right_x_ok = (right_clip_box[:, 0] - _RIGHT_HOLE_X).abs() < _X_TOL
                left_y_ok  = (left_clip_box[:,  1] > _Y_MIN) & (left_clip_box[:,  1] < _Y_MAX)
                right_y_ok = (right_clip_box[:, 1] > _Y_MIN) & (right_clip_box[:, 1] < _Y_MAX)
                left_z_ok  = (left_clip_box[:,  2] > _Z_MIN) & (left_clip_box[:,  2] < _Z_MAX)
                right_z_ok = (right_clip_box[:, 2] > _Z_MIN) & (right_clip_box[:, 2] < _Z_MAX)
            else:
                # Success: clips fully seated in pocket — X + Y at front wall + Z in window.
                #   X: ±2 mm from pocket centre
                #   Y: ±2 mm from outer clip face / front wall (Y = -0.0444)
                #   Z: [21, 29] mm — full pocket Z window
                _X_TOL        = 0.002
                _Y_TOL        = 0.003 #used to be 0.002, relaxed to 4mm to account for slight misalignments that still constitute success 
                _HOLE_WALL_Y  = -0.0444
                _Z_MIN, _Z_MAX = 0.021, 0.029

                left_x_ok  = (left_clip_box[:,  0] - _LEFT_HOLE_X).abs()  < _X_TOL
                right_x_ok = (right_clip_box[:, 0] - _RIGHT_HOLE_X).abs() < _X_TOL
                left_y_ok  = (left_clip_box[:,  1] - _HOLE_WALL_Y).abs()  < _Y_TOL
                right_y_ok = (right_clip_box[:, 1] - _HOLE_WALL_Y).abs()  < _Y_TOL
                left_z_ok  = (left_clip_box[:,  2] > _Z_MIN) & (left_clip_box[:,  2] < _Z_MAX)
                right_z_ok = (right_clip_box[:, 2] > _Z_MIN) & (right_clip_box[:, 2] < _Z_MAX)

            curr_successes = (
                left_x_ok & left_y_ok & left_z_ok & right_x_ok & right_y_ok & right_z_ok
            )
        elif self.cfg_task.name == "rj45_insert":
            # [CUSTOM] RJ45 insertion — two distinct checks dispatched by threshold value:
            #
            #   engage_threshold  > 1.0  (e.g. 2.0)  → ENGAGE: XY alignment + yaw + tilt
            #   success_threshold < 0    (e.g. -0.002) → SUCCESS: tip at full-insertion depth
            #
            # socket_target_y/z_local from task cfg (single source of truth).
            # z_disp = 0 at full insertion; negative = tip deeper than target.
            ident_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).expand(self.num_envs, -1)
            socket_opening_local = torch.zeros((self.num_envs, 3), device=self.device)
            socket_opening_local[:, 1] = self.cfg_task.socket_target_y_local
            socket_opening_local[:, 2] = self.cfg_task.socket_target_z_local
            _, socket_opening_world = torch_utils.tf_combine(
                self.fixed_quat, self.fixed_pos, ident_q, socket_opening_local
            )
            z_disp = held_base_pos[:, 2] - socket_opening_world[:, 2]
            xy_dist = torch.linalg.vector_norm(
                socket_opening_world[:, 0:2] - held_base_pos[:, 0:2], dim=1
            )

            if success_threshold > 1.0:
                # ── ENGAGE check ──────────────────────────────────────────────────────
                # XY: tip within 8 mm of cavity centre (accounts for -6 mm Y offset).
                _XY_TOL = 0.004
                is_xy = xy_dist < _XY_TOL

                # Yaw: plug yaw should match socket yaw within ±15°.
                _, _, plug_yaw = torch_utils.get_euler_xyz(self.held_quat)
                _, _, sock_yaw = torch_utils.get_euler_xyz(self.fixed_quat)
                yaw_diff = (plug_yaw - sock_yaw + torch.pi) % (2 * torch.pi) - torch.pi
                is_yaw = yaw_diff.abs() < 0.262  # ~15°

                # Tilt: plug's local -Z axis must point close to world -Z (vertical).
                plug_z_local = torch.zeros((self.num_envs, 3), device=self.device)
                plug_z_local[:, 2] = -1.0
                plug_z_world = torch_utils.quat_rotate(self.held_quat, plug_z_local)
                cos_tilt = -plug_z_world[:, 2]
                is_tilt = cos_tilt > 0.966  # deviation < ~15°

                # Z (loose): tip within 40 mm above the target tip position.
                # At engage zone (30 mm above cavity): z_disp = 0.030 < 0.040 ✓
                is_z = z_disp < 0.06

                curr_successes = is_xy & is_yaw & is_tilt & is_z
            else:
                # ── SUCCESS check ──────────────────────────────────────────────────
                # height_threshold = 0.0 + success_threshold = -0.002 m.
                # Fires when tip is 2 mm below the full-insertion target (z_disp < -0.002).
                _XY_STRICT = 0.004
                height_threshold = fixed_cfg.height + success_threshold
                is_inside = z_disp < height_threshold
                is_xy_strict = xy_dist < _XY_STRICT
                curr_successes = is_inside & is_xy_strict

        elif self.cfg_task.name == "bnc_insert":
            # [CUSTOM] BNC Small insertion — two distinct checks dispatched by threshold value:
            #
            #   engage_threshold  > 1.0  (e.g. 2.0)  → ENGAGE: XY + π-sym yaw + tilt + Z
            #   success_threshold < 0    (e.g. -0.3)  → SUCCESS: tip depth + XY (no yaw)
            ident_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).expand(self.num_envs, -1)
            socket_opening_local = torch.zeros((self.num_envs, 3), device=self.device)
            socket_opening_local[:, 2] = fixed_cfg.height  # = 0.025 m
            _, socket_opening_world = torch_utils.tf_combine(
                self.fixed_quat, self.fixed_pos, ident_q, socket_opening_local
            )
            z_disp  = held_base_pos[:, 2] - socket_opening_world[:, 2]
            xy_dist = torch.linalg.vector_norm(
                socket_opening_world[:, 0:2] - held_base_pos[:, 0:2], dim=1
            )

            if success_threshold > 1.0:
                # ── ENGAGE check ──────────────────────────────────────────────────────
                # 1. XY: tip centre within 4 mm of socket opening centre.
                is_xy = xy_dist < 0.004

                # 2. Yaw (180° symmetry, BNC bayonet).
                _, _, plug_yaw = torch_utils.get_euler_xyz(self.held_quat)
                _, _, sock_yaw = torch_utils.get_euler_xyz(self.fixed_quat)
                yaw_diff_raw = (plug_yaw - sock_yaw + torch.pi) % (2 * torch.pi) - torch.pi
                yaw_diff_sym = torch.minimum(yaw_diff_raw.abs(), torch.pi - yaw_diff_raw.abs())
                is_yaw = yaw_diff_sym < 0.175  # ~10°

                # 3. Tilt: plug -Z axis points toward world -Z (vertical, < 15° deviation).
                plug_z_local = torch.zeros((self.num_envs, 3), device=self.device)
                plug_z_local[:, 2] = -1.0
                plug_z_world = torch_utils.quat_rotate(self.held_quat, plug_z_local)
                is_tilt = -plug_z_world[:, 2] > 0.966  # cos(15°) ≈ 0.966

                # 4. Z: not too far above opening, and optionally minimum depth inside (inclusive).
                z_loose_max = float(getattr(self.cfg_task, "bnc_engage_z_max_above_opening", 0.040))
                min_depth = float(getattr(self.cfg_task, "bnc_engage_min_depth_m", 0.0))
                is_z = z_disp < z_loose_max
                if min_depth > 0.0:
                    is_z = is_z & (z_disp <= -min_depth)

                already_succeeded = self.ep_succeeded.bool()
                curr_successes = is_xy & (is_yaw | already_succeeded) & is_tilt & is_z
            else:
                # ── SUCCESS check ─────────────────────────────────────────────────
                # Tip must be inside socket past height_threshold on z_disp (tip minus opening),
                # XY aligned.  Base: height × success_threshold; optional bnc_success_min_depth_m.
                height_threshold = fixed_cfg.height * success_threshold
                succ_min = float(getattr(self.cfg_task, "bnc_success_min_depth_m", 0.0))
                if succ_min > 0.0:
                    height_threshold = min(height_threshold, -succ_min)
                # Inclusive at threshold; small eps avoids sim / transform float noise at the boundary.
                _z_eps = float(getattr(self.cfg_task, "bnc_success_z_eps_m", 1e-5))
                is_inside    = z_disp <= height_threshold + _z_eps
                is_xy_strict = xy_dist < 0.003
                curr_successes = is_inside & is_xy_strict

        else:
            raise NotImplementedError("Task not implemented")
        
        # is_close_or_below = torch.where(
        #     z_disp < height_threshold, torch.ones_like(curr_successes), torch.zeros_like(curr_successes)
        # )
        # curr_successes = torch.logical_and(is_centered, is_close_or_below)

        if check_rot:
            _, _, curr_yaw = torch_utils.get_euler_xyz(self.fingertip_midpoint_quat)
            curr_yaw = factory_utils.wrap_yaw(curr_yaw)
            is_rotated = curr_yaw < self.cfg_task.ee_success_yaw
            curr_successes = torch.logical_and(curr_successes, is_rotated)

        return curr_successes

    def _log_factory_metrics(self, rew_dict, curr_successes):
        """Keep track of episode statistics and log rewards."""
        # Only log episode success rates at the end of an episode.
        if torch.any(self.reset_buf):
            self.extras["successes"] = torch.count_nonzero(curr_successes) / self.num_envs

        # Get the time at which an episode first succeeds.
        first_success = torch.logical_and(curr_successes, torch.logical_not(self.ep_succeeded))
        self.ep_succeeded[curr_successes] = 1

        first_success_ids = first_success.nonzero(as_tuple=False).squeeze(-1)
        self.ep_success_times[first_success_ids] = self.episode_length_buf[first_success_ids]

        # [CUSTOM] RJ45 Phase 2: on first success, replace Z-axis keypoints with random body
        # keypoints spread across the connector head volume (full XYZ).  Denser signal for
        # sustained deep insertion once the plug is initially aligned and partially inserted.
        # if self.cfg_task.name == "rj45_insert" and len(first_success_ids) > 0:
        #     ns = self.cfg_task.num_success_kp
        #     r = torch.rand((len(first_success_ids), ns, 3), device=self.device)
        #     # Offsets in TIP frame (Z=0 at tip).  Connector cross-section + shallow depth zone.
        #     # X ∈ [-18.75mm, +18.75mm], Y ∈ [-5mm, +13mm], Z ∈ [0, +14mm] (tip → connector face area)
        #     self.kp_rj45_local[first_success_ids, :ns, 0] = r[:, :, 0] * 0.0375 - 0.01875
        #     self.kp_rj45_local[first_success_ids, :ns, 1] = r[:, :, 1] * 0.0182 - 0.00503
        #     self.kp_rj45_local[first_success_ids, :ns, 2] = r[:, :, 2] * 0.014

        # [CUSTOM] For box_lid_insert: on first success, keep the 2 clip keypoints (index 0,1)
        # and replace index 2..2+ns-1 with random keypoints spread across the lid body.
        # Updated here so the NEXT call to _get_factory_rew_dict picks up the new keypoints.
        if self.cfg_task.name == "box_lid_insert" and len(first_success_ids) > 0:
            ns = self.cfg_task.num_success_extra_kp
            kp = torch.rand((len(first_success_ids), ns, 3), device=self.device)
            kp[:, :, 0] = kp[:, :, 0] * 0.1046 - 0.0523  # X: [-52.3, +52.3] mm
            kp[:, :, 1] = kp[:, :, 1] * 0.0838 - 0.0444  # Y: [-44.4, +39.4] mm
            kp[:, :, 2] = kp[:, :, 2] * 0.0112 + 0.0188  # Z: [+18.8, +30.0] mm
            self.kp_lid_local[first_success_ids, 2:2 + ns] = kp
            self.kp_box_local[first_success_ids, 2:2 + ns] = kp
            dz_box = float(getattr(self.cfg_task, "kp_box_z_offset", 0.0))
            if dz_box != 0.0:
                self.kp_box_local[first_success_ids, 2 : 2 + ns, 2] += dz_box
            self.kp_box_y_target[first_success_ids, 2 : 2 + ns] = kp[:, :, 1].clone()
            y0 = self.cfg_task.kp_advance_y_start
            self.kp_box_local[first_success_ids, 2 : 2 + ns, 1] = y0
        nonzero_success_ids = self.ep_success_times.nonzero(as_tuple=False).squeeze(-1)

        if len(nonzero_success_ids) > 0:  # Only log for successful episodes.
            success_times = self.ep_success_times[nonzero_success_ids].sum() / len(nonzero_success_ids)
            self.extras["success_times"] = success_times

        for rew_name, rew in rew_dict.items():
            self.extras[f"logs_rew_{rew_name}"] = rew.mean()

    def _get_rewards(self):
        """Update rewards and compute success statistics."""
        # Get successful and failed envs at current timestep
        check_rot = self.cfg_task.name == "nut_thread"
        curr_successes = self._get_curr_successes(
            success_threshold=self.cfg_task.success_threshold, check_rot=check_rot
        )
        # BNC: optional — mask off success (no curr_success reward, no ep_succeeded latch).
        if self.cfg_task.name == "bnc_insert" and not bool(
            getattr(self.cfg_task, "bnc_apply_success_criteria", True)
        ):
            curr_successes = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        rew_dict, rew_scales = self._get_factory_rew_dict(curr_successes)

        rew_buf = torch.zeros_like(rew_dict["kp_coarse"])
        for rew_name, rew in rew_dict.items():
            rew_buf += rew_dict[rew_name] * rew_scales[rew_name]

        self.prev_actions = self.actions.clone()

        self._log_factory_metrics(rew_dict, curr_successes)
        return rew_buf

    def _get_factory_rew_dict(self, curr_successes):
        """Compute reward terms at current timestep."""
        rew_dict, rew_scales = {}, {}

        # Compute pos of keypoints on held asset, and fixed asset in world frame.
        ident = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        if self.cfg_task.name == "box_lid_insert":
            # [CUSTOM] Per-episode keypoints: lid side stays at success geometry; box-side Y
            # eases from kp_advance_y_start toward kp_box_y_target (see _get_factory_rew_dict).
            n = 2 + max(self.cfg_task.num_reset_extra_kp, self.cfg_task.num_success_extra_kp)
            keypoints_held  = torch.zeros((self.num_envs, n, 3), device=self.device)
            keypoints_fixed = torch.zeros((self.num_envs, n, 3), device=self.device)
            for i in range(n):
                _, keypoints_held[:, i]  = torch_utils.tf_combine(
                    self.held_quat,  self.held_pos,  ident, self.kp_lid_local[:, i]
                )
                _, keypoints_fixed[:, i] = torch_utils.tf_combine(
                    self.fixed_quat, self.fixed_pos, ident, self.kp_box_local[:, i]
                )
        elif self.cfg_task.name == "rj45_insert":
            n_kp = self.kp_rj45_male_local.shape[1]
            keypoints_held  = torch.zeros((self.num_envs, n_kp, 3), device=self.device)
            keypoints_fixed = torch.zeros((self.num_envs, n_kp, 3), device=self.device)
            for i in range(n_kp):
                _, keypoints_held[:, i] = torch_utils.tf_combine(
                    self.held_quat, self.held_pos, ident, self.kp_rj45_male_local[:, i]
                )
                _, keypoints_fixed[:, i] = torch_utils.tf_combine(
                    self.fixed_quat, self.fixed_pos, ident, self.kp_rj45_female_local[:, i]
                )
        elif self.cfg_task.name == "bnc_insert":
            # Z-axis keypoints on plug; socket uses kp_bnc_fixed_local (eased down when close).
            held_base_pos_kp, held_base_quat_kp = factory_utils.get_held_base_pose(
                self.held_pos, self.held_quat, self.cfg_task.name,
                self.cfg_task.fixed_asset_cfg, self.num_envs, self.device,
            )
            target_base_pos_kp, target_base_quat_kp = factory_utils.get_target_held_base_pose(
                self.fixed_pos, self.fixed_quat, self.cfg_task.name,
                self.cfg_task.fixed_asset_cfg, self.num_envs, self.device,
                task_cfg=self.cfg_task,
            )
            n_kp = self.kp_bnc_local.shape[1]
            keypoints_held  = torch.zeros((self.num_envs, n_kp, 3), device=self.device)
            keypoints_fixed = torch.zeros((self.num_envs, n_kp, 3), device=self.device)
            for i in range(n_kp):
                _, keypoints_held[:, i]  = torch_utils.tf_combine(
                    held_base_quat_kp,   held_base_pos_kp,   ident, self.kp_bnc_local[:, i]
                )
                _, keypoints_fixed[:, i] = torch_utils.tf_combine(
                    target_base_quat_kp, target_base_pos_kp, ident, self.kp_bnc_fixed_local[:, i]
                )
        else:
            held_base_pos, held_base_quat = factory_utils.get_held_base_pose(
                self.held_pos, self.held_quat, self.cfg_task.name, self.cfg_task.fixed_asset_cfg, self.num_envs, self.device
            )
            target_held_base_pos, target_held_base_quat = factory_utils.get_target_held_base_pose(
                self.fixed_pos,
                self.fixed_quat,
                self.cfg_task.name,
                self.cfg_task.fixed_asset_cfg,
                self.num_envs,
                self.device,
                task_cfg=self.cfg_task,
            )
            keypoints_held  = torch.zeros((self.num_envs, self.cfg_task.num_keypoints, 3), device=self.device)
            keypoints_fixed = torch.zeros((self.num_envs, self.cfg_task.num_keypoints, 3), device=self.device)
            offsets = factory_utils.get_keypoint_offsets(self.cfg_task.num_keypoints, self.device)
            keypoint_offsets = offsets * self.cfg_task.keypoint_scale
            for idx, keypoint_offset in enumerate(keypoint_offsets):
                keypoints_held[:, idx] = torch_utils.tf_combine(
                    held_base_quat, held_base_pos, ident, keypoint_offset.repeat(self.num_envs, 1),
                )[1]
                keypoints_fixed[:, idx] = torch_utils.tf_combine(
                    target_held_base_quat, target_held_base_pos, ident, keypoint_offset.repeat(self.num_envs, 1),
                )[1]
        keypoint_dist = torch.norm(keypoints_held - keypoints_fixed, p=2, dim=-1).mean(-1)

        # [CUSTOM] Frame distance: keypoint XYZ dist + weighted rotation error.
        # rot_dist = geodesic angle (rad) between held and fixed asset orientations.
        # Target orientation for held asset = fixed asset orientation (self.fixed_quat).
        # rot_weight = 0.0 reduces to pure keypoint_dist (default for upstream tasks).
        rot_dist = factory_utils.quat_rpy_dist(self.held_quat, self.fixed_quat)
        frame_dist = keypoint_dist + self.cfg_task.rot_weight * rot_dist

        # [CUSTOM] Progressive descent for rj45: when male keypoints are close
        # enough, pull female target Z downward to guide deeper insertion.
        if self.cfg_task.name == "rj45_insert":
            close = keypoint_dist < self.cfg_task.kp_advance_threshold
            if close.any():
                self.kp_rj45_female_local[close, :, 2] -= self.cfg_task.kp_advance_step
                self.kp_rj45_female_local[:, :, 2].clamp_(min=self.cfg_task.kp_advance_z_limit)

        # [CUSTOM] Progressive box keypoint Y for box_lid_insert: lid targets stay fixed;
        # box-side Y walks from kp_advance_y_start toward true success Y per keypoint.
        if self.cfg_task.name == "box_lid_insert":
            close = keypoint_dist < self.cfg_task.kp_advance_threshold
            if close.any():
                step = self.cfg_task.kp_advance_y_step
                cur = self.kp_box_local[:, :, 1]
                tgt = self.kp_box_y_target
                diff = tgt - cur
                delta = torch.sign(diff) * torch.minimum(diff.abs(), torch.full_like(diff, step))
                self.kp_box_local[close, :, 1] = cur[close] + delta[close]

        # [CUSTOM] BNC: when close enough, step socket-side Z down toward plug locals; snap on success.
        if self.cfg_task.name == "bnc_insert":
            nr = self.cfg_task.num_reset_kp
            thr = float(getattr(self.cfg_task, "bnc_kp_advance_threshold", 0.005))
            step = float(getattr(self.cfg_task, "bnc_kp_advance_step", 0.0002))
            close = keypoint_dist < thr
            if close.any():
                self.kp_bnc_fixed_local[close, :nr, 2] -= step
                self.kp_bnc_fixed_local[:, :nr, 2] = torch.maximum(
                    self.kp_bnc_fixed_local[:, :nr, 2], self.kp_bnc_local[:, :nr, 2]
                )
            succ = curr_successes.bool()
            if succ.any():
                self.kp_bnc_fixed_local[succ] = self.kp_bnc_local[succ].clone()

        a0, b0 = self.cfg_task.keypoint_coef_baseline
        a1, b1 = self.cfg_task.keypoint_coef_coarse
        a2, b2 = self.cfg_task.keypoint_coef_fine
        # Action penalties.
        action_penalty_ee = torch.norm(self.actions, p=2)
        action_grad_penalty = torch.norm(self.actions - self.prev_actions, p=2, dim=-1)
        curr_engaged = self._get_curr_successes(success_threshold=self.cfg_task.engage_threshold, check_rot=False)

        rew_dict = {
            # "kp_baseline": factory_utils.squashing_fn(keypoint_dist, a0, b0),
            # "kp_coarse": factory_utils.squashing_fn(keypoint_dist, a1, b1),
            # "kp_fine": factory_utils.squashing_fn(keypoint_dist, a2, b2),
            "kp_baseline": factory_utils.squashing_fn(frame_dist, a0, b0),
            "kp_coarse": factory_utils.squashing_fn(frame_dist, a1, b1),
            "kp_fine": factory_utils.squashing_fn(frame_dist, a2, b2),
            "action_penalty_ee": action_penalty_ee,
            "action_grad_penalty": action_grad_penalty,
            "curr_engaged": curr_engaged.float(),
            "curr_success": curr_successes.float(),
        }
        rew_scales = {
            "kp_baseline": 1.0,
            "kp_coarse": 1.0,
            "kp_fine": 1.0,
            "action_penalty_ee": -self.cfg_task.action_penalty_ee_scale,
            "action_grad_penalty": -self.cfg_task.action_grad_penalty_scale,
            "curr_engaged": 1.0,
            "curr_success": 1.0,
        }
        return rew_dict, rew_scales

    def _reset_idx(self, env_ids):
        """We assume all envs will always be reset at the same time."""
        super()._reset_idx(env_ids)

        # Clear history buffers for reset environments.
        self.obs_history_buf[env_ids] = 0.0
        self.state_history_buf[env_ids] = 0.0

        self._set_assets_to_default_pose(env_ids)
        self._set_franka_to_default_pose(joints=self.cfg.ctrl.reset_joints, env_ids=env_ids)
        self.step_sim_no_action()

        self.randomize_initial_state(env_ids)

        # [CUSTOM] Reset female keypoint Z for rj45_insert progressive descent.
        if self.cfg_task.name == "rj45_insert":
            self.kp_rj45_female_local[env_ids, :, 2] = self.kp_rj45_female_z_init

        # [CUSTOM] RJ45 keypoints: per-episode uniform XY plane sample.
        # num_reset_kp points sampled uniformly in male_bottom_patch XY, Z fixed per frame.
        # Female keypoints = same XY + socket_target_y_local (Y offset), Z = female_rear_edge_z_local.
        # NOTE: female Z reset (progressive-descent reset) runs just above; the assignment
        # below overwrites it consistently with the newly sampled envs.
        if self.cfg_task.name == "rj45_insert":
            n = self.cfg_task.num_reset_kp
            x_lo, x_hi = self.cfg_task.male_bottom_patch_x_range_local
            y_lo, y_hi = self.cfg_task.male_bottom_patch_y_range_local
            z_m = self.cfg_task.male_bottom_patch_z_local
            z_f = self.cfg_task.female_rear_edge_z_local
            plug_dy = self.cfg_task.socket_target_y_local
            x_rand = torch.rand((len(env_ids), n), device=self.device) * (x_hi - x_lo) + x_lo
            y_rand = torch.rand((len(env_ids), n), device=self.device) * (y_hi - y_lo) + y_lo
            self.kp_rj45_male_local[env_ids, :, 0] = x_rand
            self.kp_rj45_male_local[env_ids, :, 1] = y_rand
            self.kp_rj45_male_local[env_ids, :, 2] = z_m
            self.kp_rj45_female_local[env_ids, :, 0] = x_rand
            self.kp_rj45_female_local[env_ids, :, 1] = y_rand + plug_dy
            self.kp_rj45_female_local[env_ids, :, 2] = z_f
        # [CUSTOM] OLD: Z-axis only keypoints (kept for reference).
        # if self.cfg_task.name == "rj45_insert":
        #     nr = self.cfg_task.num_reset_kp
        #     self.kp_rj45_local[env_ids] = 0.0
        #     r = torch.rand((len(env_ids), nr), device=self.device)
        #     self.kp_rj45_local[env_ids, :nr, 2] = r * (0.08847 + 0.003)

        # [CUSTOM] BNC: Z-axis keypoints; Z ~ center ± half_spread on plug, socket + init extra on Z.
        if self.cfg_task.name == "bnc_insert":
            nr = self.cfg_task.num_reset_kp
            self.kp_bnc_local[env_ids] = 0.0
            r = torch.rand((len(env_ids), nr), device=self.device)
            z_c = float(getattr(self.cfg_task, "bnc_kp_z_center", 0.055))
            z_hs = float(getattr(self.cfg_task, "bnc_kp_z_half_spread", 0.024))
            self.kp_bnc_local[env_ids, :nr, 2] = z_c + (r - 0.5) * (2.0 * z_hs)
            self.kp_bnc_fixed_local[env_ids] = self.kp_bnc_local[env_ids].clone()
            z_base = float(getattr(self.cfg_task, "bnc_kp_socket_z_init_extra", 0.0))
            z_hi = float(getattr(self.cfg_task, "bnc_kp_socket_z_above_engage_m", 0.0))
            z_bump = z_base + z_hi
            if z_bump != 0.0:
                self.kp_bnc_fixed_local[env_ids, :nr, 2] += z_bump

        # [CUSTOM] Keypoints for box_lid_insert (box Y reset below).
        if self.cfg_task.name == "box_lid_insert":
            nr = self.cfg_task.num_reset_extra_kp
            ns = self.cfg_task.num_success_extra_kp
            n_total = 2 + max(nr, ns)
            left_lid = torch.tensor([-0.025218, -0.0444, 0.0289], device=self.device)
            right_lid = torch.tensor([0.024250, -0.0444, 0.0289], device=self.device)
            ease_l = getattr(self.cfg_task, "kp_box_clip_ease_end_left", None)
            ease_r = getattr(self.cfg_task, "kp_box_clip_ease_end_right", None)
            use_box_clip_ease_end = (
                ease_l is not None
                and ease_r is not None
                and len(ease_l) == 3
                and len(ease_r) == 3
            )

            # index 0,1: clip positions — lid locals nominal; box locals may use measured ease-end XYZ.
            self.kp_lid_local[env_ids, 0] = left_lid.unsqueeze(0).expand(len(env_ids), -1)
            self.kp_lid_local[env_ids, 1] = right_lid.unsqueeze(0).expand(len(env_ids), -1)
            if use_box_clip_ease_end:
                left_box_end = torch.tensor(ease_l, device=self.device, dtype=torch.float32)
                right_box_end = torch.tensor(ease_r, device=self.device, dtype=torch.float32)
                self.kp_box_local[env_ids, 0] = left_box_end.unsqueeze(0).expand(len(env_ids), -1)
                self.kp_box_local[env_ids, 1] = right_box_end.unsqueeze(0).expand(len(env_ids), -1)
            else:
                for idx, lc in ((0, left_lid), (1, right_lid)):
                    self.kp_box_local[env_ids, idx] = lc.unsqueeze(0).expand(len(env_ids), -1)

            # index 2..2+nr-1: extra front-face keypoints (same Y,Z as clips; X random in lid front)
            if nr > 0:
                x_rand = torch.rand((len(env_ids), nr), device=self.device) * 0.1046 - 0.0523
                kp_front = torch.stack([
                    x_rand,
                    torch.full_like(x_rand, -0.0444),
                    torch.full_like(x_rand,  0.0289),
                ], dim=-1)  # (len(env_ids), nr, 3)
                for buf in (self.kp_lid_local, self.kp_box_local):
                    buf[env_ids, 2:2 + nr] = kp_front

            # If ns > nr, the spare slots 2+nr..n_total-1 are filled with left_clip
            # to avoid zero-initialization until first success replaces them.
            if ns > nr:
                spare = ns - nr
                spare_kp = left_lid.unsqueeze(0).unsqueeze(0).expand(len(env_ids), spare, -1)
                for buf in (self.kp_lid_local, self.kp_box_local):
                    buf[env_ids, 2 + nr:n_total] = spare_kp

            # Raise box-side keypoint Z only (optional); lid ``kp_lid_local`` unchanged.
            dz_box = float(getattr(self.cfg_task, "kp_box_z_offset", 0.0))
            if dz_box != 0.0:
                self.kp_box_local[env_ids, :, 2] += dz_box

            # Save true success Y per box keypoint, then ease box-side Y from start toward target.
            self.kp_box_y_target[env_ids] = self.kp_box_local[env_ids, :, 1].clone()
            y0 = self.cfg_task.kp_advance_y_start
            self.kp_box_local[env_ids, :, 1] = y0

    def _set_assets_to_default_pose(self, env_ids):
        """Move assets to default pose before randomization."""
        if hasattr(self, "_held_asset"):
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

    def set_pos_inverse_kinematics(
        self, ctrl_target_fingertip_midpoint_pos, ctrl_target_fingertip_midpoint_quat, env_ids
    ):
        """Set robot joint position using DLS IK."""
        ik_time = 0.0
        while ik_time < 0.25:
            # Compute error to target.
            pos_error, axis_angle_error = factory_control.get_pose_error(
                fingertip_midpoint_pos=self.fingertip_midpoint_pos[env_ids],
                fingertip_midpoint_quat=self.fingertip_midpoint_quat[env_ids],
                ctrl_target_fingertip_midpoint_pos=ctrl_target_fingertip_midpoint_pos[env_ids],
                ctrl_target_fingertip_midpoint_quat=ctrl_target_fingertip_midpoint_quat[env_ids],
                jacobian_type="geometric",
                rot_error_type="axis_angle",
            )

            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)

            # Solve DLS problem.
            delta_dof_pos = factory_control.get_delta_dof_pos(
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
            self._robot.set_joint_position_target(self.ctrl_target_joint_pos)

            # Simulate and update tensors.
            self.step_sim_no_action()
            ik_time += self.physics_dt

        return pos_error, axis_angle_error

    def get_handheld_asset_relative_pose(self):
        """Get default relative pose between help asset and fingertip."""
        if self.cfg_task.name == "peg_insert":
            held_asset_relative_pos = torch.zeros((self.num_envs, 3), device=self.device)
            held_asset_relative_pos[:, 2] = self.cfg_task.held_asset_cfg.height
            held_asset_relative_pos[:, 2] -= self.cfg_task.robot_cfg.franka_fingerpad_length
        elif self.cfg_task.name == "gear_mesh":
            held_asset_relative_pos = torch.zeros((self.num_envs, 3), device=self.device)
            gear_base_offset = self.cfg_task.fixed_asset_cfg.medium_gear_base_offset
            held_asset_relative_pos[:, 0] += gear_base_offset[0]
            held_asset_relative_pos[:, 2] += gear_base_offset[2]
            held_asset_relative_pos[:, 2] += self.cfg_task.held_asset_cfg.height / 2.0 * 1.1
        elif self.cfg_task.name == "nut_thread":
            held_asset_relative_pos = factory_utils.get_held_base_pos_local(
                self.cfg_task.name, self.cfg_task.fixed_asset_cfg, self.num_envs, self.device
            )
        elif self.cfg_task.name == "box_lid_insert":
            # [CUSTOM] Compute the default EE → lid transform so that the Franka
            # finger pads are centred on the lid handle when the gripper closes.
            #
            # Step 1 — Z offset along the (flipped) fingertip Z axis:
            #   The finger pads grip the handle at the handle TOP.
            #   Handle top height from USD origin = held_asset_cfg.height = 0.055 m.
            #   Franka finger pad centre is franka_fingerpad_length = 0.0176 m below
            #   the fingertip frame origin.
            #   ∴ Z offset = 0.055 − 0.0176 = 0.0374 m
            #   (i.e. the asset origin is 37.4 mm "below" the fingertip in Z).
            held_asset_relative_pos = torch.zeros((self.num_envs, 3), device=self.device)
            held_asset_relative_pos[:, 2] = (
                self.cfg_task.held_asset_cfg.height - self.cfg_task.robot_cfg.franka_fingerpad_length
            )
            # Step 2 — Fine-tune with held_asset_pos_offset [x, y, z] (metres).
            #   This is a per-task empirical offset tuned to keep the handle
            #   centred between the pads after the gripper closes.
            #   Default = [0, 0.02, 0.005]: 20 mm inward along Y, 5 mm along Z.
            pos_offset = torch.tensor(self.cfg_task.held_asset_pos_offset, device=self.device)
            held_asset_relative_pos += pos_offset.unsqueeze(0)
        elif self.cfg_task.name == "rj45_insert":
            # [CUSTOM] RJ45 male plug: grip near the cable end (sim_Z ≈ height = 0.077 m).
            held_asset_relative_pos = torch.zeros((self.num_envs, 3), device=self.device)
            held_asset_relative_pos[:, 2] = (
                self.cfg_task.held_asset_cfg.height - self.cfg_task.robot_cfg.franka_fingerpad_length
            )
            pos_offset = torch.tensor(self.cfg_task.held_asset_pos_offset, device=self.device)
            held_asset_relative_pos += pos_offset.unsqueeze(0)
        elif self.cfg_task.name == "bnc_insert":
            # [CUSTOM] BNC Small male plug: grip at the body centre (height = 0.0717 m).
            # Franka finger pad centre is franka_fingerpad_length = 0.0176 m below fingertip.
            # For the Kuka embedded-link variant this value is not used directly
            # (body position comes from the robot's link_bnc).
            held_asset_relative_pos = torch.zeros((self.num_envs, 3), device=self.device)
            held_asset_relative_pos[:, 2] = (
                self.cfg_task.held_asset_cfg.height - self.cfg_task.robot_cfg.franka_fingerpad_length
            )
            pos_offset = torch.tensor(self.cfg_task.held_asset_pos_offset, device=self.device)
            held_asset_relative_pos += pos_offset.unsqueeze(0)
        else:
            raise NotImplementedError("Task not implemented")

        held_asset_relative_quat = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        )
        # [CUSTOM] "box_lid_insert" uses the same initial orientation logic as
        # "nut_thread": apply held_asset_rot_init (base yaw) plus held_asset_rot_offset
        # (additional roll/pitch/yaw) to orient the lid in the gripper frame.
        #
        # For box_lid_insert the defaults are:
        #   held_asset_rot_init  = 90°  yaw  — aligns lid long axis with robot approach
        #   held_asset_rot_offset = [0°, 35°, 0°]  — 35° pitch tilts the handle
        #                            forward so the body clears the finger pads.
        if self.cfg_task.name in ("nut_thread", "box_lid_insert", "rj45_insert"):
            # Rotate along z-axis of frame for default position.
            initial_rot_deg = self.cfg_task.held_asset_rot_init
            rot_offset = getattr(self.cfg_task, "held_asset_rot_offset", [0.0, 0.0, 0.0])
            rot_euler = torch.tensor(
                [
                    rot_offset[0] * np.pi / 180.0,  # roll
                    rot_offset[1] * np.pi / 180.0,  # pitch
                    (initial_rot_deg + rot_offset[2]) * np.pi / 180.0,  # yaw
                ],
                device=self.device,
            ).repeat(self.num_envs, 1)
            held_asset_relative_quat = torch_utils.quat_from_euler_xyz(
                roll=rot_euler[:, 0], pitch=rot_euler[:, 1], yaw=rot_euler[:, 2]
            )

        return held_asset_relative_pos, held_asset_relative_quat

    def _set_franka_to_default_pose(self, joints, env_ids):
        """Return Franka to its default joint position."""
        gripper_width = self.cfg_task.held_asset_cfg.diameter / 2 * 1.25
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
        """Step the simulation without an action. Used for resets only.

        This method should only be called during resets when all environments
        reset at the same time.
        """
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(dt=self.physics_dt)
        self._compute_intermediate_values(dt=self.physics_dt)

    def randomize_initial_state(self, env_ids):
        """Randomize initial state and perform any episode-level randomization."""
        # Disable gravity.
        physics_sim_view = sim_utils.SimulationContext.instance().physics_sim_view
        physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, 0.0))

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
        self._fixed_asset.write_root_pose_to_sim(fixed_state[:, 0:7], env_ids=env_ids)
        self._fixed_asset.write_root_velocity_to_sim(fixed_state[:, 7:], env_ids=env_ids)
        self._fixed_asset.reset()

        # (1.e.) Noisy position observation.
        fixed_asset_pos_noise = torch.randn((len(env_ids), 3), dtype=torch.float32, device=self.device)
        fixed_asset_pos_rand = torch.tensor(self.cfg.obs_rand.fixed_asset_pos, dtype=torch.float32, device=self.device)
        fixed_asset_pos_noise = fixed_asset_pos_noise @ torch.diag(fixed_asset_pos_rand)
        self.init_fixed_pos_obs_noise[env_ids] = fixed_asset_pos_noise

        self.step_sim_no_action()

        # Compute the frame on the bolt that would be used as observation: fixed_pos_obs_frame
        # For example, the tip of the bolt can be used as the observation frame
        fixed_tip_pos_local = torch.zeros((self.num_envs, 3), device=self.device)
        fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.height
        fixed_tip_pos_local[:, 2] += self.cfg_task.fixed_asset_cfg.base_height
        if self.cfg_task.name == "gear_mesh":
            fixed_tip_pos_local[:, 0] = self.cfg_task.fixed_asset_cfg.medium_gear_base_offset[0]

        _, fixed_tip_pos = torch_utils.tf_combine(
            self.fixed_quat,
            self.fixed_pos,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
            fixed_tip_pos_local,
        )
        self.fixed_pos_obs_frame[:] = fixed_tip_pos

        # (2) Move gripper to randomizes location above fixed asset. Keep trying until IK succeeds.
        # --- Per-env near/far init mode (computed once; stable across IK retry iterations) ---
        _init_mode = getattr(self.cfg_task, "init_mode", "far")
        _near_prob  = getattr(self.cfg_task, "near_init_prob", 0.5)
        _is_near = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if _init_mode == "near":
            _is_near[env_ids] = True
        elif _init_mode == "mixed":
            _near_mask = torch.rand(len(env_ids), device=self.device) < _near_prob
            _is_near[env_ids[_near_mask]] = True
        # "far": _is_near stays all False

        # Contact-init: sample a desired held-asset pose touching the fixed part,
        # then convert to fingertip IK targets.  RJ45: plug vs socket rear edge.
        # Box: lid front rim vs box rear top edge (left/right halves paired).
        # BNC: female bore vs male tip face (bnc_contact_init_*).
        # Works for Kuka (embedded link) and Franka (separate held asset).
        use_rj45_contact_init = (
            self.cfg_task.name == "rj45_insert"
            and _init_mode == "contact"
        )
        use_box_contact_init = (
            self.cfg_task.name == "box_lid_insert"
            and _init_mode == "contact"
        )
        use_bnc_contact_init = (
            self.cfg_task.name == "bnc_insert"
            and _init_mode == "contact"
        )
        use_contact_style_init = use_rj45_contact_init or use_box_contact_init or use_bnc_contact_init
        held_to_fingertip_quat = held_to_fingertip_pos = None
        if use_contact_style_init:
            flip_z_quat = torch.tensor(
                [0.0, 0.0, 1.0, 0.0], device=self.device
            ).unsqueeze(0).repeat(self.num_envs, 1)
            if self.cfg.ctrl.held_body_name in ("link_rj45", "link_lid", "link_bnc"):
                # Kuka: measure fingertip<->held link transform from the current
                # reset pose (embedded robot body).
                fingertip_inv_quat, fingertip_inv_pos = torch_utils.tf_inverse(
                    self.fingertip_midpoint_quat,
                    self.fingertip_midpoint_pos,
                )
                fingertip_to_held_quat, fingertip_to_held_pos = torch_utils.tf_combine(
                    fingertip_inv_quat,
                    fingertip_inv_pos,
                    self.held_quat,
                    self.held_pos,
                )
                held_to_fingertip_quat, held_to_fingertip_pos = torch_utils.tf_inverse(
                    fingertip_to_held_quat,
                    fingertip_to_held_pos,
                )
            else:
                # Franka: derive held_to_fingertip from the fixed grasp offset.
                # Teleportation places held asset via:
                #   held_world = tf_combine(fingertip * flip_z, inv(held_asset_relative))
                # Inverting:
                #   fingertip = held_world * held_asset_relative * flip_z
                # So: held_to_fingertip = tf_combine(held_asset_relative, flip_z)
                h_rel_pos, h_rel_quat = self.get_handheld_asset_relative_pose()
                held_to_fingertip_quat, held_to_fingertip_pos = torch_utils.tf_combine(
                    h_rel_quat,
                    h_rel_pos,
                    flip_z_quat,
                    torch.zeros((self.num_envs, 3), device=self.device),
                )

        # (a) get position vector to target
        bad_envs = env_ids.clone()
        ik_attempt = 0

        hand_down_quat = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        while True:
            n_bad = bad_envs.shape[0]

            above_fixed_pos = fixed_tip_pos.clone()
            above_fixed_pos[:, 2] += self.cfg_task.hand_init_pos[2]
            if use_contact_style_init:
                ident_n = torch.tensor(
                    [1.0, 0.0, 0.0, 0.0], device=self.device
                ).unsqueeze(0).expand(n_bad, -1)

                # Same convention as RJ45: small RPY jitter in the *fixed/box* frame only
                # (no large nominal euler — that wrongly over-rotated the lid in contact init).
                roll_lo, roll_hi = self.cfg_task.contact_init_roll_range_deg
                pitch_lo, pitch_hi = self.cfg_task.contact_init_pitch_range_deg
                yaw_lo, yaw_hi = self.cfg_task.contact_init_yaw_range_deg
                roll = torch.deg2rad(
                    torch.rand(n_bad, device=self.device) * (roll_hi - roll_lo) + roll_lo
                )
                pitch = torch.deg2rad(
                    torch.rand(n_bad, device=self.device) * (pitch_hi - pitch_lo) + pitch_lo
                )
                yaw = torch.deg2rad(
                    torch.rand(n_bad, device=self.device) * (yaw_hi - yaw_lo) + yaw_lo
                )
                if use_bnc_contact_init:
                    yaw = yaw + torch.randint(0, 2, (n_bad,), device=self.device).float() * torch.pi
                held_contact_quat = torch_utils.quat_mul(
                    self.fixed_quat[bad_envs],
                    torch_utils.quat_from_euler_xyz(roll, pitch, yaw),
                )

                female_point_local = torch.zeros((n_bad, 3), device=self.device)
                male_point_local = torch.zeros((n_bad, 3), device=self.device)
                if use_rj45_contact_init:
                    x_lo, x_hi = self.cfg_task.female_rear_edge_x_range_local
                    female_point_local[:, 0] = torch.rand(n_bad, device=self.device) * (x_hi - x_lo) + x_lo
                    female_point_local[:, 1] = self.cfg_task.female_rear_edge_y_local
                    female_point_local[:, 2] = self.cfg_task.female_rear_edge_z_local
                    x_lo, x_hi = self.cfg_task.male_bottom_patch_x_range_local
                    y_lo, y_hi = self.cfg_task.male_bottom_patch_y_range_local
                    male_point_local[:, 0] = torch.rand(n_bad, device=self.device) * (x_hi - x_lo) + x_lo
                    male_point_local[:, 1] = torch.rand(n_bad, device=self.device) * (y_hi - y_lo) + y_lo
                    male_point_local[:, 2] = self.cfg_task.male_bottom_patch_z_local
                elif use_bnc_contact_init:
                    fxl, fxh = self.cfg_task.bnc_contact_init_female_x_range
                    fyl, fyh = self.cfg_task.bnc_contact_init_female_y_range
                    female_point_local[:, 0] = torch.rand(n_bad, device=self.device) * (fxh - fxl) + fxl
                    female_point_local[:, 1] = torch.rand(n_bad, device=self.device) * (fyh - fyl) + fyl
                    female_point_local[:, 2] = self.cfg_task.bnc_contact_init_female_z_local
                    mxl, mxh = self.cfg_task.bnc_contact_init_male_x_range
                    myl, myh = self.cfg_task.bnc_contact_init_male_y_range
                    male_point_local[:, 0] = torch.rand(n_bad, device=self.device) * (mxh - mxl) + mxl
                    male_point_local[:, 1] = torch.rand(n_bad, device=self.device) * (myh - myl) + myl
                    male_point_local[:, 2] = self.cfg_task.bnc_contact_init_male_z_local
                else:
                    side = torch.randint(0, 2, (n_bad,), device=self.device)
                    bx_lo, bx_hi = self.cfg_task.box_contact_rear_edge_x_range
                    bx_mid = 0.5 * (bx_lo + bx_hi)
                    x_lo_b = torch.where(
                        side == 0,
                        torch.full((n_bad,), bx_lo, device=self.device),
                        torch.full((n_bad,), bx_mid, device=self.device),
                    )
                    x_hi_b = torch.where(
                        side == 0,
                        torch.full((n_bad,), bx_mid, device=self.device),
                        torch.full((n_bad,), bx_hi, device=self.device),
                    )
                    female_point_local[:, 0] = (
                        torch.rand(n_bad, device=self.device) * (x_hi_b - x_lo_b) + x_lo_b
                    )
                    female_point_local[:, 1] = self.cfg_task.box_contact_rear_edge_y_local
                    female_point_local[:, 2] = self.cfg_task.box_contact_rear_edge_z_local

                    lx_lo, lx_hi = self.cfg_task.lid_contact_front_edge_x_range
                    lx_mid = 0.5 * (lx_lo + lx_hi)
                    x_lo_l = torch.where(
                        side == 0,
                        torch.full((n_bad,), lx_lo, device=self.device),
                        torch.full((n_bad,), lx_mid, device=self.device),
                    )
                    x_hi_l = torch.where(
                        side == 0,
                        torch.full((n_bad,), lx_mid, device=self.device),
                        torch.full((n_bad,), lx_hi, device=self.device),
                    )
                    male_point_local[:, 0] = (
                        torch.rand(n_bad, device=self.device) * (x_hi_l - x_lo_l) + x_lo_l
                    )
                    male_point_local[:, 1] = self.cfg_task.lid_contact_front_edge_y_local
                    male_point_local[:, 2] = self.cfg_task.lid_contact_front_edge_z_local

                _, female_point_world = torch_utils.tf_combine(
                    self.fixed_quat[bad_envs],
                    self.fixed_pos[bad_envs],
                    ident_n,
                    female_point_local,
                )

                held_contact_pos = female_point_world - torch_utils.quat_rotate(
                    held_contact_quat,
                    male_point_local,
                )
                fingertip_target_quat, fingertip_target_pos = torch_utils.tf_combine(
                    held_contact_quat,
                    held_contact_pos,
                    held_to_fingertip_quat[bad_envs],
                    held_to_fingertip_pos[bad_envs],
                )
                above_fixed_pos[bad_envs] = fingertip_target_pos
                hand_down_quat[bad_envs, :] = fingertip_target_quat
            else:
                rand_sample = torch.rand((n_bad, 3), dtype=torch.float32, device=self.device)
                above_fixed_pos_rand = 2 * (rand_sample - 0.5)  # [-1, 1]
                hand_init_pos_rand = torch.tensor(self.cfg_task.hand_init_pos_noise, device=self.device)
                above_fixed_pos_rand = above_fixed_pos_rand @ torch.diag(hand_init_pos_rand)
                above_fixed_pos[bad_envs] += above_fixed_pos_rand

                # For box_lid_insert: apply XY and Z in box-local frame (box-yaw aligned).
                if self.cfg_task.name == "box_lid_insert":
                    xy_local = torch.zeros((self.num_envs, 3), device=self.device)
                    # Near mode: fixed XY offset from hand_init_pos (directly above box).
                    xy_local[_is_near, 0] = self.cfg_task.hand_init_pos[0]
                    xy_local[_is_near, 1] = self.cfg_task.hand_init_pos[1]
                    # Far mode: random XYZ from cfg ranges (box-local frame).
                    _is_far = ~_is_near
                    if _is_far.any():
                        x_lo, x_hi = self.cfg_task.hand_init_x_range
                        y_lo, y_hi = self.cfg_task.hand_init_y_range
                        z_lo, z_hi = self.cfg_task.hand_init_z_range
                        r = torch.rand((self.num_envs, 3), device=self.device)
                        xy_local[_is_far, 0] = r[_is_far, 0] * (x_hi - x_lo) + x_lo
                        xy_local[_is_far, 1] = r[_is_far, 1] * (y_hi - y_lo) + y_lo
                        above_fixed_pos[_is_far, 2] = (
                            fixed_tip_pos[_is_far, 2] + r[_is_far, 2] * (z_hi - z_lo) + z_lo
                        )
                    identity = torch.tensor(
                        [1.0, 0.0, 0.0, 0.0], device=self.device
                    ).unsqueeze(0).repeat(self.num_envs, 1)
                    _, xy_world = torch_utils.tf_combine(
                        self.fixed_quat,
                        torch.zeros((self.num_envs, 3), device=self.device),
                        identity,
                        xy_local,
                    )
                    above_fixed_pos += xy_world

                # Far-mode position override for rj45_insert and bnc_insert.
                # Near mode keeps the default fixed hand_init_pos directly above the socket.
                # Far mode: XY sampled in socket-local frame and rotated to world; Z above socket opening.
                if self.cfg_task.name in ("rj45_insert", "bnc_insert"):
                    far_mask = ~_is_near[bad_envs]
                    if far_mask.any():
                        far_env_ids = bad_envs[far_mask]
                        n_far = int(far_mask.sum())
                        x_lo, x_hi = self.cfg_task.hand_init_x_range
                        y_lo, y_hi = self.cfg_task.hand_init_y_range
                        z_lo, z_hi = self.cfg_task.hand_init_z_range
                        r = torch.rand((n_far, 3), device=self.device)
                        xy_local_far = torch.zeros((n_far, 3), device=self.device)
                        xy_local_far[:, 0] = r[:, 0] * (x_hi - x_lo) + x_lo
                        xy_local_far[:, 1] = r[:, 1] * (y_hi - y_lo) + y_lo
                        ident_n = torch.tensor(
                            [1.0, 0.0, 0.0, 0.0], device=self.device
                        ).unsqueeze(0).expand(n_far, -1)
                        _, xy_world_far = torch_utils.tf_combine(
                            self.fixed_quat[far_env_ids],
                            torch.zeros((n_far, 3), device=self.device),
                            ident_n, xy_local_far,
                        )
                        above_fixed_pos[far_env_ids, :2] = (
                            fixed_tip_pos[far_env_ids, :2] + xy_world_far[:, :2]
                        )
                        above_fixed_pos[far_env_ids, 2] = (
                            fixed_tip_pos[far_env_ids, 2] + r[:, 2] * (z_hi - z_lo) + z_lo
                        )

                # (b) get random orientation facing down
                hand_down_euler = (
                    torch.tensor(self.cfg_task.hand_init_orn, device=self.device).unsqueeze(0).repeat(n_bad, 1)
                )

                rand_sample = torch.rand((n_bad, 3), dtype=torch.float32, device=self.device)
                above_fixed_orn_noise = 2 * (rand_sample - 0.5)  # [-1, 1]
                hand_init_orn_rand = torch.tensor(self.cfg_task.hand_init_orn_noise, device=self.device)
                above_fixed_orn_noise = above_fixed_orn_noise @ torch.diag(hand_init_orn_rand)
                hand_down_euler += above_fixed_orn_noise

                # For box_lid_insert: align gripper yaw with box yaw so clips always face pockets,
                # then add ±yaw and ±pitch noise for initial pose diversity.
                if self.cfg_task.name == "box_lid_insert":
                    _, _, box_yaw = torch_utils.get_euler_xyz(self.fixed_quat[bad_envs])
                    hand_down_euler[:, 2] = box_yaw - 0.5 * torch.pi
                    yaw_noise = (torch.rand(n_bad, device=self.device) * 2 - 1) * np.deg2rad(
                        self.cfg_task.hand_init_yaw_noise_deg
                    )
                    pitch_noise = (torch.rand(n_bad, device=self.device) * 2 - 1) * np.deg2rad(
                        self.cfg_task.hand_init_pitch_noise_deg
                    )
                    hand_down_euler[:, 2] += yaw_noise
                    hand_down_euler[:, 1] += pitch_noise

                # Yaw alignment for rj45_insert and bnc_insert.
                # Both near and far modes align plug yaw to socket yaw so the connector
                # faces the opening regardless of socket randomisation.
                # Far mode adds ± yaw noise on top; near mode is exact.
                if self.cfg_task.name in ("rj45_insert", "bnc_insert"):
                    # RJ45 connector face is rotated 90° relative to the socket in the robot frame,
                    # so add a 90° yaw offset on top of the socket yaw to align properly.
                    # _yaw_offset = 0.5 * torch.pi if self.cfg_task.name == "rj45_insert" else 0.0
                    _yaw_offset = np.deg2rad(getattr(self.cfg_task, "hand_init_yaw_offset_deg", 0.0))

                    # Near mode: exact socket yaw alignment (no noise).
                    near_mask = _is_near[bad_envs]
                    if near_mask.any():
                        _, _, sock_yaw_near = torch_utils.get_euler_xyz(self.fixed_quat[bad_envs[near_mask]])
                        hand_down_euler[near_mask, 2] = sock_yaw_near + _yaw_offset

                    # Far mode: socket yaw ± noise; BNC also picks 0° or 180° base.
                    far_mask = ~_is_near[bad_envs]
                    if far_mask.any():
                        n_far = int(far_mask.sum())
                        yaw_deg = getattr(self.cfg_task, "hand_init_yaw_noise_deg", 0.0)
                        _, _, sock_yaw = torch_utils.get_euler_xyz(self.fixed_quat[bad_envs[far_mask]])
                        yaw_noise = (torch.rand(n_far, device=self.device) * 2 - 1) * np.deg2rad(yaw_deg)
                        base_yaw = sock_yaw.clone() + _yaw_offset
                        if self.cfg_task.name == "bnc_insert":
                            # Two valid bayonet orientations: 0° or 180° relative to socket yaw.
                            flip = torch.randint(0, 2, (n_far,), device=self.device).float() * torch.pi
                            base_yaw = base_yaw + flip
                        hand_down_euler[far_mask, 2] = base_yaw + yaw_noise
                        # pitch stays at hand_init_orn[1] — EE straight down, no pitch noise

                hand_down_quat[bad_envs, :] = torch_utils.quat_from_euler_xyz(
                    roll=hand_down_euler[:, 0], pitch=hand_down_euler[:, 1], yaw=hand_down_euler[:, 2]
                )

            # (c) iterative IK Method
            pos_error, aa_error = self.set_pos_inverse_kinematics(
                ctrl_target_fingertip_midpoint_pos=above_fixed_pos,
                ctrl_target_fingertip_midpoint_quat=hand_down_quat,
                env_ids=bad_envs,
            )
            pos_error = torch.linalg.norm(pos_error, dim=1) > 1e-3
            angle_error = torch.norm(aa_error, dim=1) > 1e-3
            any_error = torch.logical_or(pos_error, angle_error)
            bad_envs = bad_envs[any_error.nonzero(as_tuple=False).squeeze(-1)]

            # Check IK succeeded for all envs, otherwise try again for those envs
            if bad_envs.shape[0] == 0 or ik_attempt >= 100:
                break

            self._set_franka_to_default_pose(joints=self.cfg.ctrl.reset_joints, env_ids=bad_envs)

            ik_attempt += 1

        self.step_sim_no_action()

        # Add flanking gears after servo (so arm doesn't move them).
        if self.cfg_task.name == "gear_mesh" and self.cfg_task.add_flanking_gears:
            small_gear_state = self._small_gear_asset.data.default_root_state.clone()[env_ids]
            small_gear_state[:, 0:7] = fixed_state[:, 0:7]
            small_gear_state[:, 7:] = 0.0  # vel
            self._small_gear_asset.write_root_pose_to_sim(small_gear_state[:, 0:7], env_ids=env_ids)
            self._small_gear_asset.write_root_velocity_to_sim(small_gear_state[:, 7:], env_ids=env_ids)
            self._small_gear_asset.reset()

            large_gear_state = self._large_gear_asset.data.default_root_state.clone()[env_ids]
            large_gear_state[:, 0:7] = fixed_state[:, 0:7]
            large_gear_state[:, 7:] = 0.0  # vel
            self._large_gear_asset.write_root_pose_to_sim(large_gear_state[:, 0:7], env_ids=env_ids)
            self._large_gear_asset.write_root_velocity_to_sim(large_gear_state[:, 7:], env_ids=env_ids)
            self._large_gear_asset.reset()

        # (3) Randomize asset-in-gripper location.
        # flip gripper z orientation
        flip_z_quat = torch.tensor([0.0, 0.0, 1.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        fingertip_flipped_quat, fingertip_flipped_pos = torch_utils.tf_combine(
            q1=self.fingertip_midpoint_quat,
            t1=self.fingertip_midpoint_pos,
            q2=flip_z_quat,
            t2=torch.zeros((self.num_envs, 3), device=self.device),
        )

        # get default gripper in asset transform
        held_asset_relative_pos, held_asset_relative_quat = self.get_handheld_asset_relative_pose()
        asset_in_hand_quat, asset_in_hand_pos = torch_utils.tf_inverse(
            held_asset_relative_quat, held_asset_relative_pos
        )

        translated_held_asset_quat, translated_held_asset_pos = torch_utils.tf_combine(
            q1=fingertip_flipped_quat, t1=fingertip_flipped_pos, q2=asset_in_hand_quat, t2=asset_in_hand_pos
        )

        
        
        # Set _DEBUG_OBSERVE_S = 0.0 to disable.
        # DEBUG: pause before closing gripper so you can inspect object placement.
        # print("Debug observe...")
        # _DEBUG_OBSERVE_S = 20.0
        # _t = 0.0
        # if not hasattr(self, "task_prop_gains"):
        #     self.task_prop_gains = self.default_gains.clone()
        #     self.task_deriv_gains = factory_utils.get_deriv_gains(self.task_prop_gains)
        # while _t < _DEBUG_OBSERVE_S:
        #     self.close_gripper_in_place()
        #     self.scene.write_data_to_sim()
        #     self.sim.step(render=True)  # render=True prevents Fabric clone failure
        #     self.scene.update(dt=self.physics_dt)
        #     self._compute_intermediate_values(dt=self.physics_dt)
        #     _t += self.sim.get_physics_dt()
        # print("Done observing, closing gripper...")
        
        # Add asset in hand randomization
        # rand_sample = torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
        # held_asset_pos_noise = 2 * (rand_sample - 0.5)  # [-1, 1]
        rand_sample = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        held_asset_pos_noise = 2 * (rand_sample)  # [-1, 1]
        if self.cfg_task.name == "gear_mesh":
            held_asset_pos_noise[:, 2] = -rand_sample[:, 2]  # [-1, 0]

        held_asset_pos_noise_level = torch.tensor(self.cfg_task.held_asset_pos_noise, device=self.device)
        held_asset_pos_noise = held_asset_pos_noise @ torch.diag(held_asset_pos_noise_level)
        ident_noise_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        translated_held_asset_quat, translated_held_asset_pos = torch_utils.tf_combine(
            q1=translated_held_asset_quat,
            t1=translated_held_asset_pos,
            q2=ident_noise_q,
            t2=held_asset_pos_noise,
        )

        if hasattr(self, "_held_asset"):
            held_state = self._held_asset.data.default_root_state.clone()
            held_state[:, 0:3] = translated_held_asset_pos + self.scene.env_origins
            held_state[:, 3:7] = translated_held_asset_quat
            held_state[:, 7:] = 0.0
            self._held_asset.write_root_pose_to_sim(held_state[:, 0:7])
            self._held_asset.write_root_velocity_to_sim(held_state[:, 7:])
            self._held_asset.reset()

        # DEBUG: pause before closing gripper so you can inspect object placement.
        # This runs before the "Close hand" gain setup below; ForgeEnv also sets task_prop_gains
        # only after super()._reset_idx. Seed default PD gains for close_gripper_in_place.
        # if not hasattr(self, "task_prop_gains"):
        #     self.task_prop_gains = self.default_gains.clone()
        #     self.task_deriv_gains = factory_utils.get_deriv_gains(self.task_prop_gains)
        # print("Debug observe...")
        # _DEBUG_OBSERVE_S = 20.0
        # _t = 0.0
        # while _t < _DEBUG_OBSERVE_S:
        #     self.close_gripper_in_place()
        #     self.scene.write_data_to_sim()
        #     self.sim.step(render=True)  # render=True prevents Fabric clone failure
        #     self.scene.update(dt=self.physics_dt)
        #     self._compute_intermediate_values(dt=self.physics_dt)
        #     _t += self.sim.get_physics_dt()
        # print("Done observing, closing gripper...")

        #  Close hand
        # Set gains to use for quick resets.
        reset_task_prop_gains = torch.tensor(self.cfg.ctrl.reset_task_prop_gains, device=self.device).repeat(
            (self.num_envs, 1)
        )
        self.task_prop_gains = reset_task_prop_gains
        self.task_deriv_gains = factory_utils.get_deriv_gains(
            reset_task_prop_gains, self.cfg.ctrl.reset_rot_deriv_scale
        )

        self.step_sim_no_action()


        if hasattr(self, "_held_asset"):
            grasp_time = 0.0
            while grasp_time < 0.25:
                self.ctrl_target_joint_pos[env_ids, 7:] = 0.0  # Close gripper.
                self.close_gripper_in_place()
                self.step_sim_no_action()
                grasp_time += self.sim.get_physics_dt()

        self.prev_joint_pos = self.joint_pos[:, 0:7].clone()
        self.prev_fingertip_pos = self.fingertip_midpoint_pos.clone()
        self.prev_fingertip_quat = self.fingertip_midpoint_quat.clone()

        # Set initial actions to involve no-movement. Needed for EMA/correct penalties.
        self.actions = torch.zeros_like(self.actions)
        self.prev_actions = torch.zeros_like(self.actions)

        # Zero initial velocity.
        self.ee_angvel_fd[:, :] = 0.0
        self.ee_linvel_fd[:, :] = 0.0

        # Set initial gains for the episode.
        self.task_prop_gains = self.default_gains
        self.task_deriv_gains = factory_utils.get_deriv_gains(self.default_gains)

        physics_sim_view.set_gravity(carb.Float3(*self.cfg.sim.gravity))
