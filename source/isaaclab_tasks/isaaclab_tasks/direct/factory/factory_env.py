# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch

import carb
import isaacsim.core.utils.torch as torch_utils

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
        # Update number of obs/states
        cfg.observation_space = sum([OBS_DIM_CFG[obs] for obs in cfg.obs_order])
        cfg.state_space = sum([STATE_DIM_CFG[state] for state in cfg.state_order])
        cfg.observation_space += cfg.action_space
        cfg.state_space += cfg.action_space
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
        self.left_finger_body_idx = self._robot.body_names.index("panda_leftfinger")
        self.right_finger_body_idx = self._robot.body_names.index("panda_rightfinger")
        self.fingertip_body_idx = self._robot.body_names.index("panda_fingertip_centered")

        # Tensors for finite-differencing.
        self.last_update_timestamp = 0.0  # Note: This is for finite differencing body velocities.
        self.prev_fingertip_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_fingertip_quat = (
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        )
        self.prev_joint_pos = torch.zeros((self.num_envs, 7), device=self.device)

        self.ep_succeeded = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)
        self.ep_success_times = torch.zeros((self.num_envs,), dtype=torch.long, device=self.device)

        # [CUSTOM] Per-episode keypoints for box_lid_insert (sampled once per reset).
        if self.cfg_task.name == "box_lid_insert":
            n = self.cfg_task.num_keypoints
            self.kp_lid_local = torch.zeros((self.num_envs, n, 3), device=self.device)
            self.kp_box_local = torch.zeros((self.num_envs, n, 3), device=self.device)

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
        self._held_asset = Articulation(self.cfg_task.held_asset)
        if self.cfg_task.name == "gear_mesh":
            self._small_gear_asset = Articulation(self.cfg_task.small_gear_cfg)
            self._large_gear_asset = Articulation(self.cfg_task.large_gear_cfg)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            # we need to explicitly filter collisions for CPU simulation
            self.scene.filter_collisions()

        self.scene.articulations["robot"] = self._robot
        if isinstance(self._fixed_asset, RigidObject):
            self.scene.rigid_objects["fixed_asset"] = self._fixed_asset
        else:
            self.scene.articulations["fixed_asset"] = self._fixed_asset
        self.scene.articulations["held_asset"] = self._held_asset
        if self.cfg_task.name == "gear_mesh":
            self.scene.articulations["small_gear"] = self._small_gear_asset
            self.scene.articulations["large_gear"] = self._large_gear_asset

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
        return {"policy": obs_tensors, "critic": state_tensors}

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
                _Z_MIN, _Z_MAX =  0.024,   0.030

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
                _Y_TOL        = 0.002
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
            # [CUSTOM] Use per-episode randomly sampled keypoints (fixed within episode).
            # kp_lid_local and kp_box_local are identical because at the success pose
            # the lid USD origin coincides with the box USD origin.
            n = self.cfg_task.num_keypoints
            keypoints_held  = torch.zeros((self.num_envs, n, 3), device=self.device)
            keypoints_fixed = torch.zeros((self.num_envs, n, 3), device=self.device)
            for i in range(n):
                _, keypoints_held[:, i]  = torch_utils.tf_combine(
                    self.held_quat,  self.held_pos,  ident, self.kp_lid_local[:, i]
                )
                _, keypoints_fixed[:, i] = torch_utils.tf_combine(
                    self.fixed_quat, self.fixed_pos, ident, self.kp_box_local[:, i]
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

        a0, b0 = self.cfg_task.keypoint_coef_baseline
        a1, b1 = self.cfg_task.keypoint_coef_coarse
        a2, b2 = self.cfg_task.keypoint_coef_fine
        # Action penalties.
        action_penalty_ee = torch.norm(self.actions, p=2)
        action_grad_penalty = torch.norm(self.actions - self.prev_actions, p=2, dim=-1)
        curr_engaged = self._get_curr_successes(success_threshold=self.cfg_task.engage_threshold, check_rot=False)

        rew_dict = {
            "kp_baseline": factory_utils.squashing_fn(keypoint_dist, a0, b0),
            "kp_coarse": factory_utils.squashing_fn(keypoint_dist, a1, b1),
            "kp_fine": factory_utils.squashing_fn(keypoint_dist, a2, b2),
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

        self._set_assets_to_default_pose(env_ids)
        self._set_franka_to_default_pose(joints=self.cfg.ctrl.reset_joints, env_ids=env_ids)
        self.step_sim_no_action()

        self.randomize_initial_state(env_ids)

        # [CUSTOM] Sample random keypoints in lid body bounds for box_lid_insert.
        # At success pose lid origin = box origin, so kp_box_local == kp_lid_local.
        if self.cfg_task.name == "box_lid_insert":
            n = self.cfg_task.num_keypoints
            kp = torch.rand((len(env_ids), n, 3), device=self.device)
            kp[:, :, 0] = kp[:, :, 0] * 0.1046 - 0.0523  # X: [-52.3, +52.3] mm
            kp[:, :, 1] = kp[:, :, 1] * 0.0838 - 0.0444  # Y: [-44.4, +39.4] mm
            kp[:, :, 2] = kp[:, :, 2] * 0.0112 + 0.0188  # Z: [+18.8, +30.0] mm
            self.kp_lid_local[env_ids] = kp
            self.kp_box_local[env_ids] = kp

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
        if self.cfg_task.name in ("nut_thread", "box_lid_insert"):
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
        self.init_fixed_pos_obs_noise[:] = fixed_asset_pos_noise

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
        # (a) get position vector to target
        bad_envs = env_ids.clone()
        ik_attempt = 0

        hand_down_quat = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        while True:
            n_bad = bad_envs.shape[0]

            above_fixed_pos = fixed_tip_pos.clone()
            above_fixed_pos[:, 2] += self.cfg_task.hand_init_pos[2]

            rand_sample = torch.rand((n_bad, 3), dtype=torch.float32, device=self.device)
            above_fixed_pos_rand = 2 * (rand_sample - 0.5)  # [-1, 1]
            hand_init_pos_rand = torch.tensor(self.cfg_task.hand_init_pos_noise, device=self.device)
            above_fixed_pos_rand = above_fixed_pos_rand @ torch.diag(hand_init_pos_rand)
            above_fixed_pos[bad_envs] += above_fixed_pos_rand

            # For box_lid_insert: apply XY from hand_init_pos in box-local frame.
            # Rotating through fixed_quat makes the offset follow the box yaw.
            if self.cfg_task.name == "box_lid_insert":
                xy_local = torch.tensor(
                    [self.cfg_task.hand_init_pos[0], self.cfg_task.hand_init_pos[1], 0.0],
                    device=self.device,
                ).unsqueeze(0).repeat(self.num_envs, 1)
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

            # (b) get random orientation facing down
            hand_down_euler = (
                torch.tensor(self.cfg_task.hand_init_orn, device=self.device).unsqueeze(0).repeat(n_bad, 1)
            )

            rand_sample = torch.rand((n_bad, 3), dtype=torch.float32, device=self.device)
            above_fixed_orn_noise = 2 * (rand_sample - 0.5)  # [-1, 1]
            hand_init_orn_rand = torch.tensor(self.cfg_task.hand_init_orn_noise, device=self.device)
            above_fixed_orn_noise = above_fixed_orn_noise @ torch.diag(hand_init_orn_rand)
            hand_down_euler += above_fixed_orn_noise

            # For box_lid_insert: align gripper yaw with box yaw so clips always face pockets.
            if self.cfg_task.name == "box_lid_insert":
                _, _, box_yaw = torch_utils.get_euler_xyz(self.fixed_quat[bad_envs])
                hand_down_euler[:, 2] = box_yaw - 0.5 * torch.pi

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
            if bad_envs.shape[0] == 0:
                break

            self._set_franka_to_default_pose(
                joints=[0.00871, -0.10368, -0.00794, -1.49139, -0.00083, 1.38774, 0.0], env_ids=bad_envs
            )

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
        
        # Add asset in hand randomization
        # rand_sample = torch.rand((self.num_envs, 3), dtype=torch.float32, device=self.device)
        # held_asset_pos_noise = 2 * (rand_sample - 0.5)  # [-1, 1]
        rand_sample = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        held_asset_pos_noise = 2 * (rand_sample)  # [-1, 1]
        if self.cfg_task.name == "gear_mesh":
            held_asset_pos_noise[:, 2] = -rand_sample[:, 2]  # [-1, 0]
    
        held_asset_pos_noise_level = torch.tensor(self.cfg_task.held_asset_pos_noise, device=self.device)
        held_asset_pos_noise = held_asset_pos_noise @ torch.diag(held_asset_pos_noise_level)
        translated_held_asset_quat, translated_held_asset_pos = torch_utils.tf_combine(
            q1=translated_held_asset_quat,
            t1=translated_held_asset_pos,
            q2=torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1),
            t2=held_asset_pos_noise,
        )

        held_state = self._held_asset.data.default_root_state.clone()
        held_state[:, 0:3] = translated_held_asset_pos + self.scene.env_origins
        held_state[:, 3:7] = translated_held_asset_quat
        held_state[:, 7:] = 0.0
        self._held_asset.write_root_pose_to_sim(held_state[:, 0:7])
        self._held_asset.write_root_velocity_to_sim(held_state[:, 7:])
        self._held_asset.reset()

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
