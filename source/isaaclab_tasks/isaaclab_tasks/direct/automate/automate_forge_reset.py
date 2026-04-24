# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Forge-style episode reset: fixed randomization, IK (near/far/contact), grasp, per-task keypoints.

Mirrors ``factory_env.FactoryEnv.randomize_initial_state`` / ``_reset_idx`` for
``box_lid_insert``, ``rj45_insert``, ``bnc_insert`` (incl. contact / far / near init) without duplicating fixed-asset
logic (delegates to ``AssemblyEnv.randomize_fixed_initial_state``).
"""

from __future__ import annotations

import carb
import numpy as np
import torch

import isaacsim.core.utils.torch as torch_utils
import isaaclab.sim as sim_utils


def randomize_initial_state_forge(env, env_ids: torch.Tensor) -> None:
    """Randomize initial state for Forge-parity AutoMate tasks (Franka + held asset)."""
    physics_sim_view = sim_utils.SimulationContext.instance().physics_sim_view
    physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, 0.0))

    env.randomize_fixed_initial_state(env_ids)

    cfg = env.cfg_task
    device = env.device
    num_envs = env.num_envs

    fixed_tip_pos_local = torch.zeros((num_envs, 3), device=device)
    fixed_tip_pos_local[:, 2] += cfg.fixed_asset_cfg.height
    fixed_tip_pos_local[:, 2] += cfg.fixed_asset_cfg.base_height

    ident = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0).repeat(num_envs, 1)
    _, fixed_tip_pos = torch_utils.tf_combine(env.fixed_quat, env.fixed_pos, ident, fixed_tip_pos_local)
    env.fixed_pos_obs_frame[:] = fixed_tip_pos

    _init_mode = getattr(cfg, "init_mode", "far")
    _near_prob = getattr(cfg, "near_init_prob", 0.5)
    _is_near = torch.zeros(num_envs, dtype=torch.bool, device=device)
    if _init_mode == "near":
        _is_near[env_ids] = True
    elif _init_mode == "mixed":
        _near_mask = torch.rand(len(env_ids), device=device) < _near_prob
        _is_near[env_ids[_near_mask]] = True

    use_rj45_contact_init = cfg.name == "rj45_insert" and _init_mode == "contact"
    use_box_contact_init = cfg.name == "box_lid_insert" and _init_mode == "contact"
    use_bnc_contact_init = cfg.name == "bnc_insert" and _init_mode == "contact"
    use_contact_style_init = use_rj45_contact_init or use_box_contact_init or use_bnc_contact_init
    held_to_fingertip_quat = held_to_fingertip_pos = None
    if use_contact_style_init:
        flip_z_quat = torch.tensor([0.0, 0.0, 1.0, 0.0], device=device).unsqueeze(0).repeat(num_envs, 1)
        if getattr(env.cfg.ctrl, "held_body_name", "") in ("link_rj45", "link_lid", "link_bnc"):
            fingertip_inv_quat, fingertip_inv_pos = torch_utils.tf_inverse(
                env.fingertip_midpoint_quat,
                env.fingertip_midpoint_pos,
            )
            fingertip_to_held_quat, fingertip_to_held_pos = torch_utils.tf_combine(
                fingertip_inv_quat,
                fingertip_inv_pos,
                env.held_quat,
                env.held_pos,
            )
            held_to_fingertip_quat, held_to_fingertip_pos = torch_utils.tf_inverse(
                fingertip_to_held_quat,
                fingertip_to_held_pos,
            )
        else:
            h_rel_pos, h_rel_quat = env.get_handheld_asset_relative_pose()
            held_to_fingertip_quat, held_to_fingertip_pos = torch_utils.tf_combine(
                h_rel_quat,
                h_rel_pos,
                flip_z_quat,
                torch.zeros((num_envs, 3), device=device),
            )

    bad_envs = env_ids.clone()
    ik_attempt = 0
    hand_down_quat = torch.zeros((num_envs, 4), dtype=torch.float32, device=device)

    while True:
        n_bad = bad_envs.shape[0]

        above_fixed_pos = fixed_tip_pos.clone()
        above_fixed_pos[:, 2] += cfg.hand_init_pos[2]
        if use_contact_style_init:
            ident_n = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0).expand(n_bad, -1)

            roll_lo, roll_hi = cfg.contact_init_roll_range_deg
            pitch_lo, pitch_hi = cfg.contact_init_pitch_range_deg
            yaw_lo, yaw_hi = cfg.contact_init_yaw_range_deg
            roll = torch.deg2rad(torch.rand(n_bad, device=device) * (roll_hi - roll_lo) + roll_lo)
            pitch = torch.deg2rad(torch.rand(n_bad, device=device) * (pitch_hi - pitch_lo) + pitch_lo)
            yaw = torch.deg2rad(torch.rand(n_bad, device=device) * (yaw_hi - yaw_lo) + yaw_lo)
            if use_bnc_contact_init:
                yaw = yaw + torch.randint(0, 2, (n_bad,), device=device).float() * torch.pi
            held_contact_quat = torch_utils.quat_mul(
                env.fixed_quat[bad_envs],
                torch_utils.quat_from_euler_xyz(roll, pitch, yaw),
            )

            female_point_local = torch.zeros((n_bad, 3), device=device)
            male_point_local = torch.zeros((n_bad, 3), device=device)
            if use_rj45_contact_init:
                x_lo, x_hi = cfg.female_rear_edge_x_range_local
                female_point_local[:, 0] = torch.rand(n_bad, device=device) * (x_hi - x_lo) + x_lo
                female_point_local[:, 1] = cfg.female_rear_edge_y_local
                female_point_local[:, 2] = cfg.female_rear_edge_z_local
                x_lo, x_hi = cfg.male_bottom_patch_x_range_local
                y_lo, y_hi = cfg.male_bottom_patch_y_range_local
                male_point_local[:, 0] = torch.rand(n_bad, device=device) * (x_hi - x_lo) + x_lo
                male_point_local[:, 1] = torch.rand(n_bad, device=device) * (y_hi - y_lo) + y_lo
                male_point_local[:, 2] = cfg.male_bottom_patch_z_local
            elif use_bnc_contact_init:
                fxl, fxh = cfg.bnc_contact_init_female_x_range
                fyl, fyh = cfg.bnc_contact_init_female_y_range
                female_point_local[:, 0] = torch.rand(n_bad, device=device) * (fxh - fxl) + fxl
                female_point_local[:, 1] = torch.rand(n_bad, device=device) * (fyh - fyl) + fyl
                female_point_local[:, 2] = cfg.bnc_contact_init_female_z_local
                mxl, mxh = cfg.bnc_contact_init_male_x_range
                myl, myh = cfg.bnc_contact_init_male_y_range
                male_point_local[:, 0] = torch.rand(n_bad, device=device) * (mxh - mxl) + mxl
                male_point_local[:, 1] = torch.rand(n_bad, device=device) * (myh - myl) + myl
                male_point_local[:, 2] = cfg.bnc_contact_init_male_z_local
            else:
                side = torch.randint(0, 2, (n_bad,), device=device)
                bx_lo, bx_hi = cfg.box_contact_rear_edge_x_range
                bx_mid = 0.5 * (bx_lo + bx_hi)
                x_lo_b = torch.where(
                    side == 0,
                    torch.full((n_bad,), bx_lo, device=device),
                    torch.full((n_bad,), bx_mid, device=device),
                )
                x_hi_b = torch.where(
                    side == 0,
                    torch.full((n_bad,), bx_mid, device=device),
                    torch.full((n_bad,), bx_hi, device=device),
                )
                female_point_local[:, 0] = (
                    torch.rand(n_bad, device=device) * (x_hi_b - x_lo_b) + x_lo_b
                )
                female_point_local[:, 1] = cfg.box_contact_rear_edge_y_local
                female_point_local[:, 2] = cfg.box_contact_rear_edge_z_local

                lx_lo, lx_hi = cfg.lid_contact_front_edge_x_range
                lx_mid = 0.5 * (lx_lo + lx_hi)
                x_lo_l = torch.where(
                    side == 0,
                    torch.full((n_bad,), lx_lo, device=device),
                    torch.full((n_bad,), lx_mid, device=device),
                )
                x_hi_l = torch.where(
                    side == 0,
                    torch.full((n_bad,), lx_mid, device=device),
                    torch.full((n_bad,), lx_hi, device=device),
                )
                male_point_local[:, 0] = (
                    torch.rand(n_bad, device=device) * (x_hi_l - x_lo_l) + x_lo_l
                )
                male_point_local[:, 1] = cfg.lid_contact_front_edge_y_local
                male_point_local[:, 2] = cfg.lid_contact_front_edge_z_local

            _, female_point_world = torch_utils.tf_combine(
                env.fixed_quat[bad_envs],
                env.fixed_pos[bad_envs],
                ident_n,
                female_point_local,
            )

            held_contact_pos = female_point_world - torch_utils.quat_rotate(held_contact_quat, male_point_local)
            fingertip_target_quat, fingertip_target_pos = torch_utils.tf_combine(
                held_contact_quat,
                held_contact_pos,
                held_to_fingertip_quat[bad_envs],
                held_to_fingertip_pos[bad_envs],
            )
            above_fixed_pos[bad_envs] = fingertip_target_pos
            hand_down_quat[bad_envs, :] = fingertip_target_quat
        else:
            rand_sample = torch.rand((n_bad, 3), dtype=torch.float32, device=device)
            above_fixed_pos_rand = 2 * (rand_sample - 0.5)
            hand_init_pos_rand = torch.tensor(cfg.hand_init_pos_noise, device=device)
            above_fixed_pos_rand = above_fixed_pos_rand @ torch.diag(hand_init_pos_rand)
            above_fixed_pos[bad_envs] += above_fixed_pos_rand

            if cfg.name == "box_lid_insert":
                xy_local = torch.zeros((num_envs, 3), device=device)
                xy_local[_is_near, 0] = cfg.hand_init_pos[0]
                xy_local[_is_near, 1] = cfg.hand_init_pos[1]
                _is_far = ~_is_near
                if _is_far.any():
                    x_lo, x_hi = cfg.hand_init_x_range
                    y_lo, y_hi = cfg.hand_init_y_range
                    z_lo, z_hi = cfg.hand_init_z_range
                    r = torch.rand((num_envs, 3), device=device)
                    xy_local[_is_far, 0] = r[_is_far, 0] * (x_hi - x_lo) + x_lo
                    xy_local[_is_far, 1] = r[_is_far, 1] * (y_hi - y_lo) + y_lo
                    above_fixed_pos[_is_far, 2] = (
                        fixed_tip_pos[_is_far, 2] + r[_is_far, 2] * (z_hi - z_lo) + z_lo
                    )
                identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0).repeat(num_envs, 1)
                _, xy_world = torch_utils.tf_combine(
                    env.fixed_quat,
                    torch.zeros((num_envs, 3), device=device),
                    identity,
                    xy_local,
                )
                above_fixed_pos += xy_world

            if cfg.name in ("rj45_insert", "bnc_insert"):
                far_mask = ~_is_near[bad_envs]
                if far_mask.any():
                    far_env_ids = bad_envs[far_mask]
                    n_far = int(far_mask.sum())
                    x_lo, x_hi = cfg.hand_init_x_range
                    y_lo, y_hi = cfg.hand_init_y_range
                    z_lo, z_hi = cfg.hand_init_z_range
                    r = torch.rand((n_far, 3), device=device)
                    xy_local_far = torch.zeros((n_far, 3), device=device)
                    xy_local_far[:, 0] = r[:, 0] * (x_hi - x_lo) + x_lo
                    xy_local_far[:, 1] = r[:, 1] * (y_hi - y_lo) + y_lo
                    ident_n = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0).expand(n_far, -1)
                    _, xy_world_far = torch_utils.tf_combine(
                        env.fixed_quat[far_env_ids],
                        torch.zeros((n_far, 3), device=device),
                        ident_n,
                        xy_local_far,
                    )
                    above_fixed_pos[far_env_ids, :2] = fixed_tip_pos[far_env_ids, :2] + xy_world_far[:, :2]
                    above_fixed_pos[far_env_ids, 2] = (
                        fixed_tip_pos[far_env_ids, 2] + r[:, 2] * (z_hi - z_lo) + z_lo
                    )

            hand_down_euler = torch.tensor(cfg.hand_init_orn, device=device).unsqueeze(0).repeat(n_bad, 1)

            rand_sample = torch.rand((n_bad, 3), dtype=torch.float32, device=device)
            above_fixed_orn_noise = 2 * (rand_sample - 0.5)
            hand_init_orn_rand = torch.tensor(cfg.hand_init_orn_noise, device=device)
            above_fixed_orn_noise = above_fixed_orn_noise @ torch.diag(hand_init_orn_rand)
            hand_down_euler += above_fixed_orn_noise

            if cfg.name == "box_lid_insert":
                _, _, box_yaw = torch_utils.get_euler_xyz(env.fixed_quat[bad_envs])
                hand_down_euler[:, 2] = box_yaw - 0.5 * torch.pi
                yaw_noise = (torch.rand(n_bad, device=device) * 2 - 1) * np.deg2rad(cfg.hand_init_yaw_noise_deg)
                pitch_noise = (torch.rand(n_bad, device=device) * 2 - 1) * np.deg2rad(cfg.hand_init_pitch_noise_deg)
                hand_down_euler[:, 2] += yaw_noise
                hand_down_euler[:, 1] += pitch_noise

            if cfg.name in ("rj45_insert", "bnc_insert"):
                _yaw_offset = np.deg2rad(getattr(cfg, "hand_init_yaw_offset_deg", 0.0))

                near_mask = _is_near[bad_envs]
                if near_mask.any():
                    _, _, sock_yaw_near = torch_utils.get_euler_xyz(env.fixed_quat[bad_envs[near_mask]])
                    hand_down_euler[near_mask, 2] = sock_yaw_near + _yaw_offset

                far_mask = ~_is_near[bad_envs]
                if far_mask.any():
                    n_far = int(far_mask.sum())
                    yaw_deg = getattr(cfg, "hand_init_yaw_noise_deg", 0.0)
                    _, _, sock_yaw = torch_utils.get_euler_xyz(env.fixed_quat[bad_envs[far_mask]])
                    yaw_noise = (torch.rand(n_far, device=device) * 2 - 1) * np.deg2rad(yaw_deg)
                    base_yaw = sock_yaw.clone() + _yaw_offset
                    if cfg.name == "bnc_insert":
                        flip = torch.randint(0, 2, (n_far,), device=device).float() * torch.pi
                        base_yaw = base_yaw + flip
                    hand_down_euler[far_mask, 2] = base_yaw + yaw_noise

            hand_down_quat[bad_envs, :] = torch_utils.quat_from_euler_xyz(
                roll=hand_down_euler[:, 0],
                pitch=hand_down_euler[:, 1],
                yaw=hand_down_euler[:, 2],
            )

        pos_error, aa_error = env.set_pos_inverse_kinematics_to_targets(
            above_fixed_pos,
            hand_down_quat,
            bad_envs,
        )
        pos_error = torch.linalg.norm(pos_error, dim=1) > 1e-3
        angle_error = torch.norm(aa_error, dim=1) > 1e-3
        any_error = torch.logical_or(pos_error, angle_error)
        bad_envs = bad_envs[any_error.nonzero(as_tuple=False).squeeze(-1)]

        if bad_envs.shape[0] == 0 or ik_attempt >= 100:
            break

        env._set_franka_to_default_pose(joints=env.cfg.ctrl.reset_joints, env_ids=bad_envs)
        ik_attempt += 1

    env.step_sim_no_action()

    flip_z_quat = torch.tensor([0.0, 0.0, 1.0, 0.0], device=device).unsqueeze(0).repeat(num_envs, 1)
    fingertip_flipped_quat, fingertip_flipped_pos = torch_utils.tf_combine(
        q1=env.fingertip_midpoint_quat,
        t1=env.fingertip_midpoint_pos,
        q2=flip_z_quat,
        t2=torch.zeros((num_envs, 3), device=device),
    )

    held_asset_relative_pos, held_asset_relative_quat = env.get_handheld_asset_relative_pose()
    asset_in_hand_quat, asset_in_hand_pos = torch_utils.tf_inverse(held_asset_relative_quat, held_asset_relative_pos)

    translated_held_asset_quat, translated_held_asset_pos = torch_utils.tf_combine(
        q1=fingertip_flipped_quat,
        t1=fingertip_flipped_pos,
        q2=asset_in_hand_quat,
        t2=asset_in_hand_pos,
    )

    rand_sample = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
    held_asset_pos_noise = 2 * rand_sample
    held_asset_pos_noise_level = torch.tensor(cfg.held_asset_pos_noise, device=device)
    held_asset_pos_noise = held_asset_pos_noise @ torch.diag(held_asset_pos_noise_level)
    translated_held_asset_quat, translated_held_asset_pos = torch_utils.tf_combine(
        q1=translated_held_asset_quat,
        t1=translated_held_asset_pos,
        q2=torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0).repeat(num_envs, 1),
        t2=held_asset_pos_noise,
    )

    held_state = env._held_asset.data.default_root_state.clone()
    held_state[:, 0:3] = translated_held_asset_pos + env.scene.env_origins
    held_state[:, 3:7] = translated_held_asset_quat
    held_state[:, 7:] = 0.0
    env._held_asset.write_root_pose_to_sim(held_state[:, 0:7])
    env._held_asset.write_root_velocity_to_sim(held_state[:, 7:])
    env._held_asset.reset()

    reset_task_prop_gains = torch.tensor(env.cfg.ctrl.reset_task_prop_gains, device=device).repeat((num_envs, 1))
    env._set_gains(reset_task_prop_gains, env.cfg.ctrl.reset_rot_deriv_scale)

    env.step_sim_no_action()

    grasp_time = 0.0
    while grasp_time < 0.25:
        env.ctrl_target_joint_pos[env_ids, 7:] = 0.0
        env.move_gripper_in_place(0.0)
        env.step_sim_no_action()
        grasp_time += env.sim.get_physics_dt()

    env.prev_joint_pos = env.joint_pos[:, 0:7].clone()
    env.prev_fingertip_pos = env.fingertip_midpoint_pos.clone()
    env.prev_fingertip_quat = env.fingertip_midpoint_quat.clone()

    env.actions = torch.zeros_like(env.actions)
    env.prev_actions = torch.zeros_like(env.actions)
    env.fixed_pos_action_frame[:] = env.fixed_pos_obs_frame + env.init_fixed_pos_obs_noise

    env.ee_angvel_fd[:, :] = 0.0
    env.ee_linvel_fd[:, :] = 0.0

    env._set_gains(env.default_gains)

    physics_sim_view.set_gravity(carb.Float3(*env.cfg.sim.gravity))


def reset_forge_keypoints_after_randomize(env, env_ids: torch.Tensor) -> None:
    """Sample / reset Forge keypoint buffers after ``randomize_initial_state_forge`` (matches Factory)."""
    cfg = env.cfg_task
    device = env.device

    if cfg.name == "rj45_insert":
        env.kp_rj45_female_local[env_ids, :, 2] = env.kp_rj45_female_z_init
        n = cfg.num_reset_kp
        x_lo, x_hi = cfg.male_bottom_patch_x_range_local
        y_lo, y_hi = cfg.male_bottom_patch_y_range_local
        z_m = cfg.male_bottom_patch_z_local
        z_f = cfg.female_rear_edge_z_local
        plug_dy = cfg.socket_target_y_local
        x_rand = torch.rand((len(env_ids), n), device=device) * (x_hi - x_lo) + x_lo
        y_rand = torch.rand((len(env_ids), n), device=device) * (y_hi - y_lo) + y_lo
        env.kp_rj45_male_local[env_ids, :, 0] = x_rand
        env.kp_rj45_male_local[env_ids, :, 1] = y_rand
        env.kp_rj45_male_local[env_ids, :, 2] = z_m
        env.kp_rj45_female_local[env_ids, :, 0] = x_rand
        env.kp_rj45_female_local[env_ids, :, 1] = y_rand + plug_dy
        env.kp_rj45_female_local[env_ids, :, 2] = z_f

    if cfg.name == "bnc_insert":
        nr = cfg.num_reset_kp
        env.kp_bnc_local[env_ids] = 0.0
        r = torch.rand((len(env_ids), nr), device=device)
        z_c = float(getattr(cfg, "bnc_kp_z_center", 0.055))
        z_hs = float(getattr(cfg, "bnc_kp_z_half_spread", 0.024))
        env.kp_bnc_local[env_ids, :nr, 2] = z_c + (r - 0.5) * (2.0 * z_hs)
        env.kp_bnc_fixed_local[env_ids] = env.kp_bnc_local[env_ids].clone()
        z_base = float(getattr(cfg, "bnc_kp_socket_z_init_extra", 0.0))
        z_hi = float(getattr(cfg, "bnc_kp_socket_z_above_engage_m", 0.0))
        z_bump = z_base + z_hi
        if z_bump != 0.0:
            env.kp_bnc_fixed_local[env_ids, :nr, 2] += z_bump

    if cfg.name == "box_lid_insert":
        nr = cfg.num_reset_extra_kp
        ns = cfg.num_success_extra_kp
        n_total = 2 + max(nr, ns)
        left_clip = torch.tensor([-0.025218, -0.0444, 0.0289], device=device)
        right_clip = torch.tensor([0.024250, -0.0444, 0.0289], device=device)

        for buf in (env.kp_lid_local, env.kp_box_local):
            buf[env_ids, 0] = left_clip.unsqueeze(0).expand(len(env_ids), -1)
            buf[env_ids, 1] = right_clip.unsqueeze(0).expand(len(env_ids), -1)

        if nr > 0:
            x_rand = torch.rand((len(env_ids), nr), device=device) * 0.1046 - 0.0523
            kp_front = torch.stack(
                [
                    x_rand,
                    torch.full_like(x_rand, -0.0444),
                    torch.full_like(x_rand, 0.0289),
                ],
                dim=-1,
            )
            for buf in (env.kp_lid_local, env.kp_box_local):
                buf[env_ids, 2 : 2 + nr] = kp_front

        if ns > nr:
            spare = ns - nr
            spare_kp = left_clip.unsqueeze(0).unsqueeze(0).expand(len(env_ids), spare, -1)
            for buf in (env.kp_lid_local, env.kp_box_local):
                buf[env_ids, 2 + nr : n_total] = spare_kp

        env.kp_box_y_target[env_ids] = env.kp_box_local[env_ids, :, 1].clone()
        env.kp_box_local[env_ids, :, 1] = cfg.kp_advance_y_start
