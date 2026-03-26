# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch

import isaacsim.core.utils.torch as torch_utils


def get_keypoint_offsets(num_keypoints, device):
    """Get uniformly-spaced keypoints along a line of unit length, centered at 0."""
    keypoint_offsets = torch.zeros((num_keypoints, 3), device=device)
    keypoint_offsets[:, -1] = torch.linspace(0.0, 1.0, num_keypoints, device=device) - 0.5
    return keypoint_offsets


def get_rj45_tip_keypoint_offsets(device):
    """5 keypoints spread across the RJ45 male connector face (held-base / tip frame).

    The held_base for rj45_insert tracks the connector TIP (sim_Z = -3.0 mm from
    the USD origin), so held_base frame has Z=0 at the physical tip face.

    Offsets below are derived from the male plug STL bounds after 0.001 m/mm scale.
    New STL origin is at the connector mating face (Z=0); tip is 3 mm below (Z=-3 mm):
        X ∈ [-0.01875, +0.01875]  (37.5 mm wide connector face)
        Y ∈ [-0.00503, +0.01318]  (18.2 mm deep)
        Z =  0.0                  (tip face plane, = held_base Z=0)

    The same offsets are applied to both the held asset (plug tip) and the target
    (socket opening, via target_held_base_quat/pos).  At the assembled state the two
    frames coincide → keypoint_dist → 0.

    Returns: (5, 3) tensor of XYZ offsets in the tip frame.
    """
    return torch.tensor(
        [
            [ 0.00000,  0.00408,  0.0],  # face centre
            [-0.01875,  0.01318,  0.0],  # left-front corner
            [ 0.01875,  0.01318,  0.0],  # right-front corner
            [-0.01875, -0.00503,  0.0],  # left-back corner
            [ 0.01875, -0.00503,  0.0],  # right-back corner
        ],
        dtype=torch.float32,
        device=device,
    )


def get_deriv_gains(prop_gains, rot_deriv_scale=1.0):
    """Set robot gains using critical damping."""
    deriv_gains = 2 * torch.sqrt(prop_gains)
    deriv_gains[:, 3:6] /= rot_deriv_scale
    return deriv_gains


def wrap_yaw(angle):
    """Ensure yaw stays within range."""
    return torch.where(angle > np.deg2rad(235), angle - 2 * np.pi, angle)


def set_friction(asset, value, num_envs):
    """Update material properties for a given asset."""
    materials = asset.root_physx_view.get_material_properties()
    materials[..., 0] = value  # Static friction.
    materials[..., 1] = value  # Dynamic friction.
    env_ids = torch.arange(num_envs, device="cpu")
    asset.root_physx_view.set_material_properties(materials, env_ids)


def set_body_inertias(robot, num_envs):
    """Note: this is to account for the asset_options.armature parameter in IGE."""
    inertias = robot.root_physx_view.get_inertias()
    offset = torch.zeros_like(inertias)
    offset[:, :, [0, 4, 8]] += 0.01
    new_inertias = inertias + offset
    robot.root_physx_view.set_inertias(new_inertias, torch.arange(num_envs))


def get_held_base_pos_local(task_name, fixed_asset_cfg, num_envs, device):
    """Get transform between asset default frame and geometric base frame."""
    held_base_x_offset = 0.0
    if task_name == "peg_insert":
        held_base_z_offset = 0.0
    elif task_name == "gear_mesh":
        gear_base_offset = fixed_asset_cfg.medium_gear_base_offset
        held_base_x_offset = gear_base_offset[0]
        held_base_z_offset = gear_base_offset[2]
    elif task_name == "nut_thread":
        held_base_z_offset = fixed_asset_cfg.base_height
    elif task_name == "box_lid_insert":
        # [CUSTOM] The lid's USD prim origin (0,0,0) does NOT coincide with the
        # geometric bottom of the lid body.  In the original STL the body starts
        # at Z_min = 18.8 mm, so after the 0.001 m/mm scale the bottom face is
        # 0.0188 m above the USD origin.
        #
        # Shifting by this offset transforms the tracked "held base" position from
        # the USD root frame to the true bottom contact face of the lid, which is
        # the surface that must reach the box top for a successful insertion.
        # This value is also stored in LidYellowCfg.base_height for reference.
        held_base_z_offset = 0.0188  # = LidYellowCfg.base_height
    elif task_name == "rj45_insert":
        # [CUSTOM] RJ45 male plug: the connector tip (insertion end) sits BELOW
        # the USD prim origin.  The new STL has its origin at the connector mating face,
        # placing the physical connector tip at sim_Z = -3.0 mm.
        # Shifting by -0.003 transforms the tracked reference from the USD root
        # to the physical connector tip — the face that must reach the socket opening.
        held_base_z_offset = -0.003  # = RJ45MaleCfg.base_height (negative → tip below origin)
    elif task_name == "bnc_insert":
        # [CUSTOM] BNC Small male plug: the connector tip (insertion end) sits ABOVE
        # the USD prim origin.  The STL has Z in [+36.235, +107.201] mm, so the origin
        # is 36.235 mm BELOW the tip.  Shifting by +0.036235 transforms the tracked
        # reference from the USD root to the physical connector tip.
        held_base_z_offset = 0.036235  # = BNCSmallMaleCfg.base_height (positive → tip above origin)
    else:
        raise NotImplementedError("Task not implemented")

    held_base_pos_local = torch.tensor([0.0, 0.0, 0.0], device=device).repeat((num_envs, 1))
    held_base_pos_local[:, 0] = held_base_x_offset
    held_base_pos_local[:, 2] = held_base_z_offset

    return held_base_pos_local


def get_held_base_pose(held_pos, held_quat, task_name, fixed_asset_cfg, num_envs, device):
    """Get current poses for keypoint and success computation."""
    held_base_pos_local = get_held_base_pos_local(task_name, fixed_asset_cfg, num_envs, device)
    held_base_quat_local = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0).repeat(num_envs, 1)

    held_base_quat, held_base_pos = torch_utils.tf_combine(
        held_quat, held_pos, held_base_quat_local, held_base_pos_local
    )
    return held_base_pos, held_base_quat


def get_target_held_base_pose(fixed_pos, fixed_quat, task_name, fixed_asset_cfg, num_envs, device):
    """Get target poses for keypoint and success computation."""
    fixed_success_pos_local = torch.zeros((num_envs, 3), device=device)
    if task_name == "peg_insert":
        fixed_success_pos_local[:, 2] = 0.0
    elif task_name == "gear_mesh":
        gear_base_offset = fixed_asset_cfg.medium_gear_base_offset
        fixed_success_pos_local[:, 0] = gear_base_offset[0]
        fixed_success_pos_local[:, 2] = gear_base_offset[2]
    elif task_name == "nut_thread":
        head_height = fixed_asset_cfg.base_height
        shank_length = fixed_asset_cfg.height
        thread_pitch = fixed_asset_cfg.thread_pitch
        fixed_success_pos_local[:, 2] = head_height + shank_length - thread_pitch * 1.5
    elif task_name == "box_lid_insert":
        # [CUSTOM] Snap-fit assembly geometry (values in metres, scale ×0.001):
        #
        # The lid slides INSIDE the box (not on top). When fully assembled the lid
        # top plate (STL Z=30 mm) is flush with the box top face (STL Z=30 mm), so
        # both USD origins share the same world Z.  The snap clips (lid Z=26.5-28.5
        # mm) engage in the front-wall pockets (box Z=20-29 mm) at that point.
        #
        # held_base tracks the lid BOTTOM face (0.0188 m above lid USD origin, see
        # get_held_base_pos_local).  In assembled state lid origin = box origin, so
        # lid bottom is at box-local Z = 0.0188 m (= LidYellowCfg.base_height).
        #
        # Previous value was fixed_asset_cfg.height = 0.030 m (box top), which fired
        # when the lid was resting ON TOP of the box, not inserted inside it.
        _LID_BASE_HEIGHT = 0.0188  # must match LidYellowCfg.base_height
        fixed_success_pos_local[:, 2] = _LID_BASE_HEIGHT
    elif task_name == "rj45_insert":
        # [CUSTOM] RJ45 insertion geometry (empirically calibrated, values in metres):
        #
        # Empirical finding: "full insertion" has plug origin at socket-local:
        #   Y = -0.006 m  (cavity centre is 6 mm in -Y from socket USD origin)
        #   Z = +0.017 m  (cavity entrance is 17 mm above socket USD origin)
        #
        # held_base tracks the connector TIP (3 mm below plug origin, held_base_z_offset=-0.003).
        # At full insertion: tip is at socket-local Y=-0.006 m, Z = 0.017 - 0.003 = +0.014 m.
        #
        # NOTE: keypoint reward for rj45_insert does NOT use this target — it uses
        # per-episode random body keypoints sampled at reset (see _reset_idx / _get_factory_rew_dict).
        # This value drives success detection and visualisation.
        fixed_success_pos_local[:, 1] = -0.006   # cavity centre Y offset
        fixed_success_pos_local[:, 2] =  0.014   # tip Z at full insertion
    elif task_name == "bnc_insert":
        # [CUSTOM] BNC Small insertion geometry (values in metres, scale ×0.001):
        #
        # Female socket (fixed): USD origin is 35 mm above the socket base.
        #   Socket opening is at female-local sim_Z = +25 mm = 0.025 m.
        #
        # Male plug (held): connector TIP is at male-local sim_Z = +36.235 mm (tracked
        #   as held_base via the +0.036235 offset in get_held_base_pos_local).
        #
        # Empirical full insertion (visualizer calibration): plug origin at socket_origin + Z = -41 mm.
        # Tip (held_base) at full insertion: -41 + 36.235 = -4.765 mm from socket origin.
        # This is the target for keypoint convergence (keypoint_dist → 0 at full insertion).
        fixed_success_pos_local[:, 2] = -0.004765  # tip Z at full insertion (empirical)
    else:
        raise NotImplementedError("Task not implemented")
    fixed_success_quat_local = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0).repeat(num_envs, 1)

    target_held_base_quat, target_held_base_pos = torch_utils.tf_combine(
        fixed_quat, fixed_pos, fixed_success_quat_local, fixed_success_pos_local
    )
    return target_held_base_pos, target_held_base_quat


def squashing_fn(x, a, b):
    """Compute bounded reward function."""
    return 1 / (torch.exp(a * x) + b + torch.exp(-a * x))


def collapse_obs_dict(obs_dict, obs_order):
    """Stack observations in given order."""
    obs_tensors = [obs_dict[obs_name] for obs_name in obs_order]
    obs_tensors = torch.cat(obs_tensors, dim=-1)
    return obs_tensors
