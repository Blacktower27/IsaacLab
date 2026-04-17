# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Forge-style success checks for AutoMate assembly tasks (Franka).

Mirrors ``factory_env.FactoryEnv._get_curr_successes`` for ``box_lid_insert``,
``rj45_insert``, and ``bnc_insert``. For ``rj45_insert`` / ``bnc_insert``:

- ``success_threshold > 1.0`` → ENGAGE branch (XY + yaw + tilt + loose Z), same as
  Factory when using ``engage_threshold``-style values.
- ``success_threshold <= 1.0`` (e.g. Forge ``0.027``) → SUCCESS branch
  (``height_threshold = fixed_cfg.height + success_threshold``, tight XY).

Pass ``cfg_task.success_threshold`` through unchanged; do **not** coerce it above 1.0
unless you intentionally want the ENGAGE branch.
"""

from __future__ import annotations

import numpy as np
import torch

import isaacsim.core.utils.torch as torch_utils
from isaaclab.utils.math import quat_apply_inverse

FORGE_TASK_NAMES = ("box_lid_insert", "rj45_insert", "bnc_insert")


def get_curr_successes_forge(
    cfg_task,
    held_pos: torch.Tensor,
    held_quat: torch.Tensor,
    fixed_pos: torch.Tensor,
    fixed_quat: torch.Tensor,
    held_base_pos: torch.Tensor,
    ep_succeeded: torch.Tensor,
    num_envs: int,
    device: torch.device,
    success_threshold: float,
) -> torch.Tensor:
    """Return per-env success bool tensor (same semantics as Factory Forge)."""
    curr_successes = torch.zeros((num_envs,), dtype=torch.bool, device=device)
    fixed_cfg = cfg_task.fixed_asset_cfg

    if cfg_task.name == "box_lid_insert":
        _LEFT_HOLE_X = -0.025218
        _RIGHT_HOLE_X = 0.024250

        ident_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0).expand(num_envs, -1)

        left_clip_local = torch.zeros((num_envs, 3), device=device)
        left_clip_local[:, 0] = -0.025218
        left_clip_local[:, 1] = -0.0444
        left_clip_local[:, 2] = 0.0289

        right_clip_local = torch.zeros((num_envs, 3), device=device)
        right_clip_local[:, 0] = 0.024250
        right_clip_local[:, 1] = -0.0444
        right_clip_local[:, 2] = 0.0289

        _, left_clip_w = torch_utils.tf_combine(held_quat, held_pos, ident_q, left_clip_local)
        _, right_clip_w = torch_utils.tf_combine(held_quat, held_pos, ident_q, right_clip_local)
        left_clip_box = quat_apply_inverse(fixed_quat, left_clip_w - fixed_pos)
        right_clip_box = quat_apply_inverse(fixed_quat, right_clip_w - fixed_pos)

        if success_threshold >= 0.5:
            _X_TOL = 0.002
            _Y_MIN, _Y_MAX = -0.0495, 0.035
            _Z_MIN, _Z_MAX = 0.018, 0.030

            left_x_ok = (left_clip_box[:, 0] - _LEFT_HOLE_X).abs() < _X_TOL
            right_x_ok = (right_clip_box[:, 0] - _RIGHT_HOLE_X).abs() < _X_TOL
            left_y_ok = (left_clip_box[:, 1] > _Y_MIN) & (left_clip_box[:, 1] < _Y_MAX)
            right_y_ok = (right_clip_box[:, 1] > _Y_MIN) & (right_clip_box[:, 1] < _Y_MAX)
            left_z_ok = (left_clip_box[:, 2] > _Z_MIN) & (left_clip_box[:, 2] < _Z_MAX)
            right_z_ok = (right_clip_box[:, 2] > _Z_MIN) & (right_clip_box[:, 2] < _Z_MAX)
        else:
            _X_TOL = 0.002
            _Y_TOL = 0.004
            _HOLE_WALL_Y = -0.0444
            _Z_MIN, _Z_MAX = 0.021, 0.029

            left_x_ok = (left_clip_box[:, 0] - _LEFT_HOLE_X).abs() < _X_TOL
            right_x_ok = (right_clip_box[:, 0] - _RIGHT_HOLE_X).abs() < _X_TOL
            left_y_ok = (left_clip_box[:, 1] - _HOLE_WALL_Y).abs() < _Y_TOL
            right_y_ok = (right_clip_box[:, 1] - _HOLE_WALL_Y).abs() < _Y_TOL
            left_z_ok = (left_clip_box[:, 2] > _Z_MIN) & (left_clip_box[:, 2] < _Z_MAX)
            right_z_ok = (right_clip_box[:, 2] > _Z_MIN) & (right_clip_box[:, 2] < _Z_MAX)

        curr_successes = left_x_ok & left_y_ok & left_z_ok & right_x_ok & right_y_ok & right_z_ok

    elif cfg_task.name == "rj45_insert":
        ident_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0).expand(num_envs, -1)
        socket_opening_local = torch.zeros((num_envs, 3), device=device)
        socket_opening_local[:, 1] = cfg_task.socket_target_y_local
        socket_opening_local[:, 2] = cfg_task.socket_target_z_local
        _, socket_opening_world = torch_utils.tf_combine(fixed_quat, fixed_pos, ident_q, socket_opening_local)
        z_disp = held_base_pos[:, 2] - socket_opening_world[:, 2]
        xy_dist = torch.linalg.vector_norm(socket_opening_world[:, 0:2] - held_base_pos[:, 0:2], dim=1)

        if success_threshold > 1.0:
            _XY_TOL = 0.004
            is_xy = xy_dist < _XY_TOL

            _, _, plug_yaw = torch_utils.get_euler_xyz(held_quat)
            _, _, sock_yaw = torch_utils.get_euler_xyz(fixed_quat)
            yaw_diff = (plug_yaw - sock_yaw + torch.pi) % (2 * torch.pi) - torch.pi
            is_yaw = yaw_diff.abs() < 0.262

            plug_z_local = torch.zeros((num_envs, 3), device=device)
            plug_z_local[:, 2] = -1.0
            plug_z_world = torch_utils.quat_rotate(held_quat, plug_z_local)
            cos_tilt = -plug_z_world[:, 2]
            is_tilt = cos_tilt > 0.966

            is_z = z_disp < 0.06

            curr_successes = is_xy & is_yaw & is_tilt & is_z
        else:
            _XY_STRICT = 0.004
            height_threshold = fixed_cfg.height + success_threshold
            is_inside = z_disp < height_threshold
            is_xy_strict = xy_dist < _XY_STRICT
            curr_successes = is_inside & is_xy_strict

    elif cfg_task.name == "bnc_insert":
        ident_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0).expand(num_envs, -1)
        socket_opening_local = torch.zeros((num_envs, 3), device=device)
        socket_opening_local[:, 2] = fixed_cfg.height
        _, socket_opening_world = torch_utils.tf_combine(fixed_quat, fixed_pos, ident_q, socket_opening_local)
        z_disp = held_base_pos[:, 2] - socket_opening_world[:, 2]
        xy_dist = torch.linalg.vector_norm(socket_opening_world[:, 0:2] - held_base_pos[:, 0:2], dim=1)

        _, _, plug_yaw = torch_utils.get_euler_xyz(held_quat)
        _, _, sock_yaw = torch_utils.get_euler_xyz(fixed_quat)
        yaw_diff_raw = (plug_yaw - sock_yaw + torch.pi) % (2 * torch.pi) - torch.pi
        yaw_diff_sym = torch.minimum(yaw_diff_raw.abs(), torch.pi - yaw_diff_raw.abs())
        _YAW_TOL = 0.175
        is_yaw = yaw_diff_sym < _YAW_TOL
        already_succeeded = ep_succeeded > 0

        if success_threshold > 1.0:
            is_xy = xy_dist < 0.004

            plug_z_local = torch.zeros((num_envs, 3), device=device)
            plug_z_local[:, 2] = -1.0
            plug_z_world = torch_utils.quat_rotate(held_quat, plug_z_local)
            is_tilt = -plug_z_world[:, 2] > 0.966

            is_z = z_disp < 0.040

            curr_successes = is_xy & (is_yaw | already_succeeded) & is_tilt & is_z
        else:
            height_threshold = fixed_cfg.height * success_threshold
            is_inside = z_disp < height_threshold
            is_xy_strict = xy_dist < 0.003
            curr_successes = is_inside & is_xy_strict & (is_yaw | already_succeeded)

    return curr_successes
