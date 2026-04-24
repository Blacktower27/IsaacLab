"""Collect box–lid insertion reference trajectories (Forge / Automate formats).

**Success gate** still matches ``FactoryEnv._get_curr_successes`` when
``success_threshold < 0.5`` (clip tooth in **box local**: X/Y/Z tolerances vs nominal
pocket wall). **Damped snap solver goal** uses ``ForgeBoxLidInsert.kp_box_clip_ease_end_*``
when set (same box-frame “achieved” clip XYZ as training keypoint ease end); otherwise
legacy pocket-centre + mid-Z targets.

**Trajectory (three segments):**

1. **Hold** at env contact (``n_hold``).
2. **Keypoint align:** move in **world position** (orientation fixed to contact) toward a
   least-squares root pose so lid keypoints match **reset** ``kp_box_local`` (Factory’s
   eased Y = ``kp_advance_y_start``). Uses ``step_sim_no_action`` so ``kp_box_local`` is
   not advanced early (reward path would move targets once ``keypoint_dist`` is small).
3. **kp_align + insert:** ``quat_final`` = contact **roll/pitch in box frame** + success **yaw
   in box frame**, recomposed as ``held = fixed * euler(r_c,p_c,y_s)`` — **not** world-Euler
   mixing (which misaligns yaw). **hold** uses raw ``held_quat0`` (untouched). **Insert** is
   translation only at fixed ``quat_final``. Uses ``env.step`` for progressive keypoints.

Optional **tail clip**: if ``--truncate_max_tip_step_m > 0``, each demo is truncated before the
first pair of frames whose ``held_tip_local`` spacing exceeds that distance (removes end spikes).

Per-env recording **stops after the first timestep** where ``FactoryEnv._get_curr_successes``
is true (same gate as training success / success reward). Logged tip / fingertip poses use
the **commanded** root pose written this step (``held_pos_w`` / ``held_quat_w``), not
``data.root_*`` readback, so penetration / solver correction at full seat does not produce
spurious “fly-away” in the JSON.

**Arm:** RJ45-style — only ``held_asset`` teleported; EE columns = kinematic grasp inverse.

Usage
-----
    ./isaaclab.sh -p scripts/collect_box_lid_trajectories.py \\
        --num_envs 32 --num_trajectories 32 --output scripts/box_lid_ref_traj.json \\
        --n_hold 5 --n_kp_align 50 --n_move 80
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect box-lid insertion reference trajectories.")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--num_trajectories", type=int, default=200)
parser.add_argument("--output", type=str, default="scripts/box_lid_ref_traj.json")
parser.add_argument(
    "--output_automate",
    type=str,
    default=None,
    help="Automate DTW JSON path. Default: <output> stem + _automate + ext.",
)
parser.add_argument(
    "--skip_automate_output",
    action="store_true",
    help="If set, only write the Forge JSON (--output).",
)
parser.add_argument("--n_hold", type=int, default=5, help="Hold at contact init (stationary).")
parser.add_argument(
    "--n_kp_align",
    type=int,
    default=50,
    help="Steps: interp in position from contact to LSQ pose matching initial kp_box_local "
    "(after reset, Y=kp_advance_y_start). Orientation stays contact. Uses step_sim_no_action. "
    "Use 0 to skip (phase 2 starts from contact).",
)
parser.add_argument(
    "--n_move",
    type=int,
    default=80,
    help="Steps: interp toward snap-fit position; orientation contact R/P + success yaw; uses env.step.",
)
parser.add_argument(
    "--success_solve_iters",
    type=int,
    default=150,
    help="Max iterations per env to adjust held root (env frame) for snap-fit clip gates.",
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Forge-BoxLidInsert-Direct-v0",
    help="Must be a Forge box task with a separate held_asset (Franka).",
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--truncate_max_tip_step_m",
    type=float,
    default=0.01,
    help="After collection, drop tail if consecutive held_tip_local step exceeds this (m). "
    "Typical smooth segment <~2 mm/step; spikes ~100+ mm indicate fly-away. Use 0 to disable.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import json
import os

import gymnasium as gym
import numpy as np
import torch

import isaacsim.core.utils.torch as torch_utils
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse, quat_conjugate

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.factory import factory_utils
from isaaclab_tasks.utils import parse_env_cfg

_TIP_OFFSET_LOCAL = np.array([0.0, 0.0, -0.003], dtype=np.float32)

# Snap-fit success (FactoryEnv box_lid_insert, success_threshold < 0.5)
_SNAP_LEFT_X = -0.025218
_SNAP_RIGHT_X = 0.024250
_SNAP_X_TOL = 0.002
_SNAP_Y_TOL = 0.004
_SNAP_WALL_Y = -0.0444
_SNAP_Z_MIN = 0.021
_SNAP_Z_MAX = 0.029
# Clip reference (lid local, metres) — same as factory_env
_CLIP_LEFT = (-0.025218, -0.0444, 0.0289)
_CLIP_RIGHT = (0.024250, -0.0444, 0.0289)
# Nominal seated clip targets in box frame (centre of feasible region)
def _snap_targets_legacy(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Fallback solver targets: pocket X, nominal wall Y, mid-Z (metres, box frame)."""
    z_mid = 0.5 * (_SNAP_Z_MIN + _SNAP_Z_MAX)
    tl = torch.tensor([_SNAP_LEFT_X, _SNAP_WALL_Y, z_mid], device=device, dtype=torch.float32)
    tr = torch.tensor([_SNAP_RIGHT_X, _SNAP_WALL_Y, z_mid], device=device, dtype=torch.float32)
    return tl, tr


def _snap_goal_box_from_task(task_cfg, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Box-frame clip targets for damped positioning: task cfg or legacy."""
    ease_l = getattr(task_cfg, "kp_box_clip_ease_end_left", None)
    ease_r = getattr(task_cfg, "kp_box_clip_ease_end_right", None)
    if ease_l is not None and ease_r is not None and len(ease_l) == 3 and len(ease_r) == 3:
        tl = torch.tensor(ease_l, device=device, dtype=torch.float32)
        tr = torch.tensor(ease_r, device=device, dtype=torch.float32)
        return tl, tr
    return _snap_targets_legacy(device)


def _world_quat_contact_rp_success_yaw(
    fixed_quat: torch.Tensor,
    quat_contact_w: torch.Tensor,
    quat_success_w: torch.Tensor,
) -> torch.Tensor:
    """World wxyz: **box-relative** roll/pitch from contact, **box-relative** yaw from success.

    ``rel = conj(fixed) * held`` so ``held = fixed * rel``. Lock ``rel``'s Euler XYZ roll/pitch
    from contact and yaw from nominal success — **not** world-Euler splicing (which is wrong).
    Shapes: ``(N, 4)`` for all inputs; returns ``(N, 4)``.
    """
    rel_c = torch_utils.quat_mul(quat_conjugate(fixed_quat), quat_contact_w)
    rel_s = torch_utils.quat_mul(quat_conjugate(fixed_quat), quat_success_w)
    roll0, pitch0, _yc = torch_utils.get_euler_xyz(rel_c)
    _r1, _p1, yaw_s = torch_utils.get_euler_xyz(rel_s)
    rel_f = torch_utils.quat_from_euler_xyz(roll0, pitch0, yaw_s)
    return torch_utils.quat_mul(fixed_quat, rel_f)


def _clips_box_local(
    held_pos_env: torch.Tensor,
    held_quat: torch.Tensor,
    fixed_pos_env: torch.Tensor,
    fixed_quat: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Left/right clip points in box root frame (same geometry as reward)."""
    left_local = torch.tensor(_CLIP_LEFT, device=device, dtype=torch.float32)
    right_local = torch.tensor(_CLIP_RIGHT, device=device, dtype=torch.float32)
    lw = torch_utils.quat_rotate(held_quat.unsqueeze(0), left_local.unsqueeze(0)).squeeze(0) + held_pos_env
    rw = torch_utils.quat_rotate(held_quat.unsqueeze(0), right_local.unsqueeze(0)).squeeze(0) + held_pos_env
    lb = quat_apply_inverse(fixed_quat.unsqueeze(0), (lw - fixed_pos_env).unsqueeze(0)).squeeze(0)
    rb = quat_apply_inverse(fixed_quat.unsqueeze(0), (rw - fixed_pos_env).unsqueeze(0)).squeeze(0)
    return lb, rb


def _snap_fit_success_ok(lb: torch.Tensor, rb: torch.Tensor) -> bool:
    """True iff same gates as FactoryEnv (snap seated branch)."""
    lx = (lb[0] - _SNAP_LEFT_X).abs() < _SNAP_X_TOL
    rx = (rb[0] - _SNAP_RIGHT_X).abs() < _SNAP_X_TOL
    ly = (lb[1] - _SNAP_WALL_Y).abs() < _SNAP_Y_TOL
    ry = (rb[1] - _SNAP_WALL_Y).abs() < _SNAP_Y_TOL
    lz = (lb[2] > _SNAP_Z_MIN) & (lb[2] < _SNAP_Z_MAX)
    rz = (rb[2] > _SNAP_Z_MIN) & (rb[2] < _SNAP_Z_MAX)
    return bool((lx & rx & ly & ry & lz & rz).item())


def _min_success_t_on_segment_env(
    pos_a: torch.Tensor,
    pos_b: torch.Tensor,
    held_quat: torch.Tensor,
    fixed_pos: torch.Tensor,
    fixed_quat: torch.Tensor,
    device: torch.device,
    n_iter: int = 14,
) -> float:
    """Smallest ``t∈[0,1]`` with ``pos=(1-t)*pos_a+t*pos_b`` satisfying snap gates (fixed quat)."""

    def ok(p: torch.Tensor) -> bool:
        lb, rb = _clips_box_local(p, held_quat, fixed_pos, fixed_quat, device)
        return _snap_fit_success_ok(lb, rb)

    if ok(pos_a):
        return 0.0
    if not ok(pos_b):
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        p = (1.0 - mid) * pos_a + mid * pos_b
        if ok(p):
            hi = mid
        else:
            lo = mid
    return hi


def _solve_held_pos_keypoint_align_lsq(
    fixed_pos_env: torch.Tensor,
    fixed_quat: torch.Tensor,
    held_quat: torch.Tensor,
    kp_lid: torch.Tensor,
    kp_box: torch.Tensor,
) -> torch.Tensor:
    """Held root (env frame) minimizing mean squared keypoint error at fixed orientation.

    Matches FactoryEnv ``_get_factory_rew_dict`` for ``box_lid_insert``:
    ``kp_held = R_h * kp_lid + held_pos``, ``kp_fixed = R_f * kp_box + fixed_pos``.
    LSQ closed form: ``held_pos = mean(kp_fixed - R_h * kp_lid)``.
    """
    n = kp_lid.shape[0]
    r = torch_utils.quat_rotate(held_quat.unsqueeze(0).expand(n, -1), kp_lid)
    f = torch_utils.quat_rotate(fixed_quat.unsqueeze(0).expand(n, -1), kp_box) + fixed_pos_env.unsqueeze(0)
    return (f - r).mean(dim=0)


def _solve_held_pos_env_snap_fit(
    fixed_pos_env: torch.Tensor,
    fixed_quat: torch.Tensor,
    held_quat: torch.Tensor,
    pos_init_env: torch.Tensor,
    device: torch.device,
    max_iters: int,
    snap_tgt_l: torch.Tensor,
    snap_tgt_r: torch.Tensor,
    step_scale: float = 0.35,
) -> torch.Tensor:
    """Adjust held root (env) so clip gates pass; ``held_quat`` fixed.

    ``snap_tgt_l`` / ``snap_tgt_r`` are desired clip positions in **box local** (metres),
    typically from ``kp_box_clip_ease_end_*``; iteration stops when ``_snap_fit_success_ok``.
    """
    pos = pos_init_env.clone()
    tgt_l, tgt_r = snap_tgt_l, snap_tgt_r
    for _ in range(max_iters):
        lb, rb = _clips_box_local(pos, held_quat, fixed_pos_env, fixed_quat, device)
        if _snap_fit_success_ok(lb, rb):
            break
        err = 0.5 * ((tgt_l - lb) + (tgt_r - rb))
        d_env = torch_utils.quat_rotate(fixed_quat.unsqueeze(0), err.unsqueeze(0)).squeeze(0)
        pos = pos + step_scale * d_env
    return pos


def _fingertip_target_from_held_world(
    held_pos_w: torch.Tensor,
    held_quat_w: torch.Tensor,
    held_rel_pos: torch.Tensor,
    held_rel_quat: torch.Tensor,
    env_origins: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_envs = held_pos_w.shape[0]
    ft_flipped_q, ft_flipped_p = torch_utils.tf_combine(
        held_quat_w,
        held_pos_w,
        held_rel_quat,
        held_rel_pos,
    )
    flip_z = torch.tensor([0.0, 0.0, 1.0, 0.0], device=held_pos_w.device, dtype=torch.float32).unsqueeze(0)
    flip_z = flip_z.repeat(num_envs, 1)
    zero = torch.zeros_like(held_pos_w)
    flip_inv_q, flip_inv_p = torch_utils.tf_inverse(flip_z, zero)
    ft_q, ft_p = torch_utils.tf_combine(ft_flipped_q, ft_flipped_p, flip_inv_q, flip_inv_p)
    return ft_p - env_origins, ft_q


def _fingertip_pose7_env_from_held(
    held_pos_w: torch.Tensor,
    held_quat_w: torch.Tensor,
    held_rel_pos: torch.Tensor,
    held_rel_quat: torch.Tensor,
    env_origins: torch.Tensor,
) -> torch.Tensor:
    pos_env, ft_q = _fingertip_target_from_held_world(
        held_pos_w, held_quat_w, held_rel_pos, held_rel_quat, env_origins
    )
    return torch.cat([pos_env, ft_q], dim=-1)


def _held_root_pose_at_success(
    fixed_pos: torch.Tensor,
    fixed_quat: torch.Tensor,
    task_cfg,
    num_envs: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_base_pos, target_base_quat = factory_utils.get_target_held_base_pose(
        fixed_pos,
        fixed_quat,
        "box_lid_insert",
        task_cfg.fixed_asset_cfg,
        num_envs,
        device,
        task_cfg=task_cfg,
    )
    local = factory_utils.get_held_base_pos_local("box_lid_insert", task_cfg.fixed_asset_cfg, num_envs, device)
    held_quat_end = target_base_quat
    held_pos_end = target_base_pos - torch_utils.quat_rotate(held_quat_end, local)
    return held_pos_end, held_quat_end


def _smoothstep_01(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def _truncate_parallel_on_tip_jump(
    tip_local: list,
    rpy_local: list,
    pose7_local: list,
    ft_pos: list,
    ft_pose: list,
    max_step_m: float,
) -> tuple[list, list, list, list, list, int, int]:
    """Keep prefix ending at last frame before a tip step > ``max_step_m``. Returns new lists + lengths."""
    n0 = len(tip_local)
    if max_step_m <= 0.0 or n0 < 2:
        return tip_local, rpy_local, pose7_local, ft_pos, ft_pose, n0, 0
    arr = np.asarray(tip_local, dtype=np.float64)
    d = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    bad = np.nonzero(d > max_step_m)[0]
    if bad.size == 0:
        return tip_local, rpy_local, pose7_local, ft_pos, ft_pose, n0, 0
    keep = int(bad[0]) + 1
    return (
        tip_local[:keep],
        rpy_local[:keep],
        pose7_local[:keep],
        ft_pos[:keep],
        ft_pose[:keep],
        keep,
        n0 - keep,
    )


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=True,
    )
    env_cfg.seed = args_cli.seed
    env_cfg.episode_length_s = 120.0

    if getattr(env_cfg.task, "init_mode", "") != "contact":
        print(
            f"[collect] WARNING: task init_mode={getattr(env_cfg.task, 'init_mode', None)!r} "
            f"— expected 'contact' for contact-init trajectories."
        )

    env = gym.make(args_cli.task, cfg=env_cfg)
    inner = env.unwrapped
    device = inner.device
    num_envs = inner.num_envs

    if not hasattr(inner, "_held_asset"):
        env.close()
        raise RuntimeError(
            "This script teleports a separate held articulation. "
            "Use Isaac-Forge-BoxLidInsert-Direct-v0 (Franka), not the Kuka embedded-lid variant."
        )
    if not hasattr(inner, "kp_lid_local") or not hasattr(inner, "kp_box_local"):
        env.close()
        raise RuntimeError("Expected Factory box_lid_insert buffers kp_lid_local / kp_box_local on env.")

    snap_goal_l, snap_goal_r = _snap_goal_box_from_task(env_cfg.task, device)
    use_cfg_snap = getattr(env_cfg.task, "kp_box_clip_ease_end_left", None) is not None and getattr(
        env_cfg.task, "kp_box_clip_ease_end_right", None
    ) is not None

    frozen_jpos = inner._robot.data.default_joint_pos.clone()
    frozen_jpos[:, :7] = torch.tensor(inner.cfg.ctrl.reset_joints, device=device).unsqueeze(0).expand(num_envs, -1)
    frozen_jvel = torch.zeros_like(frozen_jpos)

    n_steps = args_cli.n_hold + args_cli.n_kp_align + args_cli.n_move
    print(
        f"\n{'=' * 60}\n"
        f"[collect] hold=raw contact quat; then box-frame R/P from contact + yaw from success; shallow snap\n"
        f"[collect] snap solver goal: {'kp_box_clip_ease_end_* (task cfg)' if use_cfg_snap else 'legacy pocket mid-Z'}\n"
        f"[collect] steps≤{n_steps} (early stop when Factory success); hold={args_cli.n_hold}, "
        f"kp_align={args_cli.n_kp_align}, move={args_cli.n_move}; snap_solve_iters={args_cli.success_solve_iters}\n"
        f"{'=' * 60}\n"
    )

    collected: list[dict] = []
    collected_automate: list[dict] = []
    traj_idx = 0

    while traj_idx < args_cli.num_trajectories and simulation_app.is_running():
        batch = min(num_envs, args_cli.num_trajectories - traj_idx)
        env.reset()

        inner._robot.write_joint_state_to_sim(frozen_jpos, frozen_jvel)
        inner._robot.set_joint_position_target(frozen_jpos)
        inner._robot.set_joint_effort_target(frozen_jvel)
        inner.step_sim_no_action()

        held_pos_w0 = inner._held_asset.data.root_pos_w.clone()
        held_quat0 = inner._held_asset.data.root_quat_w.clone()

        held_pos_end, held_quat_end = _held_root_pose_at_success(
            inner.fixed_pos.clone(),
            inner.fixed_quat.clone(),
            env_cfg.task,
            num_envs,
            device,
        )

        quat_final = _world_quat_contact_rp_success_yaw(
            inner.fixed_quat.clone(),
            held_quat0.clone(),
            held_quat_end.clone(),
        )

        held_pos_deep_env = torch.zeros_like(held_pos_end)
        for ei in range(batch):
            held_pos_deep_env[ei] = _solve_held_pos_env_snap_fit(
                inner.fixed_pos[ei].clone(),
                inner.fixed_quat[ei].clone(),
                quat_final[ei].clone(),
                held_pos_end[ei].clone(),
                device,
                max_iters=args_cli.success_solve_iters,
                snap_tgt_l=snap_goal_l,
                snap_tgt_r=snap_goal_r,
            )
            lb, rb = _clips_box_local(
                held_pos_deep_env[ei],
                quat_final[ei],
                inner.fixed_pos[ei],
                inner.fixed_quat[ei],
                device,
            )
            if not _snap_fit_success_ok(lb, rb):
                print(
                    f"[collect] WARNING env {ei}: deep snap-fit did not satisfy clip gates "
                    f"(contact tilt may be incompatible); using last iterate anyway."
                )
        for ei in range(batch, num_envs):
            held_pos_deep_env[ei] = held_pos_end[ei]

        held_pos_align_env = torch.zeros_like(held_pos_end)
        if args_cli.n_kp_align > 0:
            for ei in range(batch):
                held_pos_align_env[ei] = _solve_held_pos_keypoint_align_lsq(
                    inner.fixed_pos[ei].clone(),
                    inner.fixed_quat[ei].clone(),
                    quat_final[ei].clone(),
                    inner.kp_lid_local[ei].clone(),
                    inner.kp_box_local[ei].clone(),
                )
        else:
            held_pos_align_env[:batch] = held_pos_w0[:batch] - inner.scene.env_origins[:batch]
        for ei in range(batch, num_envs):
            held_pos_align_env[ei] = held_pos_w0[ei] - inner.scene.env_origins[ei]
        held_pos_w_align = held_pos_align_env + inner.scene.env_origins

        held_pos_shallow_env = torch.zeros_like(held_pos_deep_env)
        for ei in range(batch):
            t_press = _min_success_t_on_segment_env(
                held_pos_align_env[ei].clone(),
                held_pos_deep_env[ei].clone(),
                quat_final[ei].clone(),
                inner.fixed_pos[ei].clone(),
                inner.fixed_quat[ei].clone(),
                device,
            )
            held_pos_shallow_env[ei] = (1.0 - t_press) * held_pos_align_env[ei] + t_press * held_pos_deep_env[ei]
        for ei in range(batch, num_envs):
            held_pos_shallow_env[ei] = held_pos_deep_env[ei]
        held_pos_w_shallow = held_pos_shallow_env + inner.scene.env_origins

        env_tip_local = [[] for _ in range(batch)]
        env_rpy_local = [[] for _ in range(batch)]
        env_pose7_local = [[] for _ in range(batch)]
        env_ft_automate = [[] for _ in range(batch)]
        env_ft_pose_automate = [[] for _ in range(batch)]

        traj_done = torch.zeros(num_envs, dtype=torch.bool, device=device)
        frozen_pos_w = held_pos_w0.clone()
        frozen_quat_w = held_quat0.clone()
        succ_check_rot = inner.cfg_task.name == "nut_thread"
        succ_threshold = inner.cfg_task.success_threshold

        for si in range(n_steps):
            held_pos_w = torch.zeros_like(held_pos_w0)
            held_quat_w = torch.zeros_like(held_quat0)

            if si < args_cli.n_hold:
                t_align = 0.0
                t_ins = 0.0
                last_ins = args_cli.n_move <= 1
            elif si < args_cli.n_hold + args_cli.n_kp_align:
                if args_cli.n_kp_align <= 1:
                    t_align = 1.0
                else:
                    im_a = si - args_cli.n_hold
                    t_align = _smoothstep_01(im_a / (args_cli.n_kp_align - 1))
                t_ins = 0.0
                last_ins = args_cli.n_move <= 1
            else:
                t_align = 1.0
                im_ins = si - args_cli.n_hold - args_cli.n_kp_align
                if args_cli.n_move <= 1:
                    t_ins = 1.0
                    last_ins = True
                else:
                    t_ins = _smoothstep_01(im_ins / (args_cli.n_move - 1))
                    last_ins = im_ins >= args_cli.n_move - 1

            for ei in range(num_envs):
                if traj_done[ei]:
                    held_pos_w[ei] = frozen_pos_w[ei]
                    held_quat_w[ei] = frozen_quat_w[ei]
                elif ei >= batch:
                    held_pos_w[ei] = held_pos_w0[ei]
                    held_quat_w[ei] = held_quat0[ei]
                elif si < args_cli.n_hold:
                    held_pos_w[ei] = held_pos_w0[ei]
                    held_quat_w[ei] = held_quat0[ei]
                elif si < args_cli.n_hold + args_cli.n_kp_align:
                    held_pos_w[ei] = (1.0 - t_align) * held_pos_w0[ei] + t_align * held_pos_w_align[ei]
                    held_quat_w[ei] = quat_final[ei]
                else:
                    t_m = t_ins if not last_ins else 1.0
                    held_pos_w[ei] = (1.0 - t_m) * held_pos_w_align[ei] + t_m * held_pos_w_shallow[ei]
                    held_quat_w[ei] = quat_final[ei]

            pose_w = torch.cat([held_pos_w, held_quat_w], dim=-1)
            zero_vel = torch.zeros((num_envs, 6), device=device)
            inner._held_asset.write_root_pose_to_sim(pose_w)
            inner._held_asset.write_root_velocity_to_sim(zero_vel)
            inner._held_asset.reset()

            inner._robot.write_joint_state_to_sim(frozen_jpos, frozen_jvel)
            inner._robot.set_joint_position_target(frozen_jpos)
            inner._robot.set_joint_effort_target(frozen_jvel)

            if si < args_cli.n_hold + args_cli.n_kp_align:
                inner.step_sim_no_action()
            else:
                actions = torch.zeros(env.action_space.shape, device=device)
                env.step(actions)

            # Log kinematic intent (teleport target), not readback — avoids tail spikes when
            # PhysX resolves overlap after full insertion.
            held_pos_log = held_pos_w
            held_quat_log = held_quat_w
            fixed_pos_w = inner._fixed_asset.data.root_pos_w.clone()
            fixed_quat_cur = inner._fixed_asset.data.root_quat_w.clone()

            tip_offset = torch.tensor(_TIP_OFFSET_LOCAL, device=device, dtype=torch.float32).unsqueeze(0).expand(
                num_envs, -1
            )
            tip_world = held_pos_log + torch_utils.quat_rotate(held_quat_log, tip_offset)
            delta_w = tip_world - fixed_pos_w
            tip_local = quat_apply_inverse(fixed_quat_cur, delta_w)

            rel_quat = torch_utils.quat_mul(quat_conjugate(fixed_quat_cur), held_quat_log)
            rel_roll, rel_pitch, rel_yaw = euler_xyz_from_quat(rel_quat)
            rpy_local = torch.stack([rel_roll, rel_pitch, rel_yaw], dim=-1)

            held_rel_pos, held_rel_quat = inner.get_handheld_asset_relative_pose()
            ft_pose_env = _fingertip_pose7_env_from_held(
                held_pos_log,
                held_quat_log,
                held_rel_pos,
                held_rel_quat,
                inner.scene.env_origins,
            )

            for ei in range(batch):
                if traj_done[ei]:
                    continue
                env_tip_local[ei].append(tip_local[ei].cpu().numpy().tolist())
                env_rpy_local[ei].append(rpy_local[ei].cpu().numpy().tolist())
                env_pose7_local[ei].append(
                    torch.cat([tip_local[ei], rel_quat[ei]], dim=0).cpu().numpy().tolist()
                )
                env_ft_automate[ei].append(ft_pose_env[ei, :3].cpu().numpy().tolist())
                env_ft_pose_automate[ei].append(ft_pose_env[ei].cpu().numpy().tolist())

            curr_succ = inner._get_curr_successes(
                success_threshold=succ_threshold, check_rot=succ_check_rot
            )
            for ei in range(batch):
                if curr_succ[ei] and not traj_done[ei]:
                    frozen_pos_w[ei] = held_pos_w[ei].clone()
                    frozen_quat_w[ei] = held_quat_w[ei].clone()
                    traj_done[ei] = True

            if traj_done[:batch].all():
                break

        for ei in range(batch):
            tl, rpy, p7, fp, fq, n_kept, n_drop = _truncate_parallel_on_tip_jump(
                env_tip_local[ei],
                env_rpy_local[ei],
                env_pose7_local[ei],
                env_ft_automate[ei],
                env_ft_pose_automate[ei],
                args_cli.truncate_max_tip_step_m,
            )
            if n_drop > 0:
                print(
                    f"[collect] traj {traj_idx + ei}: truncated tail (tip step > "
                    f"{args_cli.truncate_max_tip_step_m} m): {n_kept + n_drop} -> {n_kept} frames"
                )
            collected.append(
                {
                    "held_tip_local": tl,
                    "held_rpy_local": rpy,
                    "held_tip_pose_local": p7,
                    "n_steps": len(tl),
                }
            )
            if not args_cli.skip_automate_output:
                collected_automate.append(
                    {
                        "fingertip_centered_pos": fp,
                        "fingertip_centered_pose": fq,
                        "n_steps": len(fp),
                    }
                )

        traj_idx += batch
        print(f"[collect] {len(collected)}/{args_cli.num_trajectories} trajectories")

    env.close()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    def _resolve_path(p: str) -> str:
        if not os.path.isabs(p):
            return os.path.join(repo_root, p)
        return p

    output_path = _resolve_path(args_cli.output)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(collected, f, indent=2)

    automate_path = None
    if not args_cli.skip_automate_output and collected_automate:
        if args_cli.output_automate:
            automate_path = _resolve_path(args_cli.output_automate)
        else:
            d, base = os.path.split(output_path)
            stem, ext = os.path.splitext(base)
            automate_path = os.path.join(d, f"{stem}_automate{ext}")
        os.makedirs(os.path.dirname(automate_path) or ".", exist_ok=True)
        with open(automate_path, "w") as f:
            json.dump(collected_automate, f, indent=2)

    automate_msg = (
        f"[collect] Automate: → {automate_path}\n"
        if automate_path
        else ("[collect] Automate: skipped\n" if args_cli.skip_automate_output else "")
    )
    print(
        f"\n{'=' * 60}\n"
        f"[collect] Forge: {len(collected)} trajectories → {output_path}\n"
        f"{automate_msg}"
        f"{'=' * 60}"
    )


if __name__ == "__main__":
    main()
    try:
        simulation_app.close()
    except Exception:
        pass
