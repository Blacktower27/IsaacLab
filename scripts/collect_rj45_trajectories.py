"""Collect RJ45 insertion reference trajectories in Isaac Sim.

Directly teleports the male plug along cubic-spline trajectories —
no robot, no OSC, no gripper physics. Each trajectory has two phases:

  Phase 1 (arc): Cubic spline from random contact-init pose to
                 directly above socket centre (XY/yaw alignment).
  Phase 2 (descent): Straight vertical drop into the socket.

The held-asset tip position and RPY are recorded every sim step in the
**socket (female) local frame**, matching Forge's coordinate conventions.

Output format (two files; no changes required inside Forge / Automate env code)
-------------------------------------------------------------------------------

**Forge / analysis** (``--output``)::

    [
      {
        "held_tip_local": [[x,y,z], ...],
        "held_rpy_local": [[r,p,y], ...],
        "held_tip_pose_local": [[x,y,z,qw,qx,qy,qz], ...],  # socket frame; for pose Soft-DTW in ForgeEnv
        "n_steps": N
      },
      ...
    ]

**Automate DTW** (``--output_automate`` or auto path, unless ``--skip_automate_output``)::

    [
      {
        "fingertip_centered_pos": [[x,y,z], ...],
        "fingertip_centered_pose": [[x,y,z,qw,qx,qy,qz], ...],
        "n_steps": N
      },
      ...
    ]

``fingertip_centered_pos`` / ``fingertip_centered_pose`` use the same convention as ``AssemblyEnv``:
position is env frame (body position minus ``scene.env_origins``); quaternion is fingertip
orientation in **world** (wxyz), matching ``preprocess_reference_pose_trajectory`` / pose DTW.
``AssemblyEnv`` loads **pose** demos when ``fingertip_centered_pose`` is present; otherwise it
uses position-only Soft-DTW on ``fingertip_centered_pos``. Rigid grasp: fingertip pose is computed
from the teleported plug pose using the same ``held_asset_*`` offsets as
``AssemblyEnv.get_handheld_asset_relative_pose`` for ``rj45_insert``, plus the Franka flip-Z chain
used in Factory reset.

Usage
-----
    ./isaaclab.sh -p scripts/collect_rj45_trajectories.py --num_envs 64 --num_trajectories 64 --output scripts/rj45_ref_traj.json
"""

from __future__ import annotations

import argparse
import math
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect RJ45 insertion reference trajectories.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_trajectories", type=int, default=200)
parser.add_argument("--output", type=str, default="scripts/rj45_ref_traj.json")
parser.add_argument(
    "--output_automate",
    type=str,
    default=None,
    help="Automate DTW JSON path (fingertip_centered_pos + fingertip_centered_pose). Default: <output> with stem "
         "suffix _automate before extension (e.g. rj45_ref_traj_automate.json).",
)
parser.add_argument(
    "--skip_automate_output",
    action="store_true",
    help="If set, only write the Forge JSON (--output).",
)
parser.add_argument("--n_phase1", type=int, default=60,
                    help="Waypoints for Phase 1 (cubic spline arc).")
parser.add_argument("--n_phase2", type=int, default=25,
                    help="Waypoints for Phase 2 (vertical descent).")
parser.add_argument("--approach_z_offset", type=float, default=0.030,
                    help="Z above target before descent begins (m).")
parser.add_argument("--target_z_offset", type=float, default=0.0,
                    help="Raise the insertion target by this amount (m). "
                         "E.g. 0.01 = stop 10 mm shallower than full insertion.")
parser.add_argument("--task", type=str, default="Isaac-Forge-RJ45Insert-Direct-v0")
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Imports after sim launch ──────────────────────────────────────────────

import json
import os

import gymnasium as gym
import numpy as np
import torch
from scipy.interpolate import CubicSpline

import isaacsim.core.utils.torch as torch_utils

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


# ── Geometry constants ────────────────────────────────────────────────────

_TIP_OFFSET_LOCAL = np.array([0.0, 0.0, -0.003])  # male tip 3 mm below USD origin


def _build_rj45_held_relative_tensors(task_cfg, device: torch.device, num_envs: int):
    """Plug origin in fingertip(flipped) frame — matches AssemblyEnv.get_handheld_asset_relative_pose (rj45)."""
    rc = task_cfg.robot_cfg
    held_rel_pos = torch.zeros((num_envs, 3), device=device, dtype=torch.float32)
    held_rel_pos[:, 2] = task_cfg.held_asset_cfg.height - rc.franka_fingerpad_length
    pos_offset = torch.tensor(task_cfg.held_asset_pos_offset, device=device, dtype=torch.float32)
    held_rel_pos = held_rel_pos + pos_offset.unsqueeze(0)

    initial_rot_deg = task_cfg.held_asset_rot_init
    rot_offset = getattr(task_cfg, "held_asset_rot_offset", [0.0, 0.0, 0.0])
    rot_euler = torch.tensor(
        [
            rot_offset[0] * np.pi / 180.0,
            rot_offset[1] * np.pi / 180.0,
            (initial_rot_deg + rot_offset[2]) * np.pi / 180.0,
        ],
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0).expand(num_envs, -1)
    held_rel_quat = torch_utils.quat_from_euler_xyz(
        roll=rot_euler[:, 0], pitch=rot_euler[:, 1], yaw=rot_euler[:, 2]
    )
    return held_rel_pos, held_rel_quat


def _fingertip_pose_env_from_plug(
    held_pos_w: torch.Tensor,
    held_quat_w: torch.Tensor,
    held_rel_pos: torch.Tensor,
    held_rel_quat: torch.Tensor,
    env_origins: torch.Tensor,
) -> torch.Tensor:
    """Fingertip pose for Automate: env-frame position (minus origins) + world wxyz quaternion."""
    # T_ft_flipped = T_plug * T_held_rel  (inverse of Factory: plug = ft_flipped * inv(held_rel))
    ft_flipped_q, ft_flipped_p = torch_utils.tf_combine(
        held_quat_w,
        held_pos_w,
        held_rel_quat,
        held_rel_pos,
    )
    flip_z = torch.tensor([0.0, 0.0, 1.0, 0.0], device=held_pos_w.device, dtype=torch.float32).unsqueeze(0).expand_as(
        held_quat_w
    )
    zero = torch.zeros_like(held_pos_w)
    flip_inv_q, flip_inv_p = torch_utils.tf_inverse(flip_z, zero)
    ft_q, ft_p = torch_utils.tf_combine(ft_flipped_q, ft_flipped_p, flip_inv_q, flip_inv_p)
    pos_env = ft_p - env_origins
    return torch.cat([pos_env, ft_q], dim=-1)


def _fingertip_pos_env_from_plug(
    held_pos_w: torch.Tensor,
    held_quat_w: torch.Tensor,
    held_rel_pos: torch.Tensor,
    held_rel_quat: torch.Tensor,
    env_origins: torch.Tensor,
) -> torch.Tensor:
    """World fingertip midpoint position, minus env_origins (first 3 dims of :func:`_fingertip_pose_env_from_plug`)."""
    return _fingertip_pose_env_from_plug(
        held_pos_w, held_quat_w, held_rel_pos, held_rel_quat, env_origins
    )[:, :3]


# ── Trajectory generation (mirrors visualize_rj45_trajectory.py) ─────────

def generate_spline_trajectory(
    start_xyz: np.ndarray,
    start_yaw: float,
    target_xyz: np.ndarray,
    target_yaw: float,
    approach_z_offset: float,
    n_phase1: int,
    n_phase2: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Two-phase insertion trajectory in socket local frame.

    Phase 1: Cubic spline arc from start to approach (above target).
    Phase 2: Linear vertical descent from approach to target.

    Returns:
        positions: (N, 3) XYZ waypoints.
        yaws:      (N,) yaw at each waypoint (rad).
    """
    approach_xyz = np.array([target_xyz[0], target_xyz[1],
                             target_xyz[2] + approach_z_offset])

    mid_xyz = np.array([
        (start_xyz[0] + approach_xyz[0]) / 2.0,
        (start_xyz[1] + approach_xyz[1]) / 2.0,
        max(start_xyz[2], approach_xyz[2]) + 0.01,
    ])

    pts = np.stack([start_xyz, mid_xyz, approach_xyz])
    diffs = np.diff(pts, axis=0)
    chord = np.concatenate([[0.0], np.cumsum(np.linalg.norm(diffs, axis=1))])
    t_norm = chord / chord[-1]

    cs_x = CubicSpline(t_norm, pts[:, 0])
    cs_y = CubicSpline(t_norm, pts[:, 1])
    cs_z = CubicSpline(t_norm, pts[:, 2])

    t1 = np.linspace(0.0, 1.0, n_phase1)
    p1_xyz = np.stack([cs_x(t1), cs_y(t1), cs_z(t1)], axis=-1)
    p1_yaw = np.linspace(start_yaw, target_yaw, n_phase1)

    z2 = np.linspace(approach_xyz[2], target_xyz[2], n_phase2)
    p2_xyz = np.column_stack([
        np.full(n_phase2, target_xyz[0]),
        np.full(n_phase2, target_xyz[1]),
        z2,
    ])
    p2_yaw = np.full(n_phase2, target_yaw)

    positions = np.concatenate([p1_xyz, p2_xyz[1:]], axis=0)
    yaws = np.concatenate([p1_yaw, p2_yaw[1:]])
    return positions, yaws


def sample_contact_init_local(cfg_task, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """Sample a random contact-init pose in socket local frame.

    Returns:
        xyz: (3,) plug USD origin in socket local frame.
        yaw: scalar yaw offset (rad).
    """
    yaw = rng.uniform(*np.deg2rad(cfg_task.contact_init_yaw_range_deg))

    fx = rng.uniform(*cfg_task.female_rear_edge_x_range_local)
    fy = cfg_task.female_rear_edge_y_local
    fz = cfg_task.female_rear_edge_z_local
    female_local = np.array([fx, fy, fz])

    mx = rng.uniform(*cfg_task.male_bottom_patch_x_range_local)
    my = rng.uniform(*cfg_task.male_bottom_patch_y_range_local)
    mz = cfg_task.male_bottom_patch_z_local
    male_local = np.array([mx, my, mz])

    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    plug_origin_local = female_local - R @ male_local

    return plug_origin_local, yaw


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=True,
    )
    env_cfg.seed = args_cli.seed
    env_cfg.episode_length_s = 60.0  # plenty of room

    # Park robot arm out of the way — plug is moved directly.
    env_cfg.task.hand_init_pos = [0.0, 0.0, 0.5]
    env_cfg.task.hand_init_pos_noise = [0.0, 0.0, 0.0]
    env_cfg.task.hand_init_orn_noise = [0.0, 0.0, 0.0]
    env_cfg.task.held_asset_pos_noise = [0.0, 0.0, 0.0]

    env = gym.make(args_cli.task, cfg=env_cfg)
    inner = env.unwrapped
    device = inner.device
    num_envs = inner.num_envs

    # Freeze robot arm.
    frozen_jpos = inner._robot.data.default_joint_pos.clone()
    frozen_jpos[:, :7] = torch.tensor(
        inner.cfg.ctrl.reset_joints, device=device
    ).unsqueeze(0).expand(num_envs, -1)
    frozen_jvel = torch.zeros_like(frozen_jpos)

    # Insertion target in socket local frame (plug USD origin).
    target_y = getattr(env_cfg.task, "socket_target_y_local", 0.0)
    target_z_tip = getattr(env_cfg.task, "socket_target_z_local", 0.014)
    base_height = env_cfg.task.held_asset_cfg.base_height  # -0.003
    # target_z_origin = target_z_tip - base_height + args_cli.target_z_offset
    target_z_origin = 0.032
    target_xyz = np.array([0.0, target_y, target_z_origin])
    print(f"[collect] target_z_origin = {target_z_origin:.4f} m "
          f"(full insertion = {target_z_tip - base_height:.4f}, "
          f"offset = {args_cli.target_z_offset:+.4f})")

    env.reset()

    held_rel_pos, held_rel_quat = _build_rj45_held_relative_tensors(env_cfg.task, device, num_envs)

    rng = np.random.default_rng(args_cli.seed)
    collected: list[dict] = []
    collected_automate: list[dict] = []
    n_waypoints = args_cli.n_phase1 + args_cli.n_phase2 - 1

    print(
        f"\n{'='*60}\n"
        f"[collect] Direct teleport mode (no robot physics)\n"
        f"[collect] Target: {args_cli.num_trajectories} trajectories, "
        f"{num_envs} envs, {n_waypoints} waypoints each\n"
        f"[collect] Insertion target (socket local): {target_xyz}\n"
        f"{'='*60}\n"
    )

    # Pre-generate all trajectories (pure math, fast).
    all_trajs: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(args_cli.num_trajectories):
        start_xyz, start_yaw = sample_contact_init_local(env_cfg.task, rng)
        positions, yaws = generate_spline_trajectory(
            start_xyz, start_yaw, target_xyz, 0.0,
            args_cli.approach_z_offset,
            args_cli.n_phase1, args_cli.n_phase2,
        )
        all_trajs.append((positions, yaws))

    # Play trajectories through the sim in batches of num_envs.
    traj_idx = 0
    while traj_idx < len(all_trajs) and simulation_app.is_running():
        batch_size = min(num_envs, len(all_trajs) - traj_idx)
        batch_trajs = all_trajs[traj_idx : traj_idx + batch_size]

        # Per-env trajectory storage (socket local frame).
        env_tip_local = [[] for _ in range(batch_size)]
        env_rpy_local = [[] for _ in range(batch_size)]
        env_pose7_local = [[] for _ in range(batch_size)]
        env_ft_automate = [[] for _ in range(batch_size)]
        env_ft_pose_automate = [[] for _ in range(batch_size)]

        # Step through waypoints.
        for wi in range(n_waypoints):
            # Teleport each env's plug to its waypoint.
            socket_pos_w = inner._fixed_asset.data.root_pos_w.clone()
            socket_quat_w = inner._fixed_asset.data.root_quat_w.clone()
            ident_q = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], device=device
            ).unsqueeze(0).expand(num_envs, -1)

            plug_pos_w = socket_pos_w.clone()
            plug_quat_w = socket_quat_w.clone()

            for ei in range(batch_size):
                pos_local_np, yaw = batch_trajs[ei][0][wi], batch_trajs[ei][1][wi]
                pos_local = torch.tensor(
                    pos_local_np, device=device, dtype=torch.float32
                ).unsqueeze(0)
                off_world = torch_utils.quat_rotate(socket_quat_w[ei:ei+1], pos_local)
                plug_pos_w[ei] = socket_pos_w[ei] + off_world.squeeze(0)

                delta_q = torch_utils.quat_from_euler_xyz(
                    torch.tensor([0.0], device=device),
                    torch.tensor([0.0], device=device),
                    torch.tensor([float(yaw)], device=device),
                )
                plug_quat_w[ei] = torch_utils.quat_mul(
                    socket_quat_w[ei:ei+1], delta_q
                ).squeeze(0)

            pose_w = torch.cat([plug_pos_w, plug_quat_w], dim=-1)
            zero_vel = torch.zeros((num_envs, 6), device=device)
            inner._held_asset.write_root_pose_to_sim(pose_w)
            inner._held_asset.write_root_velocity_to_sim(zero_vel)
            inner._held_asset.reset()

            # Freeze robot.
            inner._robot.write_joint_state_to_sim(frozen_jpos, frozen_jvel)
            inner._robot.set_joint_position_target(frozen_jpos)
            inner._robot.set_joint_effort_target(frozen_jvel)

            # Step sim so physics updates.
            actions = torch.zeros(env.action_space.shape, device=device)
            env.step(actions)

            # Record held tip position & RPY in socket local frame.
            held_pos = inner._held_asset.data.root_pos_w.clone()
            held_quat = inner._held_asset.data.root_quat_w.clone()
            fixed_pos = inner._fixed_asset.data.root_pos_w.clone()
            fixed_quat = inner._fixed_asset.data.root_quat_w.clone()

            tip_offset = torch.tensor(
                _TIP_OFFSET_LOCAL, device=device, dtype=torch.float32
            ).unsqueeze(0).expand(num_envs, -1)
            tip_world = held_pos + torch_utils.quat_rotate(held_quat, tip_offset)
            delta_w = tip_world - fixed_pos
            # Rotate to socket local frame.
            fixed_quat_inv = fixed_quat.clone()
            fixed_quat_inv[:, 1:] *= -1
            tip_local = torch_utils.quat_rotate(fixed_quat_inv, delta_w)

            # Relative orientation → RPY.
            rel_quat = torch_utils.quat_mul(fixed_quat_inv, held_quat)
            from isaaclab.utils.math import euler_xyz_from_quat
            rel_roll, rel_pitch, rel_yaw = euler_xyz_from_quat(rel_quat)
            rpy_local = torch.stack([rel_roll, rel_pitch, rel_yaw], dim=-1)

            ft_pose_env = _fingertip_pose_env_from_plug(
                held_pos,
                held_quat,
                held_rel_pos,
                held_rel_quat,
                inner.scene.env_origins,
            )

            for ei in range(batch_size):
                env_tip_local[ei].append(tip_local[ei].cpu().numpy().tolist())
                env_rpy_local[ei].append(rpy_local[ei].cpu().numpy().tolist())
                env_pose7_local[ei].append(
                    torch.cat([tip_local[ei], rel_quat[ei]], dim=0).cpu().numpy().tolist()
                )
                env_ft_automate[ei].append(ft_pose_env[ei, :3].cpu().numpy().tolist())
                env_ft_pose_automate[ei].append(ft_pose_env[ei].cpu().numpy().tolist())

        # Flush batch.
        for ei in range(batch_size):
            collected.append({
                "held_tip_local": env_tip_local[ei],
                "held_rpy_local": env_rpy_local[ei],
                "held_tip_pose_local": env_pose7_local[ei],
                "n_steps": len(env_tip_local[ei]),
            })
            if not args_cli.skip_automate_output:
                collected_automate.append({
                    "fingertip_centered_pos": env_ft_automate[ei],
                    "fingertip_centered_pose": env_ft_pose_automate[ei],
                    "n_steps": len(env_ft_automate[ei]),
                })

        traj_idx += batch_size
        print(
            f"[collect] {len(collected)}/{args_cli.num_trajectories} trajectories"
        )

    env.close()

    # ── Save ──────────────────────────────────────────────────────────────
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

    final_z = [t["held_tip_local"][-1][2] for t in collected]
    start_z = [t["held_tip_local"][0][2] for t in collected]
    final_yaw = [abs(t["held_rpy_local"][-1][2]) for t in collected]
    msg = (
        f"\n{'='*60}\n"
        f"[collect] Forge: saved {len(collected)} trajectories → {output_path}\n"
    )
    if automate_path:
        msg += f"[collect] Automate DTW: saved {len(collected_automate)} trajectories → {automate_path}\n"
    elif args_cli.skip_automate_output:
        msg += "[collect] Automate DTW: skipped (--skip_automate_output)\n"
    msg += (
        f"[collect] Start tip Z: min={min(start_z):.4f}, max={max(start_z):.4f} m\n"
        f"[collect] Final tip Z: min={min(final_z):.4f}, max={max(final_z):.4f} m\n"
        f"[collect] Final |yaw|: min={min(final_yaw):.4f}, max={max(final_yaw):.4f} rad\n"
        f"[collect] RPY at endpoint should be [0, 0, 0] (aligned with socket)\n"
        f"{'='*60}"
    )
    print(msg)


if __name__ == "__main__":
    main()
    try:
        simulation_app.close()
    except Exception:
        pass
