"""Collect BNC (bayonet) reference trajectories in Isaac Sim.

Teleports the male plug only (Franka, separate held asset) through **two** segments
in the **socket (female) local frame** (helical / phase-3 lock is **not** recorded):

1. **Approach** — cubic spline from a random contact-init pose (``bnc_contact_init_*``)
   toward centred XY, tip height ``approach_delta_z_m`` above the final-seated target.
2. **Axial insert** — straight line from the approach end to the plug at **``--tip_z_phase2_end``**
   (connector tip Z in socket frame, same depth as the old phase-3 / helix *start* — not the fully seated
   ``--tip_z_target``). Constant yaw. No bayonet twist in the reference.

Recordings use the **connector tip** in the female frame (and relative quat), matching
``ForgeEnv._get_held_tip_local`` for ``bnc_insert`` (``BNCSmallMaleCfg.base_height`` along +Z
from the plug root).

Output: same as ``collect_rj45_trajectories.py`` / ``collect_box_lid_trajectories.py``:

* ``--output`` — Forge: ``held_tip_local``, ``held_rpy_local``, ``held_tip_pose_local`` (7D)
* ``--output_automate`` — ``fingertip_centered_pos``, ``fingertip_centered_pose`` (if not skipped)

Before writing the JSON, **obviously bad** trajectories (NaNs, large per-step jumps, final tip far from
``--tip_z_phase2_end``, |XY| or tilt out of range) are dropped; tune ``--traj_*`` or
``--skip_traj_validity_filter`` to debug.

Usage::

    ./isaaclab.sh -p scripts/collect_bnc_trajectories.py \\
        --num_envs 8 --num_trajectories 64 --output scripts/bnc_ref_traj.json

"""

from __future__ import annotations

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect BNC reference trajectories (2 phases: approach + axial).")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--num_trajectories", type=int, default=200)
parser.add_argument("--output", type=str, default="scripts/bnc_ref_traj.json")
parser.add_argument(
    "--output_automate",
    type=str,
    default=None,
    help="Default: <output> with _automate before extension.",
)
parser.add_argument("--skip_automate_output", action="store_true", default=False)
parser.add_argument("--n_phase1", type=int, default=55, help="Cubic-spline approach.")
parser.add_argument("--n_phase2", type=int, default=30, help="Axial line to seated tip (constant yaw).")
parser.add_argument(
    "--tip_z_target",
    type=float,
    default=-0.004765,
    help="Reference **seated** tip Z in socket (m) — used to place the *approach* (with "
    "``--approach_delta_z_m``) above that level. **Phase-2 end** is ``--tip_z_phase2_end``, not this, "
    "unless you set them equal on purpose.",
)
parser.add_argument(
    "--tip_z_phase2_end",
    type=float,
    default=0.005,
    help="**Tip** Z in socket frame (m) at end of phase-2 axial: same as former helix / P3 *start* depth. "
    "Calibrated: tip ≈ 20 mm inside vs opening and opening at female Z=+0.025 m → tip_z ≈ 0.005 m.",
)
parser.add_argument(
    "--approach_delta_z_m",
    type=float,
    default=0.030,
    help="How far (in tip Z, m) the approach endpoint sits **above** ``--tip_z_target`` (seated ref).",
)
parser.add_argument(
    "--bnc_branch_pi",
    type=float,
    default=0.5,
    help="Per trajectory: probability to use 180 deg insertion branch (second bayonet).",
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Forge-BNCSmallInsert-Direct-v0",
    help="Franka BNC (separate held_asset).",
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--skip_traj_validity_filter",
    action="store_true",
    default=False,
    help="If set, write all collected samples as before; otherwise drop any trajectory that "
    "fails sanity (NaN, huge per-step jump, bad final tip XY/Z vs --tip_z_phase2_end, large roll/pitch).",
)
parser.add_argument(
    "--traj_max_step_norm_m",
    type=float,
    default=0.02,
    help="With validity filter: reject if any consecutive step has |Δp| above this (m).",
)
parser.add_argument(
    "--traj_max_final_xy_m",
    type=float,
    default=0.020,
    help="With validity filter: reject if final |tip_xy| in female frame exceeds this (m).",
)
parser.add_argument(
    "--traj_z_end_tol_m",
    type=float,
    default=0.030,
    help="With validity filter: reject if |final tip_z - tip_z_phase2_end| exceeds this (m).",
)
parser.add_argument(
    "--traj_max_tilt_rad",
    type=float,
    default=0.6,
    help="With validity filter: reject if |roll| or |pitch| (held_rpy_local) exceeds this (rad) at any time.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── post-launch imports ───────────────────────────────────────────────────
import json

import gymnasium as gym
import numpy as np
import torch
from scipy.interpolate import CubicSpline

import isaacsim.core.utils.torch as torch_utils

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import euler_xyz_from_quat


def _tip_offset_np(task) -> np.ndarray:
    h = float(task.held_asset_cfg.base_height)
    return np.array([0.0, 0.0, h], dtype=np.float64)


def _build_bnc_held_relative_tensors(task_cfg, device: torch.device, num_envs: int):
    """Match ``FactoryEnv.get_handheld_asset_relative_pose`` for bnc_insert (coax, pos only)."""
    rc = task_cfg.robot_cfg
    held_rel_pos = torch.zeros((num_envs, 3), device=device, dtype=torch.float32)
    held_rel_pos[:, 2] = task_cfg.held_asset_cfg.height - rc.franka_fingerpad_length
    pos_offset = torch.tensor(task_cfg.held_asset_pos_offset, device=device, dtype=torch.float32)
    held_rel_pos = held_rel_pos + pos_offset.unsqueeze(0)
    held_rel_quat = (
        torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=torch.float32)
        .unsqueeze(0)
        .expand(num_envs, -1)
    )
    return held_rel_pos, held_rel_quat


def _fingertip_pose_env_from_plug(
    held_pos_w: torch.Tensor,
    held_quat_w: torch.Tensor,
    held_rel_pos: torch.Tensor,
    held_rel_quat: torch.Tensor,
    env_origins: torch.Tensor,
) -> torch.Tensor:
    ft_flipped_q, ft_flipped_p = torch_utils.tf_combine(
        held_quat_w, held_pos_w, held_rel_quat, held_rel_pos
    )
    flip_z = torch.tensor([0.0, 0.0, 1.0, 0.0], device=held_pos_w.device, dtype=torch.float32).unsqueeze(0).expand_as(
        held_quat_w
    )
    zero = torch.zeros_like(held_pos_w)
    flip_inv_q, flip_inv_p = torch_utils.tf_inverse(flip_z, zero)
    ft_q, ft_p = torch_utils.tf_combine(ft_flipped_q, ft_flipped_p, flip_inv_q, flip_inv_p)
    pos_env = ft_p - env_origins
    return torch.cat([pos_env, ft_q], dim=-1)


def _origin_from_tip_coaxial(tip: np.ndarray, h: float) -> np.ndarray:
    """Coax: plug +Z = socket +Z, so body origin shares XY with tip, ``tip Z = origin Z + h``."""
    o = np.array(tip, dtype=np.float64, copy=True)
    o[2] = float(tip[2]) - h
    return o


def sample_bnc_contact_init(rng: np.random.Generator, task) -> tuple[np.ndarray, float]:
    """Contact-init plug **origin** in socket frame + a yaw (rad) before π branch. Matches automate reset (yaw only)."""
    fxl, fxh = task.bnc_contact_init_female_x_range
    fyl, fyh = task.bnc_contact_init_female_y_range
    fz = float(task.bnc_contact_init_female_z_local)
    mxl, mxh = task.bnc_contact_init_male_x_range
    myl, myh = task.bnc_contact_init_male_y_range
    mz = float(task.bnc_contact_init_male_z_local)

    fx = rng.uniform(fxl, fxh)
    fy = rng.uniform(fyl, fyh)
    female = np.array([fx, fy, fz], dtype=np.float64)
    mx = rng.uniform(mxl, mxh)
    my = rng.uniform(myl, myh)
    male = np.array([mx, my, mz], dtype=np.float64)
    yaw0 = float(
        rng.uniform(
            *np.radians(
                (task.contact_init_yaw_range_deg[0], task.contact_init_yaw_range_deg[1])
            )
        )
    )
    c, s = math.cos(yaw0), math.sin(yaw0)
    r = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    plug_origin = female - r @ male
    return plug_origin, yaw0


def _bnc_traj_passes_sanity(
    rec: dict,
    *,
    expect_tip_z_end: float,
    max_step_norm_m: float,
    max_final_xy_m: float,
    z_end_tol_m: float,
    max_tilt_rad: float,
) -> bool:
    """Drop trajectories with NaNs, pathological jumps, or far-off final pose vs. ``--tip_z_phase2_end``."""
    tips = np.asarray(rec.get("held_tip_local"), dtype=np.float64)
    rpy = np.asarray(rec.get("held_rpy_local"), dtype=np.float64)
    if tips.ndim != 2 or tips.shape[0] < 2 or rpy.ndim != 2 or rpy.shape[0] < 1:
        return False
    if not np.isfinite(tips).all() or not np.isfinite(rpy).all():
        return False
    d = np.linalg.norm(np.diff(tips, axis=0), axis=1)
    if d.size and float(np.max(d)) > max_step_norm_m:
        return False
    tlast = tips[-1]
    if float(np.hypot(tlast[0], tlast[1])) > max_final_xy_m:
        return False
    if abs(float(tlast[2]) - float(expect_tip_z_end)) > z_end_tol_m:
        return False
    if (np.abs(rpy[:, 0]) > max_tilt_rad).any() or (np.abs(rpy[:, 1]) > max_tilt_rad).any():
        return False
    return True


def generate_bnc_trajectory(
    start_origin: np.ndarray,
    start_yaw: float,
    h: float,
    tip_z_seated_ref: float,
    tip_z_phase2_end: float,
    n_phase1: int,
    n_phase2: int,
    approach_delta_z: float,
    use_pi: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (plug **origin** (N,3) socket frame, yaw (N,)) metres / rad. Two phases only (no helix)."""
    h = float(h)
    # Approach is defined above the *seated* ref (``tip_z_seated_ref``), not necessarily P2 end.
    tip_approach = float(tip_z_seated_ref) + float(approach_delta_z)
    o_approach = _origin_from_tip_coaxial(np.array([0.0, 0.0, tip_approach], dtype=np.float64), h)
    o_p2 = _origin_from_tip_coaxial(
        np.array([0.0, 0.0, float(tip_z_phase2_end)], dtype=np.float64), h
    )

    align_yaw = math.pi if use_pi else 0.0

    p0 = np.array(start_origin, dtype=np.float64, copy=True)
    p2 = o_approach
    p1 = np.array(
        [
            0.5 * (p0[0] + p2[0]),
            0.5 * (p0[1] + p2[1]),
            max(p0[2], p2[2]) + 0.008,
        ],
        dtype=np.float64,
    )
    pts = np.stack([p0, p1, p2], axis=0)
    diffs = np.diff(pts, axis=0)
    chord = np.concatenate([[0.0], np.cumsum(np.linalg.norm(diffs, axis=1))])
    t_norm = chord / chord[-1] if chord[-1] > 0 else np.array([0.0, 0.5, 1.0], dtype=np.float64)
    cs_x = CubicSpline(t_norm, pts[:, 0])
    cs_y = CubicSpline(t_norm, pts[:, 1])
    cs_z = CubicSpline(t_norm, pts[:, 2])
    t1 = np.linspace(0.0, 1.0, n_phase1, dtype=np.float64)
    p1_xyz = np.stack([cs_x(t1), cs_y(t1), cs_z(t1)], axis=-1)
    p1_xyz[-1] = p2
    p1_yaw = np.linspace(float(start_yaw), float(align_yaw), n_phase1, dtype=np.float64)

    t2 = np.linspace(0.0, 1.0, n_phase2, dtype=np.float64)
    a = p1_xyz[-1]
    p2_xyz = (1.0 - t2)[:, None] * a[None, :] + t2[:, None] * o_p2[None, :]
    p2_yaw = np.full(n_phase2, float(align_yaw), dtype=np.float64)

    origins = np.concatenate([p1_xyz, p2_xyz[1:]], axis=0)
    yaws = np.concatenate([p1_yaw, p2_yaw[1:]], axis=0)
    return origins, yaws


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=True,
    )
    env_cfg.seed = args_cli.seed
    env_cfg.episode_length_s = 60.0

    env_cfg.task.hand_init_pos = [0.0, 0.0, 0.5]
    env_cfg.task.hand_init_pos_noise = [0.0, 0.0, 0.0]
    env_cfg.task.hand_init_orn_noise = [0.0, 0.0, 0.0]
    env_cfg.task.held_asset_pos_noise = [0.0, 0.0, 0.0]

    env = gym.make(args_cli.task, cfg=env_cfg)
    inner = env.unwrapped
    device = inner.device
    num_envs = inner.num_envs
    task = env_cfg.task

    h = float(task.held_asset_cfg.base_height)
    tip_off = _tip_offset_np(task)

    frozen_jpos = inner._robot.data.default_joint_pos.clone()
    frozen_jpos[:, :7] = torch.tensor(inner.cfg.ctrl.reset_joints, device=device).unsqueeze(0).expand(
        num_envs, -1
    )
    frozen_jvel = torch.zeros_like(frozen_jpos)

    rng = np.random.default_rng(args_cli.seed)
    collected: list[dict] = []
    collected_automate: list[dict] = []
    n_way = args_cli.n_phase1 + (args_cli.n_phase2 - 1)

    trajs: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(args_cli.num_trajectories):
        s_o, s_y = sample_bnc_contact_init(rng, task)
        use_pi = bool(rng.random() < float(args_cli.bnc_branch_pi))
        if rng.random() < 0.5:
            s_y = s_y + math.pi
        o, y = generate_bnc_trajectory(
            s_o,
            s_y,
            h,
            args_cli.tip_z_target,
            args_cli.tip_z_phase2_end,
            args_cli.n_phase1,
            args_cli.n_phase2,
            args_cli.approach_delta_z_m,
            use_pi,
        )
        trajs.append((o, y))

    held_rel_pos, held_rel_quat = _build_bnc_held_relative_tensors(env_cfg.task, device, num_envs)

    env.reset()
    print(
        f"\n[collect] BNC (2 phases) — h={h} m, approach ref (seated) tip_z={args_cli.tip_z_target} m, "
        f"**P2 end tip_z**={args_cli.tip_z_phase2_end} m, n1={args_cli.n_phase1} n2={args_cli.n_phase2}, "
        f"waypoints={n_way} (helical lock not recorded)\n"
    )

    traj_i = 0
    while traj_i < len(trajs) and simulation_app.is_running():
        bsz = min(num_envs, len(trajs) - traj_i)
        batch = trajs[traj_i : traj_i + bsz]
        env_tip = [[] for _ in range(bsz)]
        env_rpy = [[] for _ in range(bsz)]
        env_pose7 = [[] for _ in range(bsz)]
        env_ft = [[] for _ in range(bsz)]
        env_ftp = [[] for _ in range(bsz)]

        for wi in range(n_way):
            sp = inner._fixed_asset.data.root_pos_w.clone()
            sq = inner._fixed_asset.data.root_quat_w.clone()
            ident_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0).expand(num_envs, -1)
            p_w = sp.clone()
            q_w = sq.clone()
            for ei in range(bsz):
                o_l, yv = batch[ei]
                o = torch.tensor(o_l[wi], device=device, dtype=torch.float32).unsqueeze(0)
                p_w[ei] = sp[ei] + torch_utils.quat_rotate(sq[ei : ei + 1], o).squeeze(0)
                dq = torch_utils.quat_from_euler_xyz(
                    torch.tensor([0.0], device=device),
                    torch.tensor([0.0], device=device),
                    torch.tensor([float(yv[wi])], device=device),
                )
                q_w[ei] = torch_utils.quat_mul(sq[ei : ei + 1], dq).squeeze(0)

            pose = torch.cat([p_w, q_w], dim=-1)
            inner._held_asset.write_root_pose_to_sim(pose)
            inner._held_asset.write_root_velocity_to_sim(torch.zeros((num_envs, 6), device=device))
            inner._held_asset.reset()

            inner._robot.write_joint_state_to_sim(frozen_jpos, frozen_jvel)
            inner._robot.set_joint_position_target(frozen_jpos)
            inner._robot.set_joint_effort_target(frozen_jvel)

            _ = env.step(torch.zeros(env.action_space.shape, device=device))

            hp = inner._held_asset.data.root_pos_w.clone()
            hq = inner._held_asset.data.root_quat_w.clone()
            fp = inner._fixed_asset.data.root_pos_w.clone()
            fq = inner._fixed_asset.data.root_quat_w.clone()

            tip_t = (
                torch.tensor(
                    [float(tip_off[0]), float(tip_off[1]), float(tip_off[2])],
                    device=device,
                    dtype=torch.float32,
                )
                .unsqueeze(0)
                .expand(num_envs, -1)
            )
            tip_w = hp + torch_utils.quat_rotate(hq, tip_t)
            d_w = tip_w - fp
            fq_i = fq.clone()
            fq_i[:, 1:] *= -1.0
            tip_l = torch_utils.quat_rotate(fq_i, d_w)
            relq = torch_utils.quat_mul(fq_i, hq)
            rr, pp, yy = euler_xyz_from_quat(relq)
            rpy = torch.stack([rr, pp, yy], dim=-1)

            fp_env = _fingertip_pose_env_from_plug(
                hp, hq, held_rel_pos, held_rel_quat, inner.scene.env_origins
            )

            for ei in range(bsz):
                env_tip[ei].append(tip_l[ei].cpu().numpy().tolist())
                env_rpy[ei].append(rpy[ei].cpu().numpy().tolist())
                env_pose7[ei].append(
                    torch.cat([tip_l[ei], relq[ei]], dim=0).cpu().numpy().tolist()
                )
                env_ft[ei].append(fp_env[ei, :3].cpu().numpy().tolist())
                env_ftp[ei].append(fp_env[ei].cpu().numpy().tolist())

        for ei in range(bsz):
            collected.append(
                {
                    "held_tip_local": env_tip[ei],
                    "held_rpy_local": env_rpy[ei],
                    "held_tip_pose_local": env_pose7[ei],
                    "n_steps": len(env_tip[ei]),
                }
            )
            if not args_cli.skip_automate_output:
                collected_automate.append(
                    {
                        "fingertip_centered_pos": env_ft[ei],
                        "fingertip_centered_pose": env_ftp[ei],
                        "n_steps": len(env_ft[ei]),
                    }
                )
        traj_i += bsz
        print(f"[collect] {len(collected)}/{args_cli.num_trajectories} done")

    env.close()

    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    def _p(p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(repo, p)

    n_raw = len(collected)
    n_rejected = 0
    if not bool(getattr(args_cli, "skip_traj_validity_filter", False)) and n_raw > 0:
        kept: list[dict] = []
        kept_auto: list[dict] = []
        for i, rec in enumerate(collected):
            ok = _bnc_traj_passes_sanity(
                rec,
                expect_tip_z_end=float(args_cli.tip_z_phase2_end),
                max_step_norm_m=float(args_cli.traj_max_step_norm_m),
                max_final_xy_m=float(args_cli.traj_max_final_xy_m),
                z_end_tol_m=float(args_cli.traj_z_end_tol_m),
                max_tilt_rad=float(args_cli.traj_max_tilt_rad),
            )
            if ok:
                kept.append(rec)
                if not args_cli.skip_automate_output and i < len(collected_automate):
                    kept_auto.append(collected_automate[i])
            else:
                n_rejected += 1
        collected = kept
        if not args_cli.skip_automate_output:
            collected_automate = kept_auto
    elif bool(getattr(args_cli, "skip_traj_validity_filter", False)):
        print("[collect] traj validity filter **disabled** (skip_traj_validity_filter).")

    outp = _p(args_cli.output)
    os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
    with open(outp, "w") as f:
        json.dump(collected, f, indent=2)

    auto_path = None
    if not args_cli.skip_automate_output and collected_automate:
        if args_cli.output_automate:
            auto_path = _p(args_cli.output_automate)
        else:
            d, base = os.path.split(outp)
            stem, ext = os.path.splitext(base)
            auto_path = os.path.join(d, f"{stem}_automate{ext}")
        with open(auto_path, "w") as f:
            json.dump(collected_automate, f, indent=2)

    if n_raw and not bool(getattr(args_cli, "skip_traj_validity_filter", False)):
        print(
            f"[collect] After sanity filter: kept {len(collected)}/{n_raw} "
            f"(rejected {n_rejected} obviously bad). tip_z_phase2_end={args_cli.tip_z_phase2_end} m"
        )

    ftz = [t["held_tip_local"][-1][2] for t in collected] if collected else []
    fy = [t["held_rpy_local"][-1][2] for t in collected] if collected else []
    if ftz and fy:
        print(
            f"[collect] Wrote {len(collected)} → {outp}\n"
            f"  final tip Z: [{min(ftz):.5f}, {max(ftz):.5f}]  final yaw: [{min(fy):.4f}, {max(fy):.4f}] rad"
        )
    else:
        print(f"[collect] Wrote 0 → {outp} (all rejected or no samples; widen tolerances or use --skip_traj_validity_filter)")
    if auto_path:
        print(f"  Automate: {auto_path}")


if __name__ == "__main__":
    main()
    try:
        simulation_app.close()
    except Exception:
        pass
