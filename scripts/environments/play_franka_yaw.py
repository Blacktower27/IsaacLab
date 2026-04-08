"""Play a Franka FORGE task with real-time yaw dead-zone visualization.

Draws a ring around the fixed asset (socket / box) every step:
  GREEN arc   — reachable yaw [-180°, +90°]  (action ∈ [-1, +1])
  RED arc     — dead zone     (+90°, +180°]   (unreachable by policy)
  YELLOW dots — current EE yaw direction relative to fixed asset
  CYAN dots   — policy-commanded target yaw   (decoded from raw action)

Usage
-----
./isaaclab.sh -p scripts/environments/play_franka_yaw.py \\
    --task Isaac-Forge-RJ45Insert-Direct-v0 \\
    --checkpoint <path_to_checkpoint.pth>
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Franka FORGE play with yaw dead-zone visualization.")
parser.add_argument("--num_envs",   type=int,  default=None)
parser.add_argument("--task",       type=str,  default=None)
parser.add_argument("--agent",      type=str,  default="rl_games_cfg_entry_point")
parser.add_argument("--checkpoint", type=str,  default=None)
parser.add_argument("--seed",       type=int,  default=None)
parser.add_argument("--real-time",  action="store_true", default=False)
parser.add_argument(
    "--use_last_checkpoint", action="store_true",
    help="Use last saved model instead of best.",
)
parser.add_argument(
    "--arc_radius", type=float, default=0.050,
    help="Yaw arc ring radius in metres (default 0.05).",
)
parser.add_argument(
    "--arc_z", type=float, default=0.015,
    help="Yaw arc height above fixed-asset origin in metres (default 0.015).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import os
import random
import time

import gymnasium as gym
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

import isaacsim.core.utils.torch as torch_utils
import isaaclab.sim as sim_utils
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import euler_xyz_from_quat, quat_conjugate, quat_mul

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# ---------------------------------------------------------------------------
# Yaw arc constants — Franka action → yaw mapping:
#   action ∈ [-1, +1]  →  yaw ∈ [-180°, +90°]  (270° range)
#   Dead zone: (+90°, +180°] — the 90° sector the policy can never command.
# ---------------------------------------------------------------------------
_YAW_ARC_SAMPLES = 72      # one sphere per 5° (full 360°)
_YAW_IND_SAMPLES = 5       # dots along indicator lines
# _FRANKA_YAW_MIN  = math.radians(-180.0)  # old: valid [-180°,+90°]
# _FRANKA_YAW_MAX  = math.radians(  90.0)  # old
# _FRANKA_YAW_MIN  = math.radians(   0.0)   # old: valid [0°,+270°], dead zone (-90°,0°)
# _FRANKA_YAW_MAX  = math.radians( 270.0)   # old
_FRANKA_YAW_MIN  = math.radians(-180.0)   # no dead zone: full 360°
_FRANKA_YAW_MAX  = math.radians( 180.0)

# Marker indices
_IDX_ARC_OK  = 0   # green  — reachable
_IDX_ARC_BAD = 1   # red    — dead zone
_IDX_CURR    = 2   # yellow — current EE yaw
_IDX_CMD     = 3   # cyan   — policy-commanded target yaw

_yaw_markers: VisualizationMarkers | None = None


def _get_yaw_markers() -> VisualizationMarkers:
    global _yaw_markers
    if _yaw_markers is None:
        cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/FrankaYawArc",
            markers={
                "arc_ok":  sim_utils.SphereCfg(   # reachable arc — bright green
                    radius=0.0018,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 1.0, 0.3)),
                ),
                "arc_bad": sim_utils.SphereCfg(   # dead-zone arc — red
                    radius=0.0018,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.1, 0.1)),
                ),
                "curr_yaw": sim_utils.SphereCfg(  # current EE yaw — yellow
                    radius=0.0030,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
                ),
                "cmd_yaw": sim_utils.SphereCfg(   # commanded target yaw — cyan
                    radius=0.0025,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.95, 0.95)),
                ),
            },
        )
        _yaw_markers = VisualizationMarkers(cfg)
    return _yaw_markers


def _draw_yaw_arc(base_env, raw_actions: torch.Tensor):
    """Draw yaw dead-zone arc + indicators around each env's fixed asset.

    Args:
        base_env:    Unwrapped IsaacLab env (ForgeEnv subclass).
        raw_actions: Raw policy output tensor, shape (num_envs, action_dim).
                     actions[:, 5] is the yaw action ∈ [-1, +1].
    """
    markers  = _get_yaw_markers()
    device   = base_env.device
    num_envs = base_env.num_envs

    arc_r = args_cli.arc_radius
    arc_z = args_cli.arc_z

    fixed_pos_w  = base_env._fixed_asset.data.root_pos_w    # (N,3) world frame
    fixed_quat   = base_env._fixed_asset.data.root_quat_w   # (N,4) wxyz
    ee_quat      = base_env.fingertip_midpoint_quat          # (N,4) wxyz

    ident_q = torch.zeros((num_envs, 4), device=device)
    ident_q[:, 0] = 1.0

    translations_list   = []
    marker_indices_list = []

    # ------------------------------------------------------------------
    # 1) Full 360° ring: green = reachable, red = dead zone.
    # ------------------------------------------------------------------
    for i in range(_YAW_ARC_SAMPLES):
        theta    = math.radians(-180.0 + 360.0 * i / _YAW_ARC_SAMPLES)
        theta_norm = theta % (2 * math.pi)   # normalize to [0, 2π) for range check
        in_range = _FRANKA_YAW_MIN <= theta_norm <= _FRANKA_YAW_MAX
        midx     = _IDX_ARC_OK if in_range else _IDX_ARC_BAD
        local_pt = torch.tensor(
            [arc_r * math.cos(theta), arc_r * math.sin(theta), arc_z],
            device=device, dtype=torch.float32,
        ).unsqueeze(0).expand(num_envs, -1)
        _, arc_w = torch_utils.tf_combine(fixed_quat, fixed_pos_w, ident_q, local_pt)
        translations_list.append(arc_w)
        marker_indices_list.append(torch.full((num_envs,), midx, dtype=torch.int32, device=device))

    # ------------------------------------------------------------------
    # 2) Current EE yaw indicator (yellow): EE orientation in socket frame.
    # ------------------------------------------------------------------
    rel_q = quat_mul(quat_conjugate(fixed_quat), ee_quat)
    _, _, rel_yaw = euler_xyz_from_quat(rel_q)   # (N,) radians
    cos_y = torch.cos(rel_yaw)
    sin_y = torch.sin(rel_yaw)
    for si in range(1, _YAW_IND_SAMPLES + 1):
        r        = arc_r * si / _YAW_IND_SAMPLES
        local_pt = torch.stack(
            [r * cos_y, r * sin_y, torch.full((num_envs,), arc_z, device=device)],
            dim=1,
        )
        _, ind_w = torch_utils.tf_combine(fixed_quat, fixed_pos_w, ident_q, local_pt)
        translations_list.append(ind_w)
        marker_indices_list.append(torch.full((num_envs,), _IDX_CURR, dtype=torch.int32, device=device))

    # ------------------------------------------------------------------
    # 3) Policy-commanded target yaw indicator (cyan).
    #    Decode: action ∈ [-1,+1] → yaw ∈ [-180°, +90°].
    # ------------------------------------------------------------------
    if raw_actions is not None and raw_actions.shape[-1] > 5:
        yaw_act = raw_actions[:, 5].to(device)   # (N,) ∈ [-1, +1]
        cmd_yaw = math.radians(-180.0) + math.radians(270.0) * (yaw_act + 1.0) / 2.0
        cos_c   = torch.cos(cmd_yaw)
        sin_c   = torch.sin(cmd_yaw)
        cmd_z   = arc_z + 0.006   # draw slightly above the current-yaw line to separate visually
        for si in range(1, _YAW_IND_SAMPLES + 1):
            r        = arc_r * si / _YAW_IND_SAMPLES
            local_pt = torch.stack(
                [r * cos_c, r * sin_c, torch.full((num_envs,), cmd_z, device=device)],
                dim=1,
            )
            _, cmd_w = torch_utils.tf_combine(fixed_quat, fixed_pos_w, ident_q, local_pt)
            translations_list.append(cmd_w)
            marker_indices_list.append(torch.full((num_envs,), _IDX_CMD, dtype=torch.int32, device=device))

    translations   = torch.cat(translations_list,   dim=0)
    marker_indices = torch.cat(marker_indices_list, dim=0)
    identity_q     = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(len(translations), -1)
    markers.visualize(translations=translations, orientations=identity_q, marker_indices=marker_indices)


def _print_yaw_status(base_env, raw_actions: torch.Tensor, step: int):
    """Print per-env yaw info every 30 steps."""
    if step % 30 != 0:
        return

    fixed_quat = base_env._fixed_asset.data.root_quat_w
    ee_quat    = base_env.fingertip_midpoint_quat

    # Relative yaw of EE in socket frame.
    rel_q = quat_mul(quat_conjugate(fixed_quat[0:1]), ee_quat[0:1])
    _, _, rel_yaw_t = euler_xyz_from_quat(rel_q)
    curr_yaw_deg = math.degrees(rel_yaw_t.item())
    in_range     = -180.0 <= curr_yaw_deg <= 90.0
    status       = "OK" if in_range else "*** DEAD ZONE ***"

    cmd_str = ""
    if raw_actions is not None and raw_actions.shape[-1] > 5:
        yaw_act      = raw_actions[0, 5].item()
        cmd_yaw_deg  = -180.0 + 270.0 * (yaw_act + 1.0) / 2.0
        cmd_str      = f"  cmd_yaw={cmd_yaw_deg:+.1f}° (act={yaw_act:+.3f})"

    print(
        f"[yaw step {step:5d}]  EE yaw (socket frame): {curr_yaw_deg:+.1f}°  {status}{cmd_str}"
    )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Play with RL-Games agent + real-time Franka yaw dead-zone visualization."""
    task_name       = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device     = args_cli.device    if args_cli.device    is not None else env_cfg.sim.device

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    env_cfg.seed = agent_cfg["params"]["seed"]

    log_root_path = os.path.abspath(os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"]))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if args_cli.checkpoint is None:
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        checkpoint_file = ".*" if args_cli.use_last_checkpoint else f"{agent_cfg['params']['config']['name']}.pth"
        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)

    log_dir         = os.path.dirname(os.path.dirname(resume_path))
    env_cfg.log_dir = log_dir

    rl_device         = agent_cfg["params"]["config"]["device"]
    clip_obs          = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions      = agent_cfg["params"]["env"].get("clip_actions",      math.inf)
    obs_groups        = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs),
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"]       = resume_path
    print(f"[INFO] Loading model checkpoint from: {resume_path}")

    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    dt       = env.unwrapped.step_dt
    base_env = env.unwrapped

    print(
        "\n[YAW VIS]  GREEN ring  = reachable [-180°, +90°]\n"
        "           RED ring    = dead zone  (+90°, +180°]\n"
        "           YELLOW dots = current EE yaw (socket frame)\n"
        "           CYAN dots   = policy-commanded target yaw\n"
    )

    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]

    ever_succeeded  = None
    total_episodes  = 0
    total_successes = 0
    step            = 0
    last_actions    = None

    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    while simulation_app.is_running():
        start_time = time.time()

        with torch.inference_mode():
            obs          = agent.obs_to_torch(obs)
            actions      = agent.get_action(obs, is_deterministic=agent.is_deterministic)
            obs, _, dones, _ = env.step(actions)
            last_actions = actions

            # Success tracking
            if hasattr(base_env, "_get_curr_successes") and hasattr(base_env, "cfg_task"):
                check_rot = base_env.cfg_task.name == "nut_thread"
                true_succ = base_env._get_curr_successes(
                    success_threshold=base_env.cfg_task.success_threshold,
                    check_rot=check_rot,
                )
                n_envs = base_env.num_envs
                if ever_succeeded is None:
                    ever_succeeded = torch.zeros(n_envs, dtype=torch.bool, device=true_succ.device)
                ever_succeeded |= true_succ

                if actions.shape[-1] > 6:
                    pred_val = ((actions[:, 6] + 1) / 2).mean().item()
                    pred_str = f"pred={pred_val:.3f}"
                else:
                    pred_str = "pred=N/A"
                n_ever = ever_succeeded.sum().item()
                print(
                    f"[play]  {pred_str}  |  "
                    f"true={true_succ.sum().item()}/{n_envs}  "
                    f"ever={n_ever}/{n_envs}({n_ever/n_envs:.1%})"
                )

            # Episode resets
            if len(dones) > 0:
                reset_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                if len(reset_ids) > 0 and ever_succeeded is not None:
                    total_successes += ever_succeeded[reset_ids].sum().item()
                    total_episodes  += len(reset_ids)
                    print(
                        f"[EVAL]  episodes={total_episodes}  "
                        f"successes={int(total_successes)}  "
                        f"rate={total_successes/total_episodes:.1%}"
                    )
                    ever_succeeded[reset_ids] = False
                if agent.is_rnn and agent.states is not None:
                    for s in agent.states:
                        s[:, dones, :] = 0.0

            # Yaw visualization (3D markers + periodic print)
            if hasattr(base_env, "_fixed_asset"):
                _draw_yaw_arc(base_env, last_actions)
                _print_yaw_status(base_env, last_actions, step)

        step += 1
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    try:
        simulation_app.close()
    except Exception:
        pass
