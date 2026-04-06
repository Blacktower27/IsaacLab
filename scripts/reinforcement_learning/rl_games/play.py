# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rl_games_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--record_csv",
    action="store_true",
    default=False,
    help="If set, record EE (link_ee) pose, joint states, and box pose to a CSV file named after the task.",
)
parser.add_argument(
    "--record_env_idx",
    type=int,
    default=0,
    help="Environment index to record (default: 0).",
)
parser.add_argument(
    "--max_completed_episodes",
    type=int,
    default=None,
    help=(
        "If set, stop play after this many completed vector-env episodes (sum across all envs). "
        "Useful for headless batch evaluation. Default: run until the app is closed."
    ),
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import csv
import math
import os
import random
import time

import gymnasium as gym
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
import isaacsim.core.utils.torch as torch_utils

from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.math import quat_apply_inverse

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Play with RL-Games agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]
    # set the environment seed (after multi-gpu config for updated rank from agent seed)
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg["params"]["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # find checkpoint
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rl_games", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint is None:
        # specify directory for logging runs
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        # specify name of checkpoint
        if args_cli.use_last_checkpoint:
            checkpoint_file = ".*"
        else:
            # this loads the best checkpoint
            checkpoint_file = f"{agent_cfg['params']['config']['name']}.pth"
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # wrap around environment for rl-games
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)
    obs_groups = agent_cfg["params"]["env"].get("obs_groups")
    concate_obs_groups = agent_cfg["params"]["env"].get("concate_obs_groups", True)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions, obs_groups, concate_obs_groups)

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # load previously trained model
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    # create runner from rl-games
    runner = Runner()
    runner.load(agent_cfg)
    # obtain the agent from the runner
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    dt = env.unwrapped.step_dt

    # --- CSV recording setup ---
    csv_file = None
    csv_writer = None
    episode_start_time = None  # tracks sim_time at the start of each episode
    # Find link_ee body index once (used for EE pose recording).
    _link_ee_idx = None
    if args_cli.record_csv:
        _robot = env.unwrapped._robot
        if "link_ee" in _robot.body_names:
            _link_ee_idx = _robot.body_names.index("link_ee")
        else:
            print("[WARN] link_ee not found in robot body_names; falling back to fingertip_midpoint.")
    if args_cli.record_csv:
        import datetime
        task_short = args_cli.task.split(":")[-1] if args_cli.task else "task"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.abspath(os.path.join("logs", "trajectories", f"{task_short}_{timestamp}.csv"))
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "episode_time",
            "ee_x", "ee_y", "ee_z", "ee_roll", "ee_pitch", "ee_yaw",
            "A1", "A2", "A3", "A4", "A5", "A6", "A7",
            "box_x", "box_y", "box_z", "box_roll", "box_pitch", "box_yaw",
            "episode_success",
        ])
        print(f"[INFO] Recording trajectory to: {csv_path}  (env_idx={args_cli.record_env_idx})")

    # reset environment
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    timestep = 0
    # success rate tracking: per-env flag within current episode, plus cumulative stats
    ever_succeeded = None
    total_episodes = 0
    total_successes = 0
    # required: enables the flag for batched observations
    _ = agent.get_batch_size(obs, 1)
    # initialize RNN states if used
    if agent.is_rnn:
        agent.init_rnn()
    # simulate environment
    # note: We simplified the logic in rl-games player.py (:func:`BasePlayer.run()`) function in an
    #   attempt to have complete control over environment stepping. However, this removes other
    #   operations such as masking that is used for multi-agent learning by RL-Games.
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # convert obs to agent format
            obs = agent.obs_to_torch(obs)
            # agent stepping
            actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
            # env stepping
            obs, _, dones, _ = env.step(actions)

            # Print success prediction and true geometric success each step.
            base_env = env.unwrapped

            # --- CSV recording ---
            if csv_writer is not None:
                i = args_cli.record_env_idx
                robot = base_env._robot
                scene = base_env.scene

                # Robot base pose (world frame)
                robot_base_pos_w = robot.data.root_pos_w[i]   # [3]
                robot_base_quat_w = robot.data.root_quat_w[i]  # [4] wxyz

                # EE: use link_ee (robot flange) body pose in world frame.
                if _link_ee_idx is not None:
                    ee_pos_w = robot.data.body_pos_w[i, _link_ee_idx]
                    ee_quat_w = robot.data.body_quat_w[i, _link_ee_idx]
                    ee_pos_env = ee_pos_w - scene.env_origins[i]
                else:
                    ee_pos_env = base_env.fingertip_midpoint_pos[i]
                    ee_quat_w = base_env.fingertip_midpoint_quat[i]

                # Robot base position in same env-local frame
                robot_base_pos_env = robot_base_pos_w - scene.env_origins[i]

                # EE displacement from robot base (world orientation), then rotate into base frame
                ee_pos_rel = (ee_pos_env - robot_base_pos_env).unsqueeze(0)
                ee_pos_base = quat_apply_inverse(robot_base_quat_w.unsqueeze(0), ee_pos_rel).squeeze(0)
                # EE orientation in robot base frame: q_base^* ⊗ q_world
                ee_quat_base = torch_utils.quat_mul(
                    torch_utils.quat_conjugate(robot_base_quat_w.unsqueeze(0)),
                    ee_quat_w.unsqueeze(0),
                ).squeeze(0)
                ee_r, ee_p, ee_y = torch_utils.get_euler_xyz(ee_quat_base.unsqueeze(0))

                # Joint positions (7 DOF)
                joint_pos_np = base_env.joint_pos[i, :7].cpu().numpy()

                # Box: fixed_pos is (pos_world - env_origin), fixed_quat is world orientation
                box_pos_env = base_env.fixed_pos[i]
                box_quat_w = base_env.fixed_quat[i]
                box_pos_rel = (box_pos_env - robot_base_pos_env).unsqueeze(0)
                box_pos_base = quat_apply_inverse(robot_base_quat_w.unsqueeze(0), box_pos_rel).squeeze(0)
                box_quat_base = torch_utils.quat_mul(
                    torch_utils.quat_conjugate(robot_base_quat_w.unsqueeze(0)),
                    box_quat_w.unsqueeze(0),
                ).squeeze(0)
                box_r, box_p, box_yaw = torch_utils.get_euler_xyz(box_quat_base.unsqueeze(0))

                sim_time = float(robot._data._sim_timestamp)
                # Reset episode clock on first step or when this env was done/reset
                i_done = bool(dones[i].item()) if torch.is_tensor(dones[i]) else bool(dones[i])
                if episode_start_time is None or i_done:
                    episode_start_time = sim_time
                episode_time = sim_time - episode_start_time

                ee_xyz = ee_pos_base.cpu().numpy()
                box_xyz = box_pos_base.cpu().numpy()
                ep_succ = int(ever_succeeded[i].item()) if ever_succeeded is not None else 0
                csv_writer.writerow([
                    f"{episode_time:.6f}",
                    f"{ee_xyz[0]:.6f}", f"{ee_xyz[1]:.6f}", f"{ee_xyz[2]:.6f}",
                    f"{float(ee_r[0]):.6f}", f"{float(ee_p[0]):.6f}", f"{float(ee_y[0]):.6f}",
                    *[f"{v:.6f}" for v in joint_pos_np],
                    f"{box_xyz[0]:.6f}", f"{box_xyz[1]:.6f}", f"{box_xyz[2]:.6f}",
                    f"{float(box_r[0]):.6f}", f"{float(box_p[0]):.6f}", f"{float(box_yaw[0]):.6f}",
                    ep_succ,
                ])
            if actions.shape[-1] > 6:
                success_pred = (actions[:, 6] + 1) / 2
                pred_str = (f"pred={success_pred.mean().item():.3f} "
                            f"(min={success_pred.min().item():.3f} max={success_pred.max().item():.3f})")
            else:
                pred_str = "pred=N/A"
            if hasattr(base_env, "_get_curr_successes") and hasattr(base_env, "cfg_task"):
                check_rot = base_env.cfg_task.name == "nut_thread"
                true_succ = base_env._get_curr_successes(
                    success_threshold=base_env.cfg_task.success_threshold, check_rot=check_rot
                )
                n_envs = base_env.num_envs
                # accumulate per-env ever-succeeded flag
                if ever_succeeded is None:
                    ever_succeeded = torch.zeros(n_envs, dtype=torch.bool, device=true_succ.device)
                ever_succeeded |= true_succ
                n_ever = ever_succeeded.sum().item()
                rate = n_ever / n_envs
                succ_str = f"true={true_succ.sum().item()}/{n_envs}  ever={n_ever}/{n_envs}({rate:.2%})"
            else:
                succ_str = "true=N/A"
            print(f"[play] {pred_str}  |  {succ_str}")

            # perform operations for terminated episodes
            if len(dones) > 0:
                reset_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                if len(reset_ids) > 0 and ever_succeeded is not None:
                    total_successes += ever_succeeded[reset_ids].sum().item()
                    total_episodes += len(reset_ids)
                    rate = total_successes / total_episodes
                    print(
                        f"[EVAL] episodes={total_episodes}  successes={int(total_successes)}"
                        f"  success_rate={rate:.2%}"
                    )
                    ever_succeeded[reset_ids] = False  # reset for next episode
                # reset rnn state for terminated episodes
                if agent.is_rnn and agent.states is not None:
                    for s in agent.states:
                        s[:, dones, :] = 0.0
        if args_cli.video:
            timestep += 1
            # exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

        if args_cli.max_completed_episodes is not None and total_episodes >= args_cli.max_completed_episodes:
            rate = (total_successes / total_episodes) if total_episodes > 0 else 0.0
            print(
                f"[EVAL] STOP max_completed_episodes={args_cli.max_completed_episodes}  "
                f"episodes={total_episodes}  successes={int(total_successes)}  success_rate={rate:.2%}"
            )
            break

    # close CSV file if recording
    if csv_file is not None:
        csv_file.close()
        print(f"[INFO] Trajectory saved to: {csv_path}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    try:
        simulation_app.close()
    except Exception:
        pass
    if args_cli.max_completed_episodes is not None:
        import os
        os._exit(0)
