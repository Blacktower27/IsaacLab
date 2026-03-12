# Box-Lid Insertion Task (FORGE / IsaacLab)

A robot grasps a yellow snap-fit lid and inserts it into a small box.

| Robot | Gym ID |
|---|---|
| Franka Panda | `Isaac-Forge-BoxLidInsert-Direct-v0` |
| Kuka iiwa7 | `Isaac-Forge-BoxLidInsert-Kuka-Direct-v0` |

---

## Quick-Start Commands

All commands run from the **IsaacLab root directory**.

### Train

```bash
# Franka
python scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Forge-BoxLidInsert-Direct-v0 \
    --headless

# Kuka iiwa7
python scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Forge-BoxLidInsert-Kuka-Direct-v0 \
    --headless

# Override number of environments
python scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Forge-BoxLidInsert-Direct-v0 \
    --headless --num_envs 256

# Resume from checkpoint
python scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Forge-BoxLidInsert-Direct-v0 \
    --headless --checkpoint logs/rl_games/Forge/<timestamp>/nn/Forge.pth
```

Logs and checkpoints are saved to `logs/rl_games/Forge/<timestamp>/`.
View TensorBoard: `tensorboard --logdir logs/rl_games/Forge/`

Key metrics: `successes`, `logs_rew_curr_engaged`, `logs_rew_curr_success`, `logs_rew_kp_coarse`.

---

### Play (Evaluate a Trained Policy)

To run a trained checkpoint, pass the **exact path** to the `.pth` file via `--checkpoint`.
The checkpoint file is typically located at:
```
logs/rl_games/Forge/<timestamp>/nn/Forge.pth
```

```bash
# Franka — run a specific checkpoint (replace <timestamp> with your run folder)
python scripts/reinforcement_learning/rl_games/play.py \
    --task Isaac-Forge-BoxLidInsert-Direct-v0 \
    --num_envs 16 \
    --checkpoint logs/rl_games/Forge/<timestamp>/nn/Forge.pth

# Kuka iiwa7 — run a specific checkpoint
python scripts/reinforcement_learning/rl_games/play.py \
    --task Isaac-Forge-BoxLidInsert-Kuka-Direct-v0 \
    --num_envs 16 \
    --checkpoint logs/rl_games/Forge/<timestamp>/nn/Forge.pth

# Headless evaluation (no GUI, faster)
python scripts/reinforcement_learning/rl_games/play.py \
    --task Isaac-Forge-BoxLidInsert-Direct-v0 \
    --num_envs 128 \
    --headless \
    --checkpoint logs/rl_games/Forge/<timestamp>/nn/Forge.pth
```

> **Note:** `--task` must match the task used during training.
> A Franka checkpoint cannot be loaded for the Kuka task and vice versa.

Each step prints:
```
[play] pred=N/A  |  true=3/16  ever=10/16(62.50%)
[EVAL] episodes=16  successes=10  success_rate=62.50%
```
- `true` — envs currently at a successful pose this step.
- `ever` — envs that succeeded at least once this episode.
- `success_rate` — cumulative across all completed episodes.

---

### Visualise with Zero Agent
Sends zero actions every step — useful to inspect randomisation and resets without a trained policy.
```bash
# Franka
python scripts/environments/zero_agent.py \
    --task Isaac-Forge-BoxLidInsert-Direct-v0 --num_envs 1

# Kuka
python scripts/environments/zero_agent.py \
    --task Isaac-Forge-BoxLidInsert-Kuka-Direct-v0 --num_envs 1
```

### Visualise Success Check (Interactive)
Lets you **manually move the lid with keyboard** to verify the success/engage geometry in the simulator.
Coloured sphere markers show clip tooth positions (blue) and hole targets (red); they overlap at the success pose.

```bash
# Recommended: freeze box at default orientation for a clean view
./isaaclab.sh -p scripts/environments/visualize_box_lid_success.py --num_envs 1 --freeze_box

# With box randomisation enabled
./isaaclab.sh -p scripts/environments/visualize_box_lid_success.py --num_envs 1
```

**Keyboard controls** (click the viewport once to give it focus):

| Key | Action |
|---|---|
| Arrow Up / Down | Move lid +Y / -Y |
| Arrow Left / Right | Move lid -X / +X |
| Q / E | Move lid +Z / -Z |
| I / K | Pitch +/- |
| J / L | Yaw +/- |
| U / O | Roll +/- |
| Hold Shift | 5× speed |
| R | Reset lid to exact success position |

Terminal prints clip positions, distance to hole targets, and rolling success/engaged rates every 10 steps.

---

## Key Parameters

### Episode / Simulation
| File | Parameter | Default | Notes |
|---|---|---|---|
| `forge_env_cfg.py` | `episode_length_s` | `15.0` | Seconds per episode |
| `factory_env_cfg.py` | `obs_window_size` | `64` | Timesteps stacked for policy input |
| `factory_env_cfg.py` | `state_window_size` | `64` | Timesteps stacked for critic input |
| `factory_env_cfg.py` | `scene.num_envs` | `128` | Parallel environments |

`obs_window_size = 1` → no stacking (Markovian). Larger = more context, slower training.

---

### Force/Torque Sensor in Observation

Controlled by `obs_order` in `forge_env_cfg.py → ForgeEnvCfg`:

```python
obs_order: list = [
    "fingertip_pos_rel_fixed",   # 3-dim
    "fingertip_quat",            # 4-dim
    "ee_linvel",                 # 3-dim
    "ee_angvel",                 # 3-dim
    "ft_force",                  # 3-dim  ← F/T sensor (currently ON)
    # "force_threshold",         # 1-dim  ← currently OFF
]
```

To **disable** F/T input, comment out `"ft_force"`:
```python
obs_order: list = [
    "fingertip_pos_rel_fixed",
    "fingertip_quat",
    "ee_linvel",
    "ee_angvel",
    # "ft_force",
]
```

---

### <span style="color:red">Important!</span> Robot Initial Position (box-local frame)

`hand_init_pos` is the fallback starting pose when random initialisation is disabled.
It is defined in **`forge_tasks_cfg.py → ForgeBoxLidInsert`** (line ~162):

```python
# forge_tasks_cfg.py  →  class ForgeBoxLidInsert
hand_init_pos: list = [0.00, 0.07, 0.034]   # Kuka iiwa7
# hand_init_pos: list = [0.00, 0.07, 0.043]  # Franka Panda
```

- **Franka Panda** — use `[0.00, 0.07, 0.043]`
- **Kuka iiwa7** — use `[0.00, 0.07, 0.034]` (current default at line 162)

There is only one `hand_init_pos` field shared by both robots; switch the value to match whichever
robot you are training.

The other random-init range parameters in the same class:

| Parameter | Default | Effect |
|---|---|---|
| `hand_init_x_range` | `[-0.05, 0.05]` | ±50 mm lateral range above box |
| `hand_init_y_range` | `[0.0, 0.1]` | 0–100 mm back from box centre |
| `hand_init_z_range` | `[0.055, 0.085]` | 55–85 mm above box top face |
| `hand_init_yaw_noise_deg` | `20.0` | ±yaw noise on top of box-aligned yaw |
| `hand_init_pitch_noise_deg` | `20.0` | ±pitch noise |

XY are in the box-local frame and rotate with box yaw. Gripper yaw is auto-aligned to the box, then noise is added on top.

---

### Box (Fixed Asset) Randomisation
| Parameter | Default | Effect |
|---|---|---|
| `fixed_asset_init_pos_noise` | `[0.05, 0.05, 0.0]` | ±50 mm XY table jitter |
| `fixed_asset_init_orn_range_deg` | `360.0` | Full 360° yaw randomisation |

Set `fixed_asset_init_orn_range_deg = 0` to fix the box orientation (easiest, not recommended for real deployment).

---

### Lid-in-Gripper Grasp Pose
| Parameter | Default | Effect |
|---|---|---|
| `held_asset_rot_init` | `90.0` | Base yaw of lid in flipped-fingertip frame (deg) |
| `held_asset_rot_offset` | `[0.0, 35.0, 0.0]` | Extra [roll, pitch, yaw] offset (deg) |
| `held_asset_pos_offset` | `[0.0, 0.02, 0.005]` | Fine-tune translation in fingertip frame (m) |
| `held_asset_pos_noise` | `[0.003, 0.003, 0.003]` | Grasp position uncertainty (m) |

---

### Keypoints
Two phases control which keypoints are active:

**Phase 1 (before first success):** 2 fixed clip-tooth keypoints + `num_reset_extra_kp` front-face points.
**Phase 2 (after first success):** 2 clip-tooth keypoints + `num_success_extra_kp` random body points.

| Parameter | Default | Effect |
|---|---|---|
| `num_reset_extra_kp` | `10` | Dense guidance during approach/alignment |
| `num_success_extra_kp` | `100` | Dense guidance during pressing phase |

Buffer size is automatically `2 + max(num_reset_extra_kp, num_success_extra_kp)`.

---

### Success & Engage Thresholds

- **Success** (`success_threshold`): clip geometry check — both clip teeth must be fully seated in the box pockets:
  ```
  |clip_X - pocket_centre_X| < 2 mm
  |clip_Y - front_wall_Y   | < 4 mm
  21 mm < clip_Z < 29 mm
  ```
- **Engage** (`engage_threshold`): intermediate reward — lid base within 27 mm of target Z AND within 2.5 mm XY (lid body entering the cavity).

---

### PPO Hyperparameters
Edit `forge/agents/rl_games_ppo_cfg.yaml`:

| Parameter | Default | Notes |
|---|---|---|
| `max_epochs` | `400` | Training epochs |
| `horizon_length` | `128` | Steps per env before gradient update |
| `minibatch_size` | `512` | Minibatch size |
| `learning_rate` | `1e-4` | Adaptive schedule based on KL |
| `gamma` | `0.995` | Discount factor |
| LSTM `units` | `1024` | Hidden size (2-layer LSTM before MLP) |

---

## Kuka iiwa7 Notes

The Kuka variant (`Isaac-Forge-BoxLidInsert-Kuka-Direct-v0`) differs from the Franka variant in several ways:

### Lid embedded in URDF
The lid (`Lid_Yellow`) is attached as a fixed link (`link_lid`) directly in the Kuka URDF, rather than being a separate scene object that the robot grasps. This means:
- No separate `held_asset` is spawned in the scene.
- The lid physically participates in arm dynamics and feels insertion forces through the robot joints.
- `ForgeKukaEventCfg` disables `object_scale_mass` and `held_physics_material` (not applicable).

### Body names
| Role | Franka body | Kuka body |
|---|---|---|
| Fingertip / EE tip | `panda_hand` (or similar) | `link_tcp` |
| Force sensor | `panda_hand` | `link_ee` |
| Held asset body | *(separate object)* | `link_lid` (in URDF) |

### Joint configuration
Initial joint angles at reset (`ForgeKukaCtrlCfg.reset_joints`):

| Joint | Value (rad) |
|---|---|
| A1 | 0.0 |
| A2 | 0.3 |
| A3 | 0.0 |
| A4 | -1.5 |
| A5 | 0.0 |
| A6 | 1.2 |
| A7 | 0.0 |

### Config class hierarchy (Kuka)
```
ForgeKukaBoxLidInsertCfg          ← forge_env_cfg.py  (robot=Kuka URDF, ctrl/events overridden)
  └─ ForgeTaskBoxLidInsertCfg     ← forge_env_cfg.py  (assets, scene)
       └─ ForgeBoxLidInsert       ← forge_tasks_cfg.py (hand_init_pos, thresholds, keypoints)
```

---

## File Map

```
scripts/reinforcement_learning/rl_games/train.py   ← Training
scripts/reinforcement_learning/rl_games/play.py    ← Evaluation
scripts/environments/zero_agent.py                 ← Zero-action visualisation

source/isaaclab_tasks/isaaclab_tasks/direct/
├── forge/
│   ├── __init__.py            ← Gym registration (Franka + Kuka IDs)
│   ├── forge_env.py           ← ForgeEnv (F/T sensor, obs noise)
│   ├── forge_env_cfg.py       ← ForgeTaskBoxLidInsertCfg (Franka)
│   │                             ForgeKukaBoxLidInsertCfg (Kuka)
│   │                             ForgeKukaCtrlCfg, ForgeKukaEventCfg
│   ├── forge_tasks_cfg.py     ← ForgeBoxLidInsert (hand_init_pos line ~162)
│   │                             SmallBoxCfg, LidYellowCfg
│   └── agents/rl_games_ppo_cfg.yaml  ← PPO config (shared by both robots)
└── factory/
    ├── factory_env.py         ← Reset logic, rewards, success checks
    ├── factory_env_cfg.py     ← Window sizes, sim params, robot config
    └── factory_utils.py       ← Keypoint / pose utilities

source/isaaclab_assets/isaaclab_assets/custom_assets/
├── box/middle/
│   ├── Small_Box.usd          ← Fixed box (kinematic)
│   └── Lid_Yellow.usd         ← Held lid (Franka only; Kuka embeds it in URDF)
└── robots/lbr_description/urdf/kuka_blue/
    └── kuka_blue.urdf         ← Kuka iiwa7 URDF (includes link_lid)
```

---
