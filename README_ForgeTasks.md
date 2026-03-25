# FORGE Custom Insertion Tasks (IsaacLab)

Three custom precision-insertion tasks built on the FORGE framework, each with Franka Panda and Kuka iiwa7 variants.

| Task | Description | Episode | Gym IDs |
|---|---|---|---|
| **BoxLidInsert** | Snap-fit lid onto a small box | 15 s | `Isaac-Forge-BoxLidInsert-Direct-v0` / `-Kuka-` |
| **RJ45Insert** | RJ45 male plug into female socket | 10 s | `Isaac-Forge-RJ45Insert-Direct-v0` / `-Kuka-` |
| **BNCSmallInsert** | BNC coaxial male plug into female socket | 10 s | `Isaac-Forge-BNCSmallInsert-Direct-v0` / `-Kuka-` |

Full gym IDs:

| Task | Robot | Gym ID |
|---|---|---|
| BoxLidInsert | Franka Panda | `Isaac-Forge-BoxLidInsert-Direct-v0` |
| BoxLidInsert | Kuka iiwa7 | `Isaac-Forge-BoxLidInsert-Kuka-Direct-v0` |
| RJ45Insert | Franka Panda | `Isaac-Forge-RJ45Insert-Direct-v0` |
| RJ45Insert | Kuka iiwa7 | `Isaac-Forge-RJ45Insert-Kuka-Direct-v0` |
| BNCSmallInsert | Franka Panda | `Isaac-Forge-BNCSmallInsert-Direct-v0` |
| BNCSmallInsert | Kuka iiwa7 | `Isaac-Forge-BNCSmallInsert-Kuka-Direct-v0` |

---

## Quick-Start Commands

All commands run from the **IsaacLab root directory**. Replace `<TASK_ID>` with any gym ID above.

### Train

```bash
# Basic training (headless)
python scripts/reinforcement_learning/rl_games/train.py \
    --task <TASK_ID> \
    --headless

# Override number of environments
python scripts/reinforcement_learning/rl_games/train.py \
    --task <TASK_ID> \
    --headless --num_envs 256

# Resume from checkpoint
python scripts/reinforcement_learning/rl_games/train.py \
    --task <TASK_ID> \
    --headless --checkpoint logs/rl_games/Forge/<timestamp>/nn/Forge.pth
```

Logs and checkpoints: `logs/rl_games/Forge/<timestamp>/`
TensorBoard: `tensorboard --logdir logs/rl_games/Forge/`

Key metrics: `successes`, `logs_rew_curr_engaged`, `logs_rew_curr_success`, `logs_rew_kp_coarse`.

---

### Play (Evaluate a Trained Policy)

```bash
# Run a trained checkpoint
python scripts/reinforcement_learning/rl_games/play.py \
    --task <TASK_ID> \
    --num_envs 16 \
    --checkpoint logs/rl_games/Forge/<timestamp>/nn/Forge.pth

# Headless evaluation (faster)
python scripts/reinforcement_learning/rl_games/play.py \
    --task <TASK_ID> \
    --num_envs 128 \
    --headless \
    --checkpoint logs/rl_games/Forge/<timestamp>/nn/Forge.pth
```

> **Note:** `--task` must match the task used during training. Franka and Kuka checkpoints are not interchangeable.

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
python scripts/environments/zero_agent.py \
    --task <TASK_ID> --num_envs 1
```

### Visualise BoxLidInsert Success Check (Interactive)

Lets you **manually move the lid with keyboard** to verify the success/engage geometry in the simulator.
Coloured sphere markers show clip tooth positions (blue) and hole targets (red); they overlap at the success pose.

```bash
# Freeze box at default orientation for a clean view
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

## Task Details

### BoxLidInsert

Robot grasps a yellow snap-fit lid and inserts it into a small box (120×100×30 mm).

**Assets:**
- Fixed: `Small_Box.usd` — 120×100×30 mm, Z origin at base, kinematic RigidObject
- Held (Franka only): `Lid_Yellow.usd` — snap-fit lid with 17 mm handle; Kuka embeds it as `link_lid`

**Success:** Clip-based snap engagement — both clip teeth fully seated in box pockets:
```
|clip_X - pocket_centre_X| < 2 mm
|clip_Y - front_wall_Y   | < 4 mm
21 mm < clip_Z < 29 mm
```

**Engage:** Lid base within 27 mm of assembled Z (`engage_threshold=0.9`) AND within 2.5 mm XY.

**Key parameters** (`forge_tasks_cfg.py → ForgeBoxLidInsert`):

| Parameter | Default | Effect |
|---|---|---|
| `hand_init_pos` | `[0.00, 0.07, 0.034]` (Kuka) / `[0.00, 0.07, 0.043]` (Franka) | Fallback start pose above box |
| `hand_init_x_range` | `[-0.05, 0.05]` | ±50 mm lateral range |
| `hand_init_y_range` | `[0.0, 0.1]` | 0–100 mm back from box centre |
| `hand_init_z_range` | `[0.055, 0.085]` | 55–85 mm above box top |
| `hand_init_yaw_noise_deg` | `20.0` | ±yaw noise on top of box-aligned yaw |
| `fixed_asset_init_orn_range_deg` | `360.0` | Full 360° box yaw randomisation |
| `num_reset_extra_kp` | `10` | Extra front-face keypoints at reset |
| `num_success_extra_kp` | `100` | Dense keypoints added on first success |

> There is one `hand_init_pos` field shared by both robots. Switch the value to match whichever robot you are training.

---

### RJ45Insert

Robot holds an RJ45 male plug by its cable end and inserts it downward into an RJ45 female socket.

**Assets:**
- Fixed: `Adi_Medium_RJ45_Female_Raw_Colored.usd` — socket opening at sim_Z = +29.22 mm from USD origin; placed at `pos.z=0.02223` so the base is flush with the table
- Held (Franka only): `Adi_Medium_RJ45_Male_Raw_Colored.usd` — connector tip at sim_Z = −13.98 mm from origin; Kuka embeds it as `link_rj45`

**Success** (`success_threshold=-0.24`): Connector tip ≥ 7 mm inside socket opening AND within 3 mm XY (~half-insertion of the 14 mm connector head).

**Engage** (`engage_threshold=2.0`): Plug XY-aligned (tip < 4 mm from opening centre) AND correctly oriented (yaw < 20°, tilt < 15°), regardless of height.

**Key parameters** (`forge_tasks_cfg.py → ForgeRJ45Insert`):

| Parameter | Default | Effect |
|---|---|---|
| `hand_init_pos` | `[0.00, 0.00, 0.05]` | 50 mm above socket opening |
| `hand_init_x_range` | `[-0.06, 0.06]` | ±60 mm from socket axis |
| `hand_init_y_range` | `[-0.06, 0.06]` | ±60 mm from socket axis |
| `hand_init_z_range` | `[0.015, 0.12]` | 15–120 mm above socket |
| `hand_init_yaw_noise_deg` | `30.0` | ±30° yaw noise |
| `fixed_asset_init_orn_range_deg` | `360.0` | Full 360° socket yaw randomisation |

---

### BNCSmallInsert

Robot holds a BNC coaxial male plug and inserts it downward into a BNC female socket (cylindrically symmetric).

**Assets:**
- Fixed: `Adi_BNC_Simulation_Small_Female.usd` — 44×44×60 mm socket; USD origin at 35 mm above base; placed at `pos.z=0.035`
- Held (Franka only): `Adi_BNC_Simulation_Small_Male.usd` — USD origin 36.235 mm below tip; Kuka embeds it as `link_bnc`

**Success** (`success_threshold=-0.3`): Connector tip 7.5 mm inside socket (0.025 × 0.3 = 7.5 mm) AND within 3 mm XY.

**Engage** (`engage_threshold=2.0`): Same XY + yaw + tilt alignment check as RJ45Insert.

**Key parameters** (`forge_tasks_cfg.py → ForgeBNCSmallInsert`):

| Parameter | Default | Effect |
|---|---|---|
| `hand_init_pos` | `[0.00, 0.00, 0.05]` | 50 mm above socket opening |
| `hand_init_x_range` | `[-0.06, 0.06]` | ±60 mm from socket axis |
| `hand_init_y_range` | `[-0.06, 0.06]` | ±60 mm from socket axis |
| `hand_init_z_range` | `[0.015, 0.12]` | 15–120 mm above socket |
| `hand_init_yaw_noise_deg` | `30.0` | ±30° yaw noise |
| `fixed_asset_init_orn_range_deg` | `360.0` | 360° yaw (cylindrically symmetric) |

---

## Shared Parameters

### Episode / Simulation

| File | Parameter | Default | Notes |
|---|---|---|---|
| `forge_env_cfg.py` | `episode_length_s` | per-task | 15 s (BoxLid), 10 s (RJ45, BNC) |
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
    # "ft_force",                # 3-dim  ← currently OFF
    # "force_threshold",         # 1-dim  ← currently OFF
]
```

To **enable** F/T input, uncomment `"ft_force"`.

---

### PPO Hyperparameters

Edit `forge/agents/rl_games_ppo_cfg.yaml` (shared by all tasks):

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

For all Kuka variants, the connector/held object is embedded as a **fixed link directly in the URDF** rather than being a separate scene object. This means the arm physically feels insertion resistance through its joints.

| Task | Kuka URDF | Embedded link | `held_body_name` |
|---|---|---|---|
| BoxLidInsert | `kuka_blue.urdf` | `link_lid` | `link_lid` |
| RJ45Insert | `kuka_blue_rj45.urdf` | `link_rj45` | `link_rj45` |
| BNCSmallInsert | `kuka_blue_bnc_small.urdf` | `link_bnc` | `link_bnc` |

Because the held object is part of the robot:
- `ForgeKukaEventCfg` disables `object_scale_mass` and `held_physics_material`.
- No separate `held_asset` scene object is spawned.
- `ForgeKukaCtrlCfg.held_body_name` tells `factory_env` which robot body to use for `held_pos`.

### Body names

| Role | Franka body | Kuka body |
|---|---|---|
| Fingertip / EE tip | `panda_hand` | `link_tcp` |
| Force sensor | `panda_hand` | `link_ee` |
| Held object | *(separate scene object)* | task-specific link (see table above) |

### Joint configuration

Initial joint angles at reset (`ForgeKukaCtrlCfg.reset_joints`):

| Joint | Value (rad) |
|---|---|
| A1 | 0.0 |
| A2 | 0.3 |
| A3 | 0.0 |
| A4 | −1.5 |
| A5 | 0.0 |
| A6 | 1.2 |
| A7 | 0.0 |

Null-space target (`default_dof_pos_tensor`) is set to the same values so the null-space controller does not pull the arm toward Franka's home pose.

Torque clamp: `dof_torque_clamp = 200.0 N·m` (matches Kuka iiwa7 URDF `effort` limit).

### Config class hierarchy (Kuka example — RJ45)

```
ForgeKukaRJ45InsertCfg          ← forge_env_cfg.py  (robot=kuka_blue_rj45.urdf)
  ├─ ctrl: ForgeKukaRJ45CtrlCfg ← held_body_name="link_rj45"
  ├─ events: ForgeKukaRJ45EventCfg ← object_scale_mass=None, held_physics_material=None
  └─ ForgeTaskRJ45InsertCfg     ← task assets, episode length
       └─ ForgeRJ45Insert        ← hand_init_pos, thresholds, keypoints
```

---

## File Map

```
scripts/reinforcement_learning/rl_games/train.py    ← Training
scripts/reinforcement_learning/rl_games/play.py     ← Evaluation
scripts/environments/zero_agent.py                  ← Zero-action visualisation
scripts/environments/visualize_box_lid_success.py   ← Interactive BoxLidInsert debug

source/isaaclab_tasks/isaaclab_tasks/direct/
├── forge/
│   ├── __init__.py            ← Gym registration (all 6 IDs)
│   ├── forge_env.py           ← ForgeEnv (F/T sensor, obs noise)
│   ├── forge_env_cfg.py       ← All env configs:
│   │                             ForgeTaskBoxLidInsertCfg / ForgeKukaBoxLidInsertCfg
│   │                             ForgeTaskRJ45InsertCfg   / ForgeKukaRJ45InsertCfg
│   │                             ForgeTaskBNCSmallInsertCfg / ForgeKukaBNCSmallInsertCfg
│   │                             ForgeKukaCtrlCfg, ForgeKukaEventCfg (shared Kuka base)
│   ├── forge_tasks_cfg.py     ← Task configs:
│   │                             ForgeBoxLidInsert, ForgeRJ45Insert, ForgeBNCSmallInsert
│   │                             SmallBoxCfg, LidYellowCfg
│   │                             RJ45FemaleCfg, RJ45MaleCfg
│   │                             BNCSmallFemaleCfg, BNCSmallMaleCfg
│   └── agents/rl_games_ppo_cfg.yaml  ← PPO config (shared by all tasks)
└── factory/
    ├── factory_env.py         ← Reset logic, rewards, success checks
    ├── factory_env_cfg.py     ← Window sizes, sim params, robot config
    └── factory_utils.py       ← Keypoint / pose utilities

source/isaaclab_assets/isaaclab_assets/custom_assets/
├── box/middle/
│   ├── Small_Box.usd          ← Fixed box (kinematic)
│   └── Lid_Yellow.usd         ← Held lid (Franka BoxLidInsert only)
├── rj45/medium/
│   ├── Adi_Medium_RJ45_Female_Raw_Colored.usd   ← Fixed socket (kinematic)
│   └── Adi_Medium_RJ45_Male_Raw_Colored.usd     ← Held plug (Franka RJ45 only)
├── bnc/small/
│   ├── Adi_BNC_Simulation_Small_Female.usd      ← Fixed socket (kinematic)
│   └── Adi_BNC_Simulation_Small_Male.usd        ← Held plug (Franka BNC only)
└── robots/lbr_description/urdf/kuka_blue/
    ├── kuka_blue.urdf              ← Kuka iiwa7 (BoxLidInsert: link_lid)
    ├── kuka_blue_rj45.urdf         ← Kuka iiwa7 (RJ45Insert: link_rj45)
    └── kuka_blue_bnc_small.urdf    ← Kuka iiwa7 (BNCSmallInsert: link_bnc)
```

---
