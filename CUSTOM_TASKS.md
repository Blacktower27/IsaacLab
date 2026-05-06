# FORGE Custom Insertion Tasks — Maintainer Guide

This document describes the three custom insertion tasks added on top of NVIDIA's
upstream FORGE / Factory pipeline:

1. **Box-Lid Insert** — snap-fit a yellow lid into a small plastic box.
2. **RJ45 Insert**    — plug an RJ45 male connector into a wall-mounted female socket.
3. **BNC Insert**     — push a BNC bayonet plug into a small panel-mount BNC socket.

All three are **registered for the Franka Panda only**. Kuka iiwa7 variants exist
in the registration table and config classes but **were never trained successfully**
— see §12 (Kuka status) at the bottom.

---

## 1. Quick start

### Gym IDs (Franka)


| Task      | Gym ID                                 | Episode length |
| --------- | -------------------------------------- | -------------- |
| Box-Lid   | `Isaac-Forge-BoxLidInsert-Direct-v0`   | 30 s           |
| RJ45      | `Isaac-Forge-RJ45Insert-Direct-v0`     | 30 s           |
| BNC Small | `Isaac-Forge-BNCSmallInsert-Direct-v0` | 30 s           |


Train (rl_games PPO) the same way as upstream FORGE tasks, e.g.:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
    --task Isaac-Forge-BoxLidInsert-Direct-v0 --num_envs 128 --headless
```

Registrations live in
[forge/**init**.py](source/isaaclab_tasks/isaaclab_tasks/direct/forge/__init__.py).

### File map


| File                                                                                                     | Purpose                                                                                                         |
| -------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| [forge_tasks_cfg.py](source/isaaclab_tasks/isaaclab_tasks/direct/forge/forge_tasks_cfg.py)               | Asset cfgs (`SmallBoxCfg`, `LidYellowCfg`, …) and per-task `ForgeTask` subclasses (geometry + reward knobs).    |
| [forge_env_cfg.py](source/isaaclab_tasks/isaaclab_tasks/direct/forge/forge_env_cfg.py)                   | `ForgeEnvCfg` subclasses + per-task gym kwargs + Kuka variants.                                                 |
| [forge_env.py](source/isaaclab_tasks/isaaclab_tasks/direct/forge/forge_env.py)                           | `ForgeEnv` runtime: action mapping, observation, FORGE-specific reward terms, force-sensor smoothing.           |
| [factory/factory_env.py](source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env.py)             | Base `FactoryEnv`. **Heavily extended** with per-task branches for keypoints, success/engage tests, init modes. |
| [factory/factory_utils.py](source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_utils.py)         | Shared helpers (`get_held_base_pos_local`, `get_target_held_base_pose`, etc.) with per-task branches.           |
| [factory/factory_env_cfg.py](source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_env_cfg.py)     | Configurable robot body names + `dof_torque_clamp` (added so non-Franka robots are possible).                   |
| [factory/factory_control.py](source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_control.py)     | OSC controller. Single change: torque clamp now reads `cfg.ctrl.dof_torque_clamp`.                              |
| [factory/factory_tasks_cfg.py](source/isaaclab_tasks/isaaclab_tasks/direct/factory/factory_tasks_cfg.py) | Two new fields on `FactoryTask`: `rot_weight` (rotation reward weight) and `terminate_on_success`.              |


---

## 2. RL interface — State, Action, Reward

This is the contract between the policy and the environment. The custom tasks
inherit the upstream FORGE / Factory contract and **extend it in three places**;
those extensions are how the same training pipeline can drive box-lid, RJ45,
and BNC insertion despite their very different geometry. **This is the most
important section if you only have time for one** — it summarises where the
new code differs from the upstream baseline. Per-task implementation details
live in §7 (keypoints) and §8 (engage / success).

### 2.1 State (policy observation + critic state)

The policy observation and the privileged critic state are **two separate
dictionaries** collapsed into flat tensors. Each is stacked over a sliding
**window** of `obs_window_size` / `state_window_size` timesteps (default 15).

**Policy observation** (`obs_order` in `ForgeEnvCfg`):


| Key                       | Dim | Where computed                                          | What it is                                                                                          |
| ------------------------- | --- | ------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `fingertip_pos_rel_fixed` | 3   | `forge_env._compute_intermediate_values`                | EE position relative to the (noisy) fixed-asset frame, expressed in the **fixed-asset local frame** |
| `fingertip_quat`          | 4   | `forge_env._compute_intermediate_values`                | EE orientation rel. fixed-asset frame, with random ±1 sign-flip + Gaussian noise                    |
| `ee_linvel`, `ee_angvel`  | 3+3 | finite-diff in `forge_env._compute_intermediate_values` | Computed from noisy fingertip pos/quat                                                              |
| `ft_force`                | 3   | `forge_env._compute_intermediate_values`                | Smoothed wrist force, rotated to fixed-asset frame, plus noise — **FORGE addition**                 |
| `prev_actions`            | 7   | always appended                                         | The action submitted at the previous step                                                           |


Single-step obs dim = 23. With window 15 → **345**.

**Critic state** (`state_order` in `ForgeEnvCfg`): all policy obs **without
noise**, plus privileged keys `joint_pos` (7), `held_pos` / `held_pos_rel_fixed`
/ `held_quat` (3+3+4), `fixed_pos` / `fixed_quat` (3+4), `task_prop_gains` (6),
`ema_factor` (1), `pos_threshold` / `rot_threshold` (3+3). Single-step state
dim = 49. With window 15 → **735**.

**Where defined:**


| What                      | Where                                                                                                                                                                                                                    |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Per-key dimensions        | `factory/factory_env_cfg.py: OBS_DIM_CFG / STATE_DIM_CFG`; extended in `forge_env_cfg.py:29-31` (`ft_force`, `force_threshold`)                                                                                          |
| Which keys are included   | `forge_env_cfg.ForgeEnvCfg.obs_order` / `state_order` — comment a line out to drop a key                                                                                                                                 |
| Per-key value computation | `factory_env._get_factory_obs_state_dict` (clean values), `forge_env._get_observations` (noisy + FORGE additions)                                                                                                        |
| Window stacking           | `factory_env._update_obs_state_history`                                                                                                                                                                                  |
| Final flatten             | `factory_utils.collapse_obs_dict`                                                                                                                                                                                        |
| Noise injection           | `forge_env._compute_intermediate_values` (`obs_rand.fingertip_pos`, `obs_rand.fingertip_rot_deg`, `obs_rand.ft_force`); fixed-asset noise written to `init_fixed_pos_obs_noise` in `factory_env.randomize_initial_state` |


**Differences from upstream Factory:**

1. **Wrist force in obs** — `ft_force` (3-D, smoothed by `ft_smoothing_factor=0.25`, rotated to fixed-asset frame) is added so the policy can react to contact during insertion. Upstream Factory has no force in the policy obs.
2. **Obs in fixed-asset local frame** — `fingertip_pos_rel_fixed` and `fingertip_quat` are rotated through `quat_apply_inverse(self.fixed_quat, ...)` before being added to the obs. This makes the policy invariant to the fixed asset's world placement / yaw — critical because the box can be yaw-randomised up to 360°.
3. **Window size 15** — upstream defaults to 1. With 15 timesteps stacked, the policy can learn from short-horizon history (force transients during contact, oscillations as the lid snaps).
4. **Random ±1 quaternion sign flip on `fingertip_quat`** — `flip_quats[rand_flips] = -1.0` in `_reset_idx`; breaks the policy's reliance on a specific sign convention for the quaternion.
5. **Domain-randomised gains exposed in critic state** — `task_prop_gains`, `ema_factor`, `pos_threshold`, `rot_threshold` are all randomised per-reset (`forge_env._reset_idx`) and added to `state_order` so the critic can condition on them. The policy doesn't see them.

### 2.2 Action

7-dimensional action space (vs upstream's 6):


| Index | Meaning                                       | Bounds → physical                                                                                                        |
| ----- | --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| 0,1,2 | Position delta in **fixed-asset local frame** | `[-1, +1]` × `pos_action_bounds` = ±5 cm; rotated to world via `fixed_quat`; clipped to ±`pos_threshold` from current EE |
| 3     | Roll delta                                    | **forced to 0** in `_apply_action`                                                                                       |
| 4     | Pitch delta                                   | forced to 0 for RJ45 / BNC; free for box-lid                                                                             |
| 5     | Yaw target in fixed-asset local frame         | symmetric mapping `[-1, +1] → [-180°, +180°]` — full 360°, no dead zone                                                  |
| 6     | Success-prediction logit                      | rescaled to `[0, 1]`; the policy is trained to predict its own success — **FORGE addition**                              |


Translation flow: `pos_actions @ pos_action_bounds → quat_rotate(fixed_quat, ⋅) → fixed_pos_action_frame + delta → clip to ±pos_threshold from current EE`.

Rotation flow: yaw mapped to `[-π, +π]` → built as Euler in the fixed-asset
local frame → composed to world via `fixed_quat` → composed with the EE-flip
rotation (π roll, so the gripper points down) → clipped per-Euler against
the current EE quaternion.

**Where defined:**


| What                              | Where                                                                                                                                           |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| Action dim                        | `ForgeEnvCfg.action_space = 7` (forge_env_cfg.py)                                                                                               |
| Action bounds                     | `CtrlCfg.pos_action_bounds = [0.05, 0.05, 0.05]`, `rot_action_bounds = [1, 1, 1]` (factory_env_cfg.py)                                          |
| Per-step clip thresholds          | `CtrlCfg.pos_action_threshold`, `rot_action_threshold`; **randomised per-reset** by `forge_utils.get_random_prop_gains`                         |
| Action → EE-target mapping        | `forge_env._apply_action`                                                                                                                       |
| EMA smoothing                     | `factory_env._pre_physics_step`: `actions = ema * raw + (1-ema) * prev`, `ema_factor` randomised per-env from `ema_factor_range = [0.025, 0.1]` |
| Reset bootstrap (inverse mapping) | `forge_env._reset_idx` — recovers the policy's `actions[:, 5]` (yaw) and `actions[:, 0:3]` (pos) from the current EE pose so EMA starts smooth  |
| OSC controller                    | `factory_env.generate_ctrl_signals` → `factory_control.compute_dof_torque`                                                                      |


**Differences from upstream Factory:**

1. **Asset-relative position deltas** — pos action is in the fixed-asset local frame and rotated to world via `fixed_quat` before being added to `fixed_pos_action_frame`. Upstream Factory adds the action directly in world frame. With this change "+X for the policy" means "toward the box's local +X" regardless of how the box is yaw-randomised.
2. **7th action dim — success prediction** — `actions[:, 6]` is the policy's bet that it is currently in the success state. Compared against the env's `_get_curr_successes` to produce `success_pred_error`, scaled by `success_pred_scale` which only switches on once `delay_until_ratio = 0.25` of envs have reached success at least once. Carried over from upstream FORGE; relied on by all three custom tasks.
3. **Yaw mapping with no dead zone** — `[-1, +1] → [-180°, +180°]` symmetric. Earlier iterations used `[-180°, +90°]` with a 90° dead zone (still visible in the legacy yaw arc scripts under `scripts/environments/`). The current full-360° mapping is implemented in both `_apply_action` (forward) and `_reset_idx` (inverse); **the two MUST stay in sync.**
4. **Pitch zeroed for plug tasks** — `actions[:, 4] = 0` for RJ45 / BNC because plug tasks need the connector vertical. Box-lid leaves pitch free since the lid rocks during snap-fit.
5. **Configurable torque clamp** — final OSC torque is clamped to `cfg.ctrl.dof_torque_clamp` (Franka 100 N·m, Kuka 200 N·m). Upstream had this hard-coded at 100.

### 2.3 Reward

Per-env reward at every env step is the **sum** of weighted terms:

```
rew_buf =   1.0 · kp_baseline                     # squashing(frame_dist; 5,4)
          + 1.0 · kp_coarse                       # squashing(frame_dist; 50,2)
          + 1.0 · kp_fine                         # squashing(frame_dist; 100,0)
          + 1.0 · curr_engaged                    # 0/1 from engage check
          + 1.0 · curr_success                    # 0/1 from success check
          - action_penalty_ee_scale    · |actions|
          - action_grad_penalty_scale  · |actions − prev_actions|
          - action_penalty_asset_scale · (pos_err + rot_err)             # FORGE
          - contact_penalty_scale      · ReLU(force − force_threshold)   # FORGE
          - success_pred_scale         · |true_success − policy_logit|   # FORGE
          + imitation_rwd_scale        · soft-DTW(policy_traj, ref_traj) # if ref_traj_json
```

with `frame_dist = mean(‖kp_held - kp_fixed‖) + rot_weight · ‖rpy(q_held^-1 q_fixed)‖`.

The three keypoint terms differ only in their squashing coefficients —
`kp_fine` is a sharp spike when keypoints are near-aligned; `kp_baseline` is
a smooth long-range attractor. All three operate on the **same per-task
keypoint geometry** (see §7).

**Where defined:**


| What                                          | Where                                                                                                                                                                           |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Per-task reward coefficients                  | `ForgeTask` subclasses in `forge/forge_tasks_cfg.py`: `*_scale`, `keypoint_coef_`*, `engage_threshold`, `success_threshold`, `rot_weight`, `contact_penalty_threshold_range`    |
| Keypoint distance computation                 | `factory_env._get_factory_rew_dict` — collects `kp_held`/`kp_fixed` per task, computes `keypoint_dist`, `rot_dist`, `frame_dist`                                                |
| Engage / success check branches               | `factory_env._get_curr_successes` — per-task `if self.cfg_task.name == ...` ladder; dispatched **twice per step** (once with `engage_threshold`, once with `success_threshold`) |
| Per-task keypoint allocation                  | `factory_env._init_tensors`                                                                                                                                                     |
| Per-task keypoint reset & sampling            | `factory_env._reset_idx` (after `randomize_initial_state`)                                                                                                                      |
| Phase-2 keypoint replacement on first success | `factory_env._log_factory_metrics` (box-lid only)                                                                                                                               |
| Progressive descent / Y-ease curriculum       | `factory_env._get_factory_rew_dict` — separate blocks for rj45 / box-lid / bnc                                                                                                  |
| FORGE-only reward terms                       | `forge_env._get_rewards` — action_penalty_asset, contact_penalty, success_pred_error                                                                                            |
| Soft-DTW imitation reward                     | `forge_env._init_imitation_reward`, `_load_ref_traj_entries`, computed in `_get_rewards` via `automate_algo.get_imitation_reward_from_dtw[_pose_traj]`                          |


**Differences from upstream Factory (this is where the tasks really diverge):**

1. **Per-task custom keypoint geometry.** Upstream Factory uses K colinear
  keypoints along the held-base Z axis — the *same* geometry for all tasks.
   The custom tasks override this entirely. See §7 for the full picture; the
   summary:
  - **box-lid** — 2 fixed clip points (indices 0,1) + 10 front-rim points
  (indices 2..11) at reset, replaced after first success by 100 random
  body-volume points (two-phase).
  - **rj45** — 64 random points uniformly on the male bottom patch (XY
  plane); female-side Z progressively pulled down.
  - **bnc** — 4 colinear Z-axis points on the plug; socket-side Z
  progressively walked down.
   Allocated in `_init_tensors`, sampled in `_reset_idx`, used in
   `_get_factory_rew_dict`.
2. **Progressive target descent (built-in curriculum).** Once
  `keypoint_dist < kp_advance_threshold`, the task-specific reward block
   walks the **fixed-side** keypoints toward their final position — RJ45 and
   BNC: Z descent; box-lid: Y ease. The policy is rewarded for matching a
   target that gets harder as the policy gets better. Upstream has nothing
   analogous.
3. **Per-task engage / success geometry.** Upstream uses one threshold per
  task (`z_disp < height × success_threshold` AND `xy_dist < 2.5 mm`). The
   custom tasks override `_get_curr_successes` with per-task geometric
   checks dispatched by *threshold-value sentinels*:
  - **box-lid** — clip-in-pocket geometry.
  - **rj45** — XY+yaw+tilt cone for engage, depth-based for success;
  sentinel `success_threshold > 1.0`.
  - **bnc** — same pattern as rj45 but with π-symmetric yaw and configurable
  min-depth.
   See §8 for thresholds and constants.
4. `**rot_weight` field on `FactoryTask`.** Mixes a geodesic rotation
  distance into `frame_dist`. All custom tasks set 0 (pure XYZ keypoint
   reward worked best), but the field is in upstream now for ablations.
5. `**terminate_on_success` field on `FactoryTask`.** When `True`,
  `_get_dones` ends the episode on first success. Custom tasks leave it
   `False` so the policy keeps getting `+1` per step while holding the part
   inserted (encourages "hold", not just "reach").
6. **Soft-DTW imitation reward.** Optional `imitation_rwd_scale ·
  soft_dtw(policy_traj, ref_traj)`term, gated by`cfg_task.ref_traj_json `(set to`""`to disable). Each task has its own collector script under` scripts/collect_*_trajectories.py` that produces the JSON. See §14.
7. **Success-prediction reward.** `success_pred_error = |actions[:,6] −
  true_success|`, scaled by` success_pred_scale`which switches on only  after`delay_until_ratio = 0.25` of envs have actually achieved success at
   least once in the rollout. Prevents the policy from collapsing to "always
   predict 0" early in training.

---

## 3. Asset geometry & coordinate conventions

All assets are stored under
`source/isaaclab_assets/isaaclab_assets/custom_assets/`. STL units are millimetres,
and the USDs were converted with scale `0.001`. **Therefore every dimension you
see in the configs is in metres**, and all "mm" numbers in this document have
already been ×0.001 when used in code.

### 3.1 Box-Lid

```
Small_Box.usd  (fixed asset)
  X ∈ [-60, 60]   (120 mm wide)
  Y ∈ [-50, 50]   (100 mm deep)
  Z ∈ [  0, 30]   ( 30 mm tall)         ← USD origin at base face (Z=0)
  Inner rim step at Z=27 mm (X=±52.5, Y=±39.5)
  Two clip pockets in front wall (Y≈-50): centred at X=±25 mm, Z=20-29 mm

Lid_Yellow.usd  (held asset)
  Body:    X ∈ [-52.3, 52.3], Y ∈ [-44.4, 39.4], Z ∈ [18.8, 30]
  Handle:  X ∈ [ -8.5,  8.5], Y ∈ [ -0.6, 37.9], Z ∈ [30, 55]   ← USD origin
  USD origin is NOT at the body's geometric base. Body bottom is at Z=18.8 mm.
  Two clip protrusions at Y- side, X=±25 mm, Z=26.5-28.5 mm.
```

**Assembled state (snap fit):** lid origin Z = box origin Z. The lid top plate
(Z=30 mm) ends up flush with the box top face. The clips at the lid Y- side
engage in the front-wall pockets; the opposite (Y+) side then snaps down.

The non-zero `LidYellowCfg.base_height = 0.0188` is the **distance from the lid
USD origin to the bottom of the lid body** (18.8 mm × 0.001). This single
number is the reason the success target is at box-local Z = 0.0188 m and not
at box-local Z = 0.030 m (the box top).

### 3.2 RJ45

```
rs_female_rj45.usd  (fixed asset)
  USD origin at the opening face (Z=0).
  Bounds: X ∈ [-26.4, 26.7], Y ∈ [-18.65, 12.6], Z ∈ [-51.45, 0]
  pos.z = 0.05145 m at spawn → socket bottom flush with table.

rs_male_rj45.usd  (held asset)
  USD origin at the connector mating face (Z=0).
  Tip (insertion end): Z = -3.0 mm   (held_base_z_offset = -0.003)
  Cable / housing top: Z = +88.47 mm (`height` field in cfg).
```

**Assembled state:** male origin coincides with female origin. The held base
is offset by `-0.003` so it tracks the physical connector tip, not the USD
origin.

### 3.3 BNC Small

```
Adi_BNC_Simulation_Small_Female.usd  (fixed asset)
  Bounds: X ∈ [-22, 22], Y ∈ [-22, 22], Z ∈ [-35, +25]
  USD origin is 35 mm above the base, 25 mm below the opening.
  pos.z = 0.035 m at spawn → socket base flush with table.
  Socket opening at female-local Z = 0.025 m.

Adi_BNC_Simulation_Small_Male.usd  (held asset)
  Bounds: X ∈ [-20.7, 20.7], Y ∈ [-18.3, 18.3], Z ∈ [+36.235, +107.2]
  USD origin is 36.235 mm BELOW the connector tip — the entire part lives
  above the origin. base_height = +0.036235 m (positive — tip above origin).
```

**Assembled state:** plug tip at female-local Z = -0.004765 m (empirically
calibrated from the visualizer; see `get_target_held_base_pose`).

### 3.4 Held-base offset summary

`factory_utils.get_held_base_pos_local` computes the offset from the held
asset's USD origin to the **physical contact face** that the policy is rewarded
for placing on the target:


| Task             | `held_base_z_offset` | Meaning                                       |
| ---------------- | -------------------- | --------------------------------------------- |
| `peg_insert`     | 0.000                | Peg origin already at the base face.          |
| `nut_thread`     | `fixed.base_height`  | Use the bolt-fixture height (legacy).         |
| `box_lid_insert` | **+0.0188**          | Lid USD origin is above the body bottom face. |
| `rj45_insert`    | **-0.003**           | Tip is 3 mm below the USD origin.             |
| `bnc_insert`     | **+0.036235**        | Tip is 36.235 mm above the USD origin.        |


If you ever swap an asset, **start by re-measuring this offset** — every
downstream success / keypoint / target computation uses it.

---

## 4. Anatomy of a custom task

A custom FORGE task plugs into the upstream `FactoryEnv` by adding a `name`
string and per-name branches in five places. To add a fourth task, you add
the same five branches:


| Place                                                                  | What you add                                                                       |
| ---------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `factory_utils.get_held_base_pos_local`                                | Z offset from held USD origin to physical contact face.                            |
| `factory_utils.get_target_held_base_pose`                              | Target position of the held base in the fixed-asset local frame at full insertion. |
| `factory_env.get_handheld_asset_relative_pose`                         | Default EE → asset transform at grasp time (Franka only — see §5).                 |
| `factory_env._get_curr_successes`                                      | Engage check (`success_threshold > some sentinel`) and success check.              |
| `factory_env.randomize_initial_state` + `_init_tensors` + `_reset_idx` | Per-task keypoint sampling, contact-init geometry, optional progressive descent.   |


Then in `forge/forge_tasks_cfg.py`:

- `FixedAssetCfg` subclass for the static asset (USD path, height, mass, friction).
- `HeldAssetCfg` subclass for the gripped asset.
- `ForgeTask` subclass tying the two together with task-specific knobs (init mode, keypoint coefs, success/engage thresholds, contact-init geometry, etc.).

And in `forge/forge_env_cfg.py`:

- `ForgeTaskXxxCfg(ForgeEnvCfg)` with `task_name`, `task`, `episode_length_s`.
- A line in `forge/__init__.py` to register the gym ID.

---

## 5. Grasp model (Franka)

`factory_env.get_handheld_asset_relative_pose` returns `(pos, quat)` for the
asset in the **flipped fingertip frame**. The grasp pipeline in
`randomize_initial_state` then teleports the held asset into this offset before
closing the gripper.

For each task the position is `(0, 0, height − franka_fingerpad_length)` plus
a per-task `held_asset_pos_offset` for fine-tuning, where:

- `height` is the cfg field that points to the **grasp face along Z** (handle
top for the lid, cable top for RJ45, body centre for BNC).
- `franka_fingerpad_length = 0.0176 m` is the distance from the fingertip frame
origin to the centre of the finger pads.

For the lid the orientation is non-trivial because the lid handle is short and
the body is large:

```python
held_asset_rot_init  = 90°    # yaw — long axis aligned with approach direction
held_asset_rot_offset = [0°, 35°, 0°]  # 35° pitch tilts handle forward
```

The 35° pitch is the empirically tuned amount that lets the gripper close on
the handle without the body colliding with the finger pads.

> **Tip when adding a new grasp:** use the `held_asset_pos_offset` knob to
> centre the asset between the pads after the gripper closes. The values you
> see in the cfgs were tuned by spawning the env with the gripper open,
> closing it, and adjusting `held_asset_pos_offset` until visible interpenetration
> disappeared. This is what the (deleted) DEBUG observe loop in
> `randomize_initial_state` was used for; you can resurrect it from git
> history if you need to re-tune.

---

## 6. Initial state (`init_mode`)

Every custom task supports four init modes via `cfg_task.init_mode`. The
selected mode applies after the fixed asset has been randomised:

- `**near`** — Hand placed deterministically directly above the socket using
`hand_init_pos`, with EE pointing straight down. No noise.
- `**far**`  — Hand position sampled uniformly from `hand_init_*_range`
(in the fixed-asset local frame, then rotated to world). Yaw aligned to the
fixed asset ± `hand_init_yaw_noise_deg`. No pitch noise (RJ45/BNC) or
± `hand_init_pitch_noise_deg` noise (Box-Lid).
- `**mixed**` — Bernoulli per-env: `near_init_prob` of envs use `near`, the
rest use `far`.
- `**contact**` — *Most useful for hard tasks.* Sample a contact pose between
the held part and the fixed part directly: pick one random point on a
designated face/edge of the female and one on the male, jitter the
orientation in the fixed-asset frame, and place the held part so the two
points coincide. This puts the lid resting on the box rear edge / the plug
resting on the socket rim at episode start, dramatically reducing exploration.

The contact-init geometry is configured per task:


| Task             | Female / fixed reference                                  | Male / held reference                             |
| ---------------- | --------------------------------------------------------- | ------------------------------------------------- |
| `box_lid_insert` | `box_contact_rear_edge_`* — top edge of the box rear wall | `lid_contact_front_edge_*` — front rim of the lid |
| `rj45_insert`    | `female_rear_edge_*_local`                                | `male_bottom_patch_*_local`                       |
| `bnc_insert`     | `bnc_contact_init_female_*`                               | `bnc_contact_init_male_*` (annular tip face)      |


For Box-Lid the rear edge is split at the X midpoint and one half is chosen
per env, paired with the matching half of the lid front rim. This guarantees
the clips of the lid line up over the pockets of the box.

For BNC, contact init also adds a random ±π yaw flip (bayonet has 0/180°
symmetry).

The contact-init code path lives in `randomize_initial_state` around the
`use_contact_style_init` block (currently lines ~1290-1460 in
`factory/factory_env.py`). It is shared between Franka (separate held asset,
held-to-fingertip computed from the grasp transform) and Kuka (held link
embedded in URDF, held-to-fingertip measured from the actual reset pose).

---

## 7. Keypoint reward

`FactoryEnv._get_factory_rew_dict` computes a dense reward by placing K
keypoints on the held asset and on the fixed asset, then squashing the mean
distance between matched pairs. Each task picks its own keypoint geometry; the
shaping coefficients (`keypoint_coef_baseline / coarse / fine`) are the same
across tasks. A non-zero `rot_weight` adds a geodesic rotation distance term
to `frame_dist`, but **all three custom tasks set `rot_weight = 0.0`** — the
pure XYZ keypoint signal works better in practice.

### 7.1 Box-Lid keypoints (per-episode, two phases)

```
Buffer size = 2 + max(num_reset_extra_kp, num_success_extra_kp)
              ≡ 2 + max(10, 100)  = 102

Indices 0,1 : the two clip protrusion points on the lid; matched against the
              corresponding pocket points on the box. Box-side targets can be
              overridden via `kp_box_clip_ease_end_left/right` to land the
              clips slightly off CAD (matches real snap-fit physics).

Reset (phase 1)   indices 2..2+num_reset_extra_kp-1:
              Random X across the lid front face, Y/Z fixed (the front rim).
              Provides front-edge alignment signal during approach.

On first success indices 2..2+num_success_extra_kp-1 are overwritten with
              random XYZ samples across the lid body volume — denser signal
              for the seating phase.
```

The box-side Y of every keypoint **eases** from `kp_advance_y_start = 0.025`
toward the per-keypoint true success Y (`kp_box_y_target`) once the mean
keypoint distance drops below `kp_advance_threshold`. This implements a
"come-and-snap-shut" curriculum and avoids the policy getting stuck in a
front-wall-touching local optimum.

### 7.2 RJ45 keypoints (per-episode, single phase)

`num_reset_kp = 64` random points uniformly on the male bottom patch (XY plane,
Z fixed at `male_bottom_patch_z_local`). The female-side keypoints share the
same XY plus the cavity-Y offset (`socket_target_y_local`), Z at
`female_rear_edge_z_local`.

Once the mean keypoint distance drops below `kp_advance_threshold = 0.005`,
the female keypoints' Z is decremented by `kp_advance_step = 0.0002` per env
step, clamped at `kp_advance_z_limit = -0.060`. This **progressively pulls
the target deeper** into the socket and is the main mechanism that drives the
policy from "tip touching the rim" to "fully seated".

### 7.3 BNC keypoints (per-episode, colinear Z-only)

`num_reset_kp = 4` keypoints collinear on the plug's Z axis, with
Z ~ Uniform(`bnc_kp_z_center` ± `bnc_kp_z_half_spread`) — i.e. a stack of
points along the connector axis. Socket-side Z starts above the plug-side Z
by `bnc_kp_socket_z_init_extra + bnc_kp_socket_z_above_engage_m` and walks
down by `bnc_kp_advance_step` per env step once close enough.

The progressive-descent mechanism is the same idea as RJ45 — the difference
is that BNC's reward gradient is Z-only because the bayonet is rotationally
near-symmetric.

---

## 8. Engage / success criteria

`FactoryEnv._get_curr_successes` is dispatched twice per env step:

- Once with `engage_threshold` → returns `curr_engaged` (intermediate bonus).
- Once with `success_threshold` → returns `curr_successes` (sparse terminal
bonus, also latched as `ep_succeeded`).

Each task uses its own threshold *value sentinels* to choose which check
branch to execute:

### Box-Lid


| Threshold                          | Value   | Branch                                                                                                           |
| ---------------------------------- | ------- | ---------------------------------------------------------------------------------------------------------------- |
| `engage_threshold = 0.9` (≥ 0.5)   | engage  | Clips at correct X, inside the groove (Y), at a reasonable Z (above floor, ≤ box top + 5 mm).                    |
| `success_threshold = 0.04` (< 0.5) | success | Both clips fully seated: X aligned, Y at the front wall (`Y_TOL = 3 mm`), Z within the pocket window (21–29 mm). |


The `_LEFT_HOLE_X / _RIGHT_HOLE_X` constants in `_get_curr_successes` are
measured from the box STL — leave them alone unless you re-cut the box.

### RJ45


| Threshold                                 | Value   | Branch                                                                                             |
| ----------------------------------------- | ------- | -------------------------------------------------------------------------------------------------- |
| `engage_threshold = 2.0` (> 1.0)          | engage  | XY < 4 mm of the cavity centre, yaw error < 15°, plug tilt < 15°, `z_disp < 60 mm`.                |
| `success_threshold = 0.027` (< 0) intent? | success | Tip below `socket_target_z_local` by at least 2 mm (`height_threshold = 0 + (-0.002)`), XY < 4 mm. |


> The `success_threshold = 0.027` value in the cfg looks unusual; the active
> SUCCESS code path uses the **task-cfg** `socket_target_z_local` for the
> reference and `height_threshold = fixed_cfg.height + success_threshold` for
> the depth tolerance. `fixed_cfg.height = 0.0` for RJ45, so the trigger is
> `z_disp < success_threshold` (i.e. tip 27 mm below the cavity-entrance
> reference). When tightening the success bar, lower this number.

### BNC


| Threshold                        | Value   | Branch                                                                                                                                                                   |
| -------------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `engage_threshold = 2.0` (> 1.0) | engage  | XY < 4 mm, π-symmetric yaw < 10°, tilt < 15°, `z_disp` not more than `bnc_engage_z_max_above_opening = 0.040 m` above the opening, optional minimum depth.               |
| `success_threshold = -0.3` (< 0) | success | Tip past `min(height × success_threshold, -bnc_success_min_depth_m)` below the opening, XY < 3 mm. With `bnc_success_min_depth_m = 0.018` this triggers at ~18 mm depth. |


`bnc_apply_success_criteria = False` lets you train BNC without any success
bonus / `ep_succeeded` latch — useful for pure-imitation experiments.

---

## 9. Force/torque sensor

`forge/forge_env.py` reads the wrist F/T from PhysX's
`get_link_incoming_joint_force` on the `force_sensor` body. The body name is
configurable via `cfg.ctrl.force_sensor_body_name`; for Franka it defaults to
the upstream `force_sensor` link. The reading is exponentially smoothed
(`ft_smoothing_factor = 0.25`) and rotated into the fixed-asset local frame
before being added to the obs/state dicts as `ft_force` and noisy in the policy
obs as `noisy_force`.

To turn the force observation **off** for ablations, remove `"ft_force"` from
`obs_order` in `ForgeEnvCfg`. The base classes do not require it.

---

## 10. Modifications to the upstream Factory base classes

The upstream `FactoryEnv` was extended in-place to support the custom tasks.
The biggest changes (relative to NVIDIA's main):

### 10.1 Configurable robot body names + held-via-robot-body mode

`factory_env_cfg.CtrlCfg` now exposes:

- `fingertip_body_name`, `left_finger_body_name`, `right_finger_body_name`,
`force_sensor_body_name` — overridable so non-Franka URDFs can be used
(Kuka subclass overrides them; not yet working — see §12).
- `held_body_name` — when **non-empty**, the held asset is treated as a body of
the robot URDF (e.g. a `link_lid` rigidly fixed to the TCP). In that case
`factory_env._setup_scene` skips creating a separate `_held_asset`, and
`_compute_intermediate_values` reads `self.held_pos / held_quat` from the
robot body instead. This was the mechanism intended to support a Kuka with
the part baked into the URDF; the path is exercised by Kuka cfgs but those
variants don't train.

### 10.2 RigidObject vs Articulation for the fixed asset

A box / socket fixed to the table cannot be modelled as a `fix_root_link=True`
articulation in the same PhysX partition as the robot (`gpu_max_num_partitions=1`
forbids mixing fixed-base and free-base articulations). The custom tasks
declare their fixed asset as a `**RigidObjectCfg` with `kinematic_enabled=True`**
instead, which gives PhysX an infinite-mass static body and avoids the
partition incompatibility.

`factory_env._setup_scene` was updated to detect the cfg type and register the
fixed asset under either `scene.articulations` or `scene.rigid_objects`
accordingly:

```python
if isinstance(self.cfg_task.fixed_asset, RigidObjectCfg):
    self._fixed_asset = RigidObject(self.cfg_task.fixed_asset)
else:
    self._fixed_asset = Articulation(self.cfg_task.fixed_asset)
```

All other `_fixed_asset` accesses (root_pos_w, root_quat_w, write_root_pose_to_sim)
work uniformly for both types, so this is the only branch that was needed.

### 10.3 ArticulationRoot on a child prim

For URDFs whose articulation root sits on a child link (`/Robot/link_0`)
instead of `/Robot`, `_apply_robot_articulation_props_on_source_env` finds the
real root prim (via `UsdPhysics.ArticulationRootAPI`) and applies the spawn
articulation properties there before the scene is cloned. Without this, the
cloned envs would have inconsistent fixed-base / free-base settings across
articulation roots.

This was added for the Kuka URDF; Franka is unaffected because its
articulation root already sits on `/Robot`.

### 10.4 Configurable torque clamp

`factory_control.compute_dof_torque` previously hard-coded
`torch.clamp(dof_torque, -100.0, 100.0)`, sized for Franka. It now reads
`cfg.ctrl.dof_torque_clamp` (default 100 N·m) so robots with higher torque
budgets can use them. Kuka iiwa7 sets it to 200.

### 10.5 New `FactoryTask` fields

- `rot_weight: float = 0.0` — added so the keypoint reward can include a
geodesic rotation term. All custom tasks set 0.0; available for ablations.
- `terminate_on_success: bool = False` — when `True`, `_get_dones` ends the
episode as soon as the success criterion fires (instead of always running
to time-out). All custom tasks leave it `False` because the dense post-
success reward (curr_success staying 1) helps the policy learn to *hold*
the part inserted, not just *reach* the inserted state.

### 10.6 Observation/state history (window) buffers

`FactoryEnvCfg.obs_window_size = 15` and `state_window_size = 15` (defaults).
`_update_obs_state_history` rolls the latest obs/state into a fixed-size
buffer and flattens; the policy and critic networks see the last 15 timesteps
concatenated. Reduce these to 1 to fall back to the upstream behaviour.

---

## 11. Geometry code pointers

This is a quick lookup for the geometry-related code blocks not already covered
by the "Where defined" tables in §2 (which index State / Action / Reward code
locations). Read these alongside §3 (asset geometry), §4 (anatomy), §5 (grasp)
and §6 (init mode).


| Concern                                               | Location                                                                                    |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| Held-base offset per task (USD origin → contact face) | `factory/factory_utils.py: get_held_base_pos_local`                                         |
| Target held-base pose at full insertion               | `factory/factory_utils.py: get_target_held_base_pose`                                       |
| EE → asset transform at grasp (Franka)                | `factory/factory_env.py: get_handheld_asset_relative_pose`                                  |
| Contact-init sampling (paired held / fixed points)    | `factory/factory_env.py: randomize_initial_state` — large `if use_contact_style_init` block |


---

## 12. Kuka status (NOT WORKING)

The repository contains Kuka iiwa7 variants of all three tasks:

- `Isaac-Forge-BoxLidInsert-Kuka-Direct-v0`
- `Isaac-Forge-RJ45Insert-Kuka-Direct-v0`
- `Isaac-Forge-BNCSmallInsert-Kuka-Direct-v0`

In all three, the held part (lid / RJ45 male / BNC male) is **embedded
directly in the robot URDF** as a fixed child link of `link_tcp`
(`link_lid`, `link_rj45`, `link_bnc`). There is no separate held asset in
the scene, and `cfg.ctrl.held_body_name` selects which robot link to track.

Per-task URDFs are at:

```
source/isaaclab_assets/isaaclab_assets/custom_assets/robots/lbr_description/urdf/
    kuka_blue_lid.urdf
    kuka_blue_rj45.urdf
    kuka_blue_bnc_small.urdf
```

with corresponding USDs at `…/usd/kuka_blue_*/`.

**These configurations were never trained to convergence.** Several issues
were investigated (and the fixes are in the code):

- Null-space controller `default_dof_pos_tensor` overridden to Kuka's home
pose (Franka's value would pull joints toward Franka's home and fight the
OSC).
- `dof_torque_clamp = 200` to match Kuka's URDF effort limits.
- `world_joint` origin in the URDF set to (0, 0, 0) so the base sits on the
table.
- `_apply_robot_articulation_props_on_source_env` finds the actual
articulation root on a child prim.

But the policies still do not learn the tasks reliably. The likely remaining
issues are link collision geometry on the Kuka URDF and the fact that the
embedded-held-link grasp model differs subtly from the Franka grasp transform
(the contact-init code branches on `held_body_name in {link_rj45, link_lid, link_bnc}` to handle this). **Treat the Kuka variants as scaffolding only —
they are exercised on `_setup_scene`, but the reward shaping and contact-init
geometry was tuned for Franka and is unlikely to give a working policy on
Kuka without further work.**

If you want to retire the Kuka path entirely, deleting the four Kuka cfg
classes in `forge_env_cfg.py`, the four Kuka registrations in
`forge/__init__.py`, and the `held_body_name` branch in
`factory_env._setup_scene` / `_compute_intermediate_values` is enough — no
Franka path depends on them.

---

## 13. Common pitfalls

1. **STL units.** Never write a number in mm into a config field. The cfgs
  are all metres. The asset comments use mm only to make the geometry
   readable.
2. **Held-base offset.** When swapping the held asset USD, recompute
  `held_asset_cfg.base_height` and update both `get_held_base_pos_local`
   and `get_target_held_base_pose` for that task. Otherwise the keypoint
   reward and the success check will silently disagree.
3. `**fix_root_link=True` on the fixed asset.** Don't. The custom tasks
  use `RigidObjectCfg(kinematic_enabled=True)` instead — see §10.2.
4. **ArticulationRoot on `/Robot`.** Custom robot USDs sometimes put it on a
  child link. `_apply_robot_articulation_props_on_source_env` patches the
   real root before cloning; if you add a new robot, verify the articulation
   root path.
5. **ref_traj_json paths.** The path is relative to the IsaacLab repo root.
  If the file is missing, the imitation reward is silently disabled with a
   warning — check the launch log.
6. `**obs_order` and `state_order`.** Adding/removing a key changes the
  tensor sizes downstream. Both are read at env construction time and used
   to size the policy/critic input layers.
7. `**held_asset_pos_offset` for new grasps.** Re-tune empirically by
  spawning the env with the gripper open, watching the asset settle, and
   nudging until it sits centred between the pads when the gripper closes.
8. **Yaw mapping forward/inverse pairing.** `_apply_action` and `_reset_idx`
  both contain the yaw mapping (forward and inverse respectively). The
   inverse seeds the action buffer so EMA starts from the actual pose. If
   you change the forward mapping (e.g. add a dead zone), update the
   inverse — otherwise the first step after reset will be wrong.

---

## 14. Companion scripts under `scripts/`

A successor will spend a significant amount of time in these helper scripts
when debugging geometry, recalibrating contact-init regions, or building new
reference trajectories. They are not part of the gym training loop, but they
are tightly coupled to the task configs (they import the same `cfg_task`,
read the same geometry constants, and exercise the exact same
engage / success / keypoint code paths).

### 14.1 Reference-trajectory collectors

Each task has its own teleport-only collector that drives the held part along
a synthetic trajectory and writes a JSON consumed by the Soft-DTW imitation
reward (`cfg_task.ref_traj_json`).


| Script                                                                             | What it does                                                                                                                                                                                                                                          | Output                                             |
| ---------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| [scripts/collect_box_lid_trajectories.py](scripts/collect_box_lid_trajectories.py) | 3-phase teleport: hold at contact init → world-position keypoint align → snap-fit insert with a least-squares solver against `kp_box_clip_ease_end_`*. Stops per-env at the first frame `_get_curr_successes` fires.                                  | `box_lid_ref_traj.json` (Forge) + `_automate.json` |
| [scripts/collect_rj45_trajectories.py](scripts/collect_rj45_trajectories.py)       | 2-phase cubic-spline teleport: arc from contact-init to centred above the socket, then straight vertical descent.                                                                                                                                     | `rj45_ref_traj.json` + `_automate.json`            |
| [scripts/collect_bnc_trajectories.py](scripts/collect_bnc_trajectories.py)         | 2-phase teleport: cubic-spline approach (drops to `--approach_delta_z_m` above the seated tip), then axial line to `--tip_z_phase2_end`. Bayonet helix is **not** recorded — set `--tip_z_phase2_end` to the depth where you want the helix to start. | `bnc_ref_traj.json` + `_automate.json`             |


All three:

- Teleport the **held asset only** (Franka separate-asset model). The robot
EE columns in the Automate output are computed from the kinematic grasp
inverse using the task's `held_asset_`* offsets.
- Record `held_tip_local`, `held_rpy_local`, `held_tip_pose_local` (7-D
quaternion) in the **fixed-asset (female / box) local frame** — same
convention `ForgeEnv._get_held_tip_pose_local` uses at runtime, so the
Soft-DTW reward sees apples-to-apples poses.
- Stop recording per-env at the first `_get_curr_successes = True`, so a
trajectory naturally ends at full insertion.

The committed JSONs (`scripts/{box_lid,rj45,bnc}_ref_traj.json` plus
`_automate` companions) are the baselines that were used during training.
They are not loaded by default — set `ref_traj_json` in the task cfg to
re-enable the imitation reward.

> **Caveat (BNC).** The bayonet has a phase-3 helical L-slot lock that the
> STL does not encode (see [scripts/inspect_bnc_stl.py](scripts/inspect_bnc_stl.py)
> output). The collector therefore only records "axial insert", and the
> trained policies never learn the rotational lock — they treat the BNC
> like a friction-fit plug. This is a known limitation of the asset.

### 14.2 Interactive engage / success / keypoint visualizers (Franka)

These are the "tester" scripts. Each spawns the env in a single sim
instance, **bypasses the robot**, and lets you drive the held asset
directly with the keyboard while drawing every relevant frame
(keypoints, success target, contact-init sample regions, engage cone) as
coloured spheres / markers. Use them to:

1. Verify that `kp_*_local` buffers in `factory_env._reset_idx` produce the
  geometry you expect.
2. Hand-test the engage / success threshold dispatch — e.g. teleport to the
  "F" preset and check the engage flag toggles.
3. Recalibrate contact-init regions (`box_contact_rear_edge_`*,
  `female_rear_edge_*_local`, `male_bottom_patch_*_local`,
   `bnc_contact_init_*`) against the actual STL geometry — the visualizers
   draw the configured regions so you can drag values in the cfg until they
   sit on the right edges.


| Script                                                                                                   | Use for                                                                                                 |
| -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| [scripts/environments/visualize_box_lid_success.py](scripts/environments/visualize_box_lid_success.py)   | Box-lid engage/success geometry; clip vs pocket markers; contact-init paired-half guides.               |
| [scripts/environments/visualize_rj45_insert.py](scripts/environments/visualize_rj45_insert.py)           | RJ45 engage/success; per-episode plug & socket keypoints; contact-init rear-edge / bottom-patch guides. |
| [scripts/environments/visualize_bnc_small_insert.py](scripts/environments/visualize_bnc_small_insert.py) | BNC engage/success; Z-axis keypoint stack; contact-init bore vs tip-face regions.                       |


**Common keyboard controls:**

```
Arrows / Q / E   move held asset in fixed-asset XY / Z
I / K            pitch ±
J / L            yaw   ±
U / O            roll  ±
Hold Shift       5× speed
F                teleport to ENGAGE preset
G                teleport to SUCCESS preset
C                resample CONTACT init pose (RJ45, box-lid)
R                reset to defaults / origin coincide
```

**Common flags:**

- `--freeze_box` / `--freeze_socket` — disable fixed-asset randomisation
for axis-aligned debugging.
- `--num_envs N` — defaults to 1.

**Marker colour convention (consistent across the three Franka scripts):**


| Colour       | Meaning                                                                                 |
| ------------ | --------------------------------------------------------------------------------------- |
| BLUE         | Held tip / clip points (lid/plug local → world)                                         |
| ORANGE       | Fixed reference (socket opening / box hole rim)                                         |
| GREEN        | Per-episode keypoints on the **held** asset (`kp_*_male_local`, `kp_lid_local`, ...)    |
| YELLOW       | Per-episode keypoints on the **fixed** asset (`kp_*_female_local`, `kp_box_local`, ...) |
| MAGENTA      | Success target from `get_target_held_base_pose`                                         |
| WHITE / CYAN | Contact-init sampling guides (rear edge / bottom patch / inner bore)                    |


When the green and yellow keypoints overlap, `keypoint_dist → 0` and you are
at the geometric success state — this is a quick visual sanity check that
the keypoint code is consistent with the success code.

### 14.3 Kuka visualizer counterparts (KEEP-FOR-REFERENCE only)


| Script                                                                                                               |
| -------------------------------------------------------------------------------------------------------------------- |
| [scripts/environments/visualize_box_lid_success_kuka.py](scripts/environments/visualize_box_lid_success_kuka.py)     |
| [scripts/environments/visualize_rj45_success_kuka.py](scripts/environments/visualize_rj45_success_kuka.py)           |
| [scripts/environments/visualize_bnc_small_success_kuka.py](scripts/environments/visualize_bnc_small_success_kuka.py) |


These spawn the **Kuka** variants and route keyboard input through the FORGE
asset-relative action space (i.e. they go through the OSC controller, not
direct teleport). They were used during the failed Kuka bring-up — see §12.
The action-mapping comments at the top of each file describe an old
`yaw ∈ [-180°, +90°]` mapping with a 90° dead zone; **the active env now uses
full 360° with no dead zone**, so the F / G presets in the Kuka
scripts may overshoot the policy's reachable yaw range (see §2.2 for the
current mapping). Treat these scripts as historical context, not as accurate
testers of the current code.

### 14.4 Yaw range diagnostic scripts (legacy)


| Script                                                                                 | Purpose                                                                           |
| -------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| [scripts/environments/play_franka_yaw.py](scripts/environments/play_franka_yaw.py)     | Play a trained checkpoint with a coloured arc showing the policy's commanded yaw. |
| [scripts/environments/teleop_franka_yaw.py](scripts/environments/teleop_franka_yaw.py) | Keyboard teleop through the OSC; arc lets you eyeball where the dead zone is.     |


Both were written when the yaw mapping had a dead zone (`[-180°, +90°]`,
gap at `(+90°, +180°]`). The current `_apply_action` uses a full-360°
symmetric mapping, so the **dead-zone arc drawn by these scripts is no
longer accurate**. They are still useful for visualising the policy's
commanded yaw vs the EE's actual yaw — just ignore the red "dead zone" arc.

### 14.5 Offline trajectory plotting

[scripts/visualization/visualize_rj45_trajectory.py](scripts/visualization/visualize_rj45_trajectory.py)
is a matplotlib-only renderer (no Isaac Sim). It draws the female socket and
male plug STLs, overlaid with the cubic-spline trajectory used by the
collector, and writes a multi-page PDF. Useful for sanity-checking that the
`approach_delta_z_m`, contact-init samples, and final descent line all live
in the right region of space without spinning up the full simulator.

### 14.6 One-off asset utilities


| Script                                                                 | Purpose                                                                                                                                                                                               |
| ---------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [scripts/inspect_bnc_stl.py](scripts/inspect_bnc_stl.py)               | Print bounding boxes of the BNC male/female STLs (numpy + stdlib only). Run this when you suspect an STL was re-exported with a different origin or scale.                                            |
| [scripts/fix_link_lid_collision.py](scripts/fix_link_lid_collision.py) | Replace `link_lid`'s STL collision mesh with the `Lid_Yellow.usd` reference. **Must be run inside Isaac Sim's Script Editor** after re-importing the Kuka URDF — there is no command-line invocation. |


### 14.7 Suggested workflow for adding a new task

The scripts make a fairly mechanical workflow possible:

1. Convert STL → USD with the right `collision_approximation` (triangle mesh
  for static fixed assets, convex decomposition for dynamic held assets).
2. Add `FixedAssetCfg` / `HeldAssetCfg` / `ForgeTask` / `ForgeTaskXxxCfg`
  classes with measured geometry; register the gym ID.
3. Add the five branches to `factory_utils` and `factory_env` (see §4).
4. Spin up `visualize_xxx_success.py` (or copy/adapt one) and:
  - Verify the held-base offset is correct (BLUE marker should track the
   physical contact face, not the USD origin).
  - Calibrate the contact-init regions (WHITE / CYAN guides) against the
  STL bounds.
  - Hand-test the engage / success presets (`F` / `G` keys) until the
  thresholds fire where they should.
5. Run `collect_xxx_trajectories.py` to produce a JSON, plot it offline,
  then enable `ref_traj_json` and start training.

---

*Last verified against branch `custom_forge` at HEAD `8bd0549d` (all three
tasks working on Franka).*