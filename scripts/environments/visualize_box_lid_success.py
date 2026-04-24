"""Visualize and interactively test the ForgeBoxLidInsert success / contact geometry.

Keyboard controls (click the viewport once to give it focus):
  Arrow Up / Down   — move lid +Y / -Y  (world axes)
  Arrow Left / Right— move lid -X / +X  (world axes)
  Q / E             — move lid +Z / -Z  (up / down)
  I / K             — pitch lid +/-
  J / L             — yaw   lid +/-
  U / O             — roll  lid +/-
  Hold Shift        — 5× speed
  C                 — CONTACT init: sample paired left/right half + base RPY (matches factory_env)
  R                 — reset lid to nominal success offsets (pos/euler → defaults)

Coloured sphere markers (updated every frame):
  BLUE   — clip tooth positions (lid frame → world)
  RED    — hole rim reference (box frame → world)
  GREEN  — reward keypoints on lid (success geometry; fixed within episode)
  YELLOW — reward keypoints on box — current eased Y (starts at kp_advance_y_start, steps toward target)
  PURPLE — same keypoints at full success Y in box frame (kp_box_y_target + lid X/Z) — overlap yellow when eased
  WHITE  — box rear top edge (full X span from cfg) — calibrate Y/Z here
  CYAN   — lid front edge (full X span from cfg) — calibrate Y/Z here
  ORANGE — left X-half of box rear edge  (pair with green lid half)
  MAGENTA— right X-half of box rear edge
  LIME   — left X-half of lid front edge
  HOT PINK — right X-half of lid front edge
  (small) RED dots — X midpoints on rear / front guides (split plane)

When the lid is at the success position the blue and red markers overlap.

Usage
-----
./isaaclab.sh -p scripts/environments/visualize_box_lid_success.py --num_envs 1 --freeze_box

Flags
-----
--freeze_box   Disable box randomisation for a clean axis-aligned view.
--num_envs N   Number of parallel environments (default: 1).
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Interactive ForgeBoxLidInsert success visualizer.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--freeze_box", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
from collections import deque

import carb
import carb.input
import omni.appwindow
import torch
import gymnasium as gym

import isaacsim.core.utils.torch as torch_utils
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse, quat_conjugate, quat_from_euler_xyz, quat_mul

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


# ---------------------------------------------------------------------------
# Snap-fit reference points (from STL analysis, metres).
# ---------------------------------------------------------------------------
# Lid STL: clip width=9.3mm, tooth top Z=28.9mm, outer face Y=-44.4mm
_LEFT_CLIP_LID_LOCAL  = (-0.025218, -0.0444, 0.0289)
_RIGHT_CLIP_LID_LOCAL = ( 0.024250, -0.0444, 0.0289)

# Box STL: inner platform width=11mm, pocket Z=[21,29]mm, inner back Y=-39.5mm
_LEFT_HOLE_BOX_LOCAL  = (-0.025218, -0.0444, 0.0289)
_RIGHT_HOLE_BOX_LOCAL = ( 0.024250, -0.0444, 0.0289)

# Hole target centres in box-local frame (Z = midpoint of pocket [0.021, 0.029]).
_LEFT_HOLE_TARGET  = (-0.025218, -0.0444, 0.025)
_RIGHT_HOLE_TARGET = ( 0.024250, -0.0444, 0.025)


# ---------------------------------------------------------------------------
# Keyboard input  (event-subscription approach — more reliable than polling)
# ---------------------------------------------------------------------------
MOVE_STEP = 0.001    # metres per frame (×5 when Shift held)
ROT_STEP  = 0.01     # radians per frame
SHIFT_MUL = 5.0

Ki = carb.input.KeyboardInput

_key_states: dict = {}   # Ki → bool  (True = currently held)
_key_sub    = None       # subscription handle (must stay alive)


def _on_keyboard_event(event, *args, **kwargs):
    """Callback: update held-key state on KEY_PRESS / KEY_RELEASE."""
    if event.type == carb.input.KeyboardEventType.KEY_PRESS:
        _key_states[event.input] = True
    elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
        _key_states[event.input] = False
    return True   # continue propagating


def _init_keyboard():
    global _key_sub
    kbi = carb.input.acquire_input_interface()
    kbw = omni.appwindow.get_default_app_window().get_keyboard()
    _key_sub = kbi.subscribe_to_keyboard_events(kbw, _on_keyboard_event)
    print("[keyboard] subscribed — click the viewport once to give it focus.")


def _key(k) -> bool:
    return _key_states.get(k, False)


# ---------------------------------------------------------------------------
# Interactive lid pose state
# ---------------------------------------------------------------------------
# Position offset in world frame (metres). Default lifts lid slightly for visibility.
_pos_offset = [0.0, 0.0, 0.015]
# Rotation offset on top of box orientation: quat_mul(box_quat, euler_xyz(...)).
_euler_offset = [0.0, 0.0, 0.0]

_REAR_EDGE_SAMPLES = 13
_FRONT_EDGE_SAMPLES = 13


def _teleport_contact(inner):
    """Sample the same contact init as factory_env (paired L/R X halves + base RPY + noise)."""
    global _pos_offset, _euler_offset

    cfg = inner.cfg_task
    device = inner.device
    n = inner.num_envs
    box_pos_w = inner._fixed_asset.data.root_pos_w.clone()
    box_quat_w = inner._fixed_asset.data.root_quat_w.clone()
    ident_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0).expand(n, -1)

    side = torch.randint(0, 2, (n,), device=device)
    bx_lo, bx_hi = cfg.box_contact_rear_edge_x_range
    bx_mid = 0.5 * (bx_lo + bx_hi)
    x_lo_b = torch.where(side == 0, torch.full((n,), bx_lo, device=device), torch.full((n,), bx_mid, device=device))
    x_hi_b = torch.where(side == 0, torch.full((n,), bx_mid, device=device), torch.full((n,), bx_hi, device=device))
    female_local = torch.zeros((n, 3), device=device)
    female_local[:, 0] = torch.rand(n, device=device) * (x_hi_b - x_lo_b) + x_lo_b
    female_local[:, 1] = cfg.box_contact_rear_edge_y_local
    female_local[:, 2] = cfg.box_contact_rear_edge_z_local

    lx_lo, lx_hi = cfg.lid_contact_front_edge_x_range
    lx_mid = 0.5 * (lx_lo + lx_hi)
    x_lo_l = torch.where(side == 0, torch.full((n,), lx_lo, device=device), torch.full((n,), lx_mid, device=device))
    x_hi_l = torch.where(side == 0, torch.full((n,), lx_mid, device=device), torch.full((n,), lx_hi, device=device))
    male_local = torch.zeros((n, 3), device=device)
    male_local[:, 0] = torch.rand(n, device=device) * (x_hi_l - x_lo_l) + x_lo_l
    male_local[:, 1] = cfg.lid_contact_front_edge_y_local
    male_local[:, 2] = cfg.lid_contact_front_edge_z_local

    r_lo, r_hi = cfg.contact_init_roll_range_deg
    p_lo, p_hi = cfg.contact_init_pitch_range_deg
    y_lo, y_hi = cfg.contact_init_yaw_range_deg
    roll = torch.deg2rad(torch.rand(n, device=device) * (r_hi - r_lo) + r_lo)
    pitch = torch.deg2rad(torch.rand(n, device=device) * (p_hi - p_lo) + p_lo)
    yaw = torch.deg2rad(torch.rand(n, device=device) * (y_hi - y_lo) + y_lo)
    held_quat_w = quat_mul(
        box_quat_w,
        quat_from_euler_xyz(roll, pitch, yaw),
    )

    _, female_w = torch_utils.tf_combine(box_quat_w, box_pos_w, ident_q, female_local)
    male_off_w = torch_utils.quat_rotate(held_quat_w, male_local)
    lid_pos_w = female_w - male_off_w

    pose_w = torch.cat([lid_pos_w, held_quat_w], dim=-1)
    zero_vel = torch.zeros((n, 6), device=device)
    inner._held_asset.write_root_pose_to_sim(pose_w)
    inner._held_asset.write_root_velocity_to_sim(zero_vel)
    inner._held_asset.reset()

    # Keep keyboard nudging in sync (broadcast env0 offsets to all envs, like RJ45 visualizer).
    rel_p = lid_pos_w[0] - box_pos_w[0]
    _pos_offset = [rel_p[0].item(), rel_p[1].item(), rel_p[2].item()]
    rel_q = quat_mul(quat_conjugate(box_quat_w[0:1]), held_quat_w[0:1])
    rr, rp, ry = euler_xyz_from_quat(rel_q)
    _euler_offset = [rr.item(), rp.item(), ry.item()]

    side0 = "LEFT" if side[0].item() == 0 else "RIGHT"
    print(
        "[TELEPORT -> CONTACT]  "
        f"env0 side={side0}  "
        f"box_pt_local(mm)=[{female_local[0,0].item()*1000:+.1f},{female_local[0,1].item()*1000:+.1f},{female_local[0,2].item()*1000:+.1f}]  "
        f"lid_pt_local(mm)=[{male_local[0,0].item()*1000:+.1f},{male_local[0,1].item()*1000:+.1f},{male_local[0,2].item()*1000:+.1f}]  "
        f"rpy(deg)=[{math.degrees(roll[0].item()):+.1f},{math.degrees(pitch[0].item()):+.1f},{math.degrees(yaw[0].item()):+.1f}]"
    )


def _poll_keys_and_move_lid(inner):
    """Read held keys, update offsets, and write the new lid pose to sim."""
    global _pos_offset, _euler_offset

    mul = SHIFT_MUL if (_key(Ki.LEFT_SHIFT) or _key(Ki.RIGHT_SHIFT)) else 1.0
    ms  = MOVE_STEP * mul
    rs  = ROT_STEP  * mul

    # Translation (world frame axes — box may be rotated, but these are intuitive)
    if _key(Ki.UP):    _pos_offset[1] += ms
    if _key(Ki.DOWN):  _pos_offset[1] -= ms
    if _key(Ki.LEFT):  _pos_offset[0] -= ms
    if _key(Ki.RIGHT): _pos_offset[0] += ms
    if _key(Ki.Q):     _pos_offset[2] += ms
    if _key(Ki.E):     _pos_offset[2] -= ms

    # Rotation
    if _key(Ki.I): _euler_offset[1] += rs   # pitch +
    if _key(Ki.K): _euler_offset[1] -= rs   # pitch -
    if _key(Ki.J): _euler_offset[2] -= rs   # yaw  -
    if _key(Ki.L): _euler_offset[2] += rs   # yaw  +
    if _key(Ki.U): _euler_offset[0] += rs   # roll +
    if _key(Ki.O): _euler_offset[0] -= rs   # roll -

    # Reset to nominal pose (slight Z lift)
    if _key(Ki.R):
        _pos_offset = [0.0, 0.0, 0.015]
        _euler_offset = [0.0, 0.0, 0.0]

    device   = inner.device
    num_envs = inner.num_envs

    # Base pose: lid at same world position as box (= success position).
    box_pos_w  = inner._fixed_asset.data.root_pos_w.clone()   # (N, 3)
    box_quat_w = inner._fixed_asset.data.root_quat_w.clone()  # (N, 4) wxyz

    # Apply position offset in world frame.
    off = torch.tensor(_pos_offset, device=device, dtype=torch.float32).unsqueeze(0)
    lid_pos_w = box_pos_w + off

    # Apply rotation offset on top of box orientation.
    r, p, y = _euler_offset
    delta_q = quat_from_euler_xyz(
        torch.tensor([r], device=device),
        torch.tensor([p], device=device),
        torch.tensor([y], device=device),
    )                                             # (1, 4)
    delta_q = delta_q.expand(num_envs, -1)
    lid_quat_w = quat_mul(box_quat_w, delta_q)   # (N, 4)

    pose_w   = torch.cat([lid_pos_w, lid_quat_w], dim=-1)
    zero_vel = torch.zeros((num_envs, 6), device=device)

    inner._held_asset.write_root_pose_to_sim(pose_w)
    inner._held_asset.write_root_velocity_to_sim(zero_vel)
    inner._held_asset.reset()


# ---------------------------------------------------------------------------
# Visualisation markers
# ---------------------------------------------------------------------------
_markers: VisualizationMarkers | None = None


def _get_markers() -> VisualizationMarkers:
    global _markers
    if _markers is None:
        cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/SnapFitMarkers",
            markers={
                "clip": sim_utils.SphereCfg(
                    radius=0.005,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.4, 1.0)),
                ),
                "hole": sim_utils.SphereCfg(
                    radius=0.005,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 0.2)),
                ),
                "kp_held": sim_utils.SphereCfg(
                    radius=0.003,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.9, 0.2)),
                ),
                "kp_target": sim_utils.SphereCfg(
                    radius=0.003,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.8, 0.0)),
                ),
                # Contact calibration (indices 4–9)
                "rear_full": sim_utils.SphereCfg(
                    radius=0.0025,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
                ),
                "front_full": sim_utils.SphereCfg(
                    radius=0.0025,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.85, 0.95)),
                ),
                "rear_L": sim_utils.SphereCfg(
                    radius=0.003,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.55, 0.1)),
                ),
                "rear_R": sim_utils.SphereCfg(
                    radius=0.003,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.2, 0.9)),
                ),
                "front_L": sim_utils.SphereCfg(
                    radius=0.003,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 1.0, 0.2)),
                ),
                "front_R": sim_utils.SphereCfg(
                    radius=0.003,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 0.55)),
                ),
                "split_dot": sim_utils.SphereCfg(
                    radius=0.002,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.15, 0.15)),
                ),
                "kp_box_goal": sim_utils.SphereCfg(
                    radius=0.0035,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.15, 0.95)),
                ),
            },
        )
        _markers = VisualizationMarkers(cfg)
    return _markers


def _mean_keypoint_dist_factory_style(inner) -> torch.Tensor:
    """Match FactoryEnv _get_factory_rew_dict mean KP distance for box_lid_insert."""
    device = inner.device
    num_envs = inner.num_envs
    ident_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0).expand(num_envs, -1)
    held_pos = inner._held_asset.data.root_pos_w - inner.scene.env_origins
    held_quat = inner._held_asset.data.root_quat_w
    fixed_pos = inner._fixed_asset.data.root_pos_w - inner.scene.env_origins
    fixed_quat = inner._fixed_asset.data.root_quat_w
    n = inner.kp_lid_local.shape[1]
    kh = torch.zeros((num_envs, n, 3), device=device)
    kf = torch.zeros((num_envs, n, 3), device=device)
    for i in range(n):
        _, kh[:, i] = torch_utils.tf_combine(held_quat, held_pos, ident_q, inner.kp_lid_local[:, i])
        _, kf[:, i] = torch_utils.tf_combine(fixed_quat, fixed_pos, ident_q, inner.kp_box_local[:, i])
    return torch.norm(kh - kf, p=2, dim=-1).mean(-1)


def _draw_clip_hole_markers(inner):
    markers  = _get_markers()
    device   = inner.device
    num_envs = inner.num_envs

    ident_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0).expand(num_envs, -1)

    held_pos   = inner._held_asset.data.root_pos_w  - inner.scene.env_origins
    held_quat  = inner._held_asset.data.root_quat_w
    fixed_pos  = inner._fixed_asset.data.root_pos_w - inner.scene.env_origins
    fixed_quat = inner._fixed_asset.data.root_quat_w
    env_orig   = inner.scene.env_origins

    lc_loc = torch.tensor(_LEFT_CLIP_LID_LOCAL,  device=device).expand(num_envs, -1).clone()
    rc_loc = torch.tensor(_RIGHT_CLIP_LID_LOCAL, device=device).expand(num_envs, -1).clone()
    lh_loc = torch.tensor(_LEFT_HOLE_BOX_LOCAL,  device=device).expand(num_envs, -1).clone()
    rh_loc = torch.tensor(_RIGHT_HOLE_BOX_LOCAL, device=device).expand(num_envs, -1).clone()

    _, lc_env = torch_utils.tf_combine(held_quat,  held_pos,  ident_q, lc_loc)
    _, rc_env = torch_utils.tf_combine(held_quat,  held_pos,  ident_q, rc_loc)
    _, lh_env = torch_utils.tf_combine(fixed_quat, fixed_pos, ident_q, lh_loc)
    _, rh_env = torch_utils.tf_combine(fixed_quat, fixed_pos, ident_q, rh_loc)

    # clip=0, hole=1
    translations_list  = [lc_env + env_orig, rc_env + env_orig, lh_env + env_orig, rh_env + env_orig]
    marker_indices_list = [
        torch.zeros(2 * num_envs, dtype=torch.int32),
        torch.ones( 2 * num_envs, dtype=torch.int32),
    ]

    # Per-episode keypoints: lid (green=2), box eased Y (yellow=3), box success Y goal (purple=11).
    if hasattr(inner, "kp_lid_local") and hasattr(inner, "kp_box_y_target"):
        n = inner.kp_lid_local.shape[1]
        for i in range(n):
            _, kp_h = torch_utils.tf_combine(held_quat, held_pos, ident_q, inner.kp_lid_local[:, i])
            _, kp_t = torch_utils.tf_combine(fixed_quat, fixed_pos, ident_q, inner.kp_box_local[:, i])
            goal_loc = torch.stack(
                [
                    inner.kp_lid_local[:, i, 0],
                    inner.kp_box_y_target[:, i],
                    inner.kp_lid_local[:, i, 2],
                ],
                dim=1,
            )
            _, kp_g = torch_utils.tf_combine(fixed_quat, fixed_pos, ident_q, goal_loc)
            translations_list.append(kp_h + env_orig)
            translations_list.append(kp_t + env_orig)
            translations_list.append(kp_g + env_orig)
            marker_indices_list.append(torch.full((num_envs,), 2, dtype=torch.int32))   # kp_held
            marker_indices_list.append(torch.full((num_envs,), 3, dtype=torch.int32))   # kp_box current
            marker_indices_list.append(torch.full((num_envs,), 11, dtype=torch.int32))  # kp_box goal Y
    elif hasattr(inner, "kp_lid_local"):
        n = inner.kp_lid_local.shape[1]
        for i in range(n):
            _, kp_h = torch_utils.tf_combine(held_quat, held_pos, ident_q, inner.kp_lid_local[:, i])
            _, kp_t = torch_utils.tf_combine(fixed_quat, fixed_pos, ident_q, inner.kp_box_local[:, i])
            translations_list.append(kp_h + env_orig)
            translations_list.append(kp_t + env_orig)
            marker_indices_list.append(torch.full((num_envs,), 2, dtype=torch.int32))
            marker_indices_list.append(torch.full((num_envs,), 3, dtype=torch.int32))

    # --- Contact-init guides (box rear top edge vs lid front edge), from task cfg ---
    tcfg = inner.cfg_task
    bx_lo, bx_hi = tcfg.box_contact_rear_edge_x_range
    bx_mid = 0.5 * (bx_lo + bx_hi)
    by = float(tcfg.box_contact_rear_edge_y_local)
    bz = float(tcfg.box_contact_rear_edge_z_local)
    lx_lo, lx_hi = tcfg.lid_contact_front_edge_x_range
    lx_mid = 0.5 * (lx_lo + lx_hi)
    ly = float(tcfg.lid_contact_front_edge_y_local)
    lz = float(tcfg.lid_contact_front_edge_z_local)

    def _append_box_line(x0, x1, nseg, midx):
        xs = torch.linspace(float(x0), float(x1), nseg, device=device)
        for xi in xs:
            loc = torch.tensor([[xi, by, bz]], device=device).expand(num_envs, -1)
            _, pw = torch_utils.tf_combine(fixed_quat, fixed_pos, ident_q, loc)
            translations_list.append(pw + env_orig)
            marker_indices_list.append(torch.full((num_envs,), midx, dtype=torch.int32))

    def _append_lid_line(x0, x1, nseg, midx):
        xs = torch.linspace(float(x0), float(x1), nseg, device=device)
        for xi in xs:
            loc = torch.tensor([[xi, ly, lz]], device=device).expand(num_envs, -1)
            _, pw = torch_utils.tf_combine(held_quat, held_pos, ident_q, loc)
            translations_list.append(pw + env_orig)
            marker_indices_list.append(torch.full((num_envs,), midx, dtype=torch.int32))

    # Full span + left/right halves (same split as factory_env contact init).
    _append_box_line(bx_lo, bx_hi, _REAR_EDGE_SAMPLES, 4)
    _append_lid_line(lx_lo, lx_hi, _FRONT_EDGE_SAMPLES, 5)
    _append_box_line(bx_lo, bx_mid, max(2, _REAR_EDGE_SAMPLES // 2), 6)
    _append_box_line(bx_mid, bx_hi, max(2, _REAR_EDGE_SAMPLES // 2), 7)
    _append_lid_line(lx_lo, lx_mid, max(2, _FRONT_EDGE_SAMPLES // 2), 8)
    _append_lid_line(lx_mid, lx_hi, max(2, _FRONT_EDGE_SAMPLES // 2), 9)

    # Midpoint markers (red) on box rear and lid front guides.
    loc_bm = torch.tensor([[bx_mid, by, bz]], device=device).expand(num_envs, -1)
    _, pw_bm = torch_utils.tf_combine(fixed_quat, fixed_pos, ident_q, loc_bm)
    translations_list.append(pw_bm + env_orig)
    marker_indices_list.append(torch.full((num_envs,), 10, dtype=torch.int32))
    loc_lm = torch.tensor([[lx_mid, ly, lz]], device=device).expand(num_envs, -1)
    _, pw_lm = torch_utils.tf_combine(held_quat, held_pos, ident_q, loc_lm)
    translations_list.append(pw_lm + env_orig)
    marker_indices_list.append(torch.full((num_envs,), 10, dtype=torch.int32))

    translations   = torch.cat(translations_list,   dim=0)
    marker_indices = torch.cat(marker_indices_list, dim=0)
    identity_q     = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(len(translations), -1)

    markers.visualize(translations=translations, orientations=identity_q, marker_indices=marker_indices)


# ---------------------------------------------------------------------------
# Success / failure rate tracking
# ---------------------------------------------------------------------------
_success_count  = 0   # cumulative steps where ANY env succeeded
_engaged_count  = 0   # cumulative steps where ANY env was engaged (clips in groove)
_total_count    = 0   # cumulative steps checked
_recent_window:         deque = deque(maxlen=100)  # rolling 100-step success fractions
_recent_engaged_window: deque = deque(maxlen=100)  # rolling 100-step engaged fractions


def _track_success(successes: torch.Tensor, engaged: torch.Tensor):
    """Call every step to update success-rate counters."""
    global _success_count, _engaged_count, _total_count
    _total_count   += 1
    _success_count += int(successes.any().item())
    _engaged_count += int(engaged.any().item())
    _recent_window.append(successes.float().mean().item())
    _recent_engaged_window.append(engaged.float().mean().item())


def _reset_rate_counters():
    global _success_count, _engaged_count, _total_count
    _success_count = 0
    _engaged_count = 0
    _total_count   = 0
    _recent_window.clear()
    _recent_engaged_window.clear()


# ---------------------------------------------------------------------------
# Success info print
# ---------------------------------------------------------------------------

def _print_success_info(inner, step, successes: torch.Tensor, engaged: torch.Tensor):
    """Print clip positions, distances to hole targets, and success/engaged rates."""
    device   = inner.device
    num_envs = inner.num_envs
    ident_q  = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0).expand(num_envs, -1)

    held_pos   = inner._held_asset.data.root_pos_w  - inner.scene.env_origins
    held_quat  = inner._held_asset.data.root_quat_w
    fixed_pos  = inner._fixed_asset.data.root_pos_w - inner.scene.env_origins
    fixed_quat = inner._fixed_asset.data.root_quat_w

    lc_loc = torch.tensor(_LEFT_CLIP_LID_LOCAL,  device=device).expand(num_envs, -1).clone()
    rc_loc = torch.tensor(_RIGHT_CLIP_LID_LOCAL, device=device).expand(num_envs, -1).clone()

    _, lc_w = torch_utils.tf_combine(held_quat, held_pos, ident_q, lc_loc)
    _, rc_w = torch_utils.tf_combine(held_quat, held_pos, ident_q, rc_loc)

    lc_box = quat_apply_inverse(fixed_quat, lc_w - fixed_pos)   # (N, 3) box-local
    rc_box = quat_apply_inverse(fixed_quat, rc_w - fixed_pos)   # (N, 3) box-local

    # Hole targets in box-local frame (use env 0 for display).
    lh_tgt = torch.tensor(_LEFT_HOLE_TARGET,  device=device)
    rh_tgt = torch.tensor(_RIGHT_HOLE_TARGET, device=device)

    lc0 = lc_box[0].cpu()
    rc0 = rc_box[0].cpu()
    lh0 = lh_tgt.cpu()
    rh0 = rh_tgt.cpu()

    dl  = lc0 - lh0   # (3,) delta clip→hole, left
    dr  = rc0 - rh0   # (3,) delta clip→hole, right
    dl_norm = dl.norm().item() * 1000   # mm
    dr_norm = dr.norm().item() * 1000   # mm

    def _fmt_pos(v):
        return f"X={v[0]*1000:+6.1f} Y={v[1]*1000:+6.1f} Z={v[2]*1000:+6.1f} mm"

    def _fmt_dist(d, norm):
        return (f"ΔX={d[0]*1000:+5.1f} ΔY={d[1]*1000:+5.1f} ΔZ={d[2]*1000:+5.1f} |d|={norm:5.2f} mm")

    # Status for env 0.
    is_success = successes[0].item()
    is_engaged = engaged[0].item()
    if is_success:
        state_str = "SUCCESS"
    elif is_engaged:
        state_str = "ENGAGED"
    else:
        state_str = "FAILURE"

    # Rate stats.
    cumul_success_rate  = (_success_count / _total_count * 100) if _total_count > 0 else 0.0
    cumul_engaged_rate  = (_engaged_count / _total_count * 100) if _total_count > 0 else 0.0
    recent_success_frac = (sum(_recent_window) / len(_recent_window) * 100) if _recent_window else 0.0
    recent_engaged_frac = (sum(_recent_engaged_window) / len(_recent_engaged_window) * 100) if _recent_engaged_window else 0.0

    off_mm  = [x * 1000 for x in _pos_offset]
    eul_deg = [math.degrees(x) for x in _euler_offset]

    print(
        f"[step {step:5d}] env0={state_str}"
        f"  engaged={_engaged_count}/{_total_count}({cumul_engaged_rate:.1f}%)"
        f"  success={_success_count}/{_total_count}({cumul_success_rate:.1f}%)"
        f"  recent-engaged={recent_engaged_frac:.1f}%  recent-success={recent_success_frac:.1f}%\n"
        f"  pos=[{off_mm[0]:+.1f},{off_mm[1]:+.1f},{off_mm[2]:+.1f}]mm"
        f"  euler=[{eul_deg[0]:+.1f},{eul_deg[1]:+.1f},{eul_deg[2]:+.1f}]°\n"
        f"  left  clip pos: {_fmt_pos(lc0)}  →  dist-to-hole: {_fmt_dist(dl, dl_norm)}\n"
        f"  right clip pos: {_fmt_pos(rc0)}  →  dist-to-hole: {_fmt_dist(dr, dr_norm)}\n"
        f"  (hole targets — left: X=-25.2 Y=-44.4 Z=+25.0 mm | right: X=+24.3 Y=-44.4 Z=+25.0 mm)"
    )

    # --- Progressive box KP Y (Factory reward) ---
    if hasattr(inner, "kp_box_y_target"):
        cfg = inner.cfg_task
        kdist = _mean_keypoint_dist_factory_style(inner)
        thr = float(cfg.kp_advance_threshold)
        adv = bool((kdist[0] < thr).item())
        y_cur = inner.kp_box_local[0, :, 1].cpu()
        y_tgt = inner.kp_box_y_target[0].cpu()
        dy = (y_cur - y_tgt) * 1000.0
        n_kp = y_cur.shape[0]
        max_dy = dy.abs().max().item()
        mean_dy = dy.abs().mean().item()
        k0 = min(3, n_kp)
        y_pairs = ", ".join(
            f"kp{i}:Ycur={y_cur[i].item()*1000:+.2f} Ytgt={y_tgt[i].item()*1000:+.2f} Δ={dy[i].item():+.2f}mm"
            for i in range(k0)
        )
        print(
            f"  --- KP GUIDE (box Y ease, RJ45-style) ---\n"
            f"  mean_kp_dist={kdist[0].item()*1000:.3f} mm  threshold={thr*1000:.3f} mm  "
            f"advance_active={'YES' if adv else 'no'}\n"
            f"  y_start(cfg)={cfg.kp_advance_y_start*1000:+.2f} mm  step={cfg.kp_advance_y_step*1000:.4f} mm/step  "
            f"|Ycur−Ytgt| mean={mean_dy:.3f} max={max_dy:.3f} mm\n"
            f"  {y_pairs}{' ...' if n_kp > k0 else ''}\n"
            f"  markers: GREEN=lid  YELLOW=box(current Y)  PURPLE=box(goal Y) → yellow meets purple when eased"
        )

    # --- ORIENTATION CHECK ---
    fixed_r, fixed_p, fixed_y = euler_xyz_from_quat(fixed_quat[0:1])
    held_r,  held_p,  held_y  = euler_xyz_from_quat(held_quat[0:1])
    rel_quat = quat_mul(quat_conjugate(fixed_quat[0:1]), held_quat[0:1])
    rel_r, rel_p, rel_y = euler_xyz_from_quat(rel_quat)
    rq = rel_quat[0].cpu()
    print(
        f"  --- ORIENTATION CHECK ---\n"
        f"  fixed  RPY: [{math.degrees(fixed_r.item()):+7.2f}, {math.degrees(fixed_p.item()):+7.2f}, {math.degrees(fixed_y.item()):+7.2f}] deg\n"
        f"  held   RPY: [{math.degrees(held_r.item()):+7.2f}, {math.degrees(held_p.item()):+7.2f}, {math.degrees(held_y.item()):+7.2f}] deg\n"
        f"  rel    RPY: [{math.degrees(rel_r.item()):+7.2f}, {math.degrees(rel_p.item()):+7.2f}, {math.degrees(rel_y.item()):+7.2f}] deg  <- should be [0,0,0]\n"
        f"  rel   quat: [{rq[0]:.4f}, {rq[1]:.4f}, {rq[2]:.4f}, {rq[3]:.4f}]  (wxyz, should be [1,0,0,0])"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    env_cfg = parse_env_cfg(
        "Isaac-Forge-BoxLidInsert-Direct-v0",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=True,
    )

    if args_cli.freeze_box:
        env_cfg.task.fixed_asset_init_pos_noise = [0.0, 0.0, 0.0]
        env_cfg.task.fixed_asset_init_orn_range_deg = 0.0

    env_cfg.task.hand_init_pos        = [0.0, 0.0, 0.5]  # park arm 50 cm above box, out of the way
    env_cfg.task.hand_init_pos_noise  = [0.0, 0.0, 0.0]
    env_cfg.task.hand_init_orn_noise  = [0.0, 0.0, 0.0]
    env_cfg.task.held_asset_pos_noise = [0.0, 0.0, 0.0]

    env   = gym.make("Isaac-Forge-BoxLidInsert-Direct-v0", cfg=env_cfg)
    inner = env.unwrapped

    env.reset()
    _init_keyboard()

    t = env_cfg.task
    print(
        "\n[CONTROLS]  Arrow=XY  Q/E=Z  I/K=pitch  J/L=yaw  U/O=roll  "
        "Shift=5×speed  C=contact-sample  R=reset\n"
        "[CONTACT CFG]  box rear: "
        f"x∈{tuple(t.box_contact_rear_edge_x_range)} y={t.box_contact_rear_edge_y_local:+.4f} z={t.box_contact_rear_edge_z_local:+.4f}\n"
        "               lid front: "
        f"x∈{tuple(t.lid_contact_front_edge_x_range)} y={t.lid_contact_front_edge_y_local:+.4f} z={t.lid_contact_front_edge_z_local:+.4f}\n"
        "               contact RPY jitter (deg): "
        f"roll={tuple(t.contact_init_roll_range_deg)} pitch={tuple(t.contact_init_pitch_range_deg)} "
        f"yaw={tuple(t.contact_init_yaw_range_deg)}\n"
        "[MARKERS]  GREEN=lid KP  YELLOW=box KP (eased Y)  PURPLE=box KP goal Y  "
        "WHITE/CYAN/ORANGE…=contact guides  RED dot=mid X\n"
        "[KP Y EASE]  "
        f"y_start={t.kp_advance_y_start*1000:+.2f} mm  step={t.kp_advance_y_step*1000:.4f} mm  "
        f"advance_when_mean_kp_dist<{t.kp_advance_threshold*1000:.3f} mm\n"
    )

    # Pre-compute frozen joint state (default pose, gripper open to lid handle width).
    _all_env_ids = torch.arange(inner.num_envs, device=inner.device)
    _frozen_joint_pos = inner._robot.data.default_joint_pos.clone()
    _frozen_joint_pos[:, :7] = torch.tensor(
        inner.cfg.ctrl.reset_joints, device=inner.device
    ).unsqueeze(0).expand(inner.num_envs, -1)
    gripper_width = inner.cfg_task.held_asset_cfg.diameter / 2 * 1.25
    _frozen_joint_pos[:, 7:] = gripper_width
    _frozen_joint_vel = torch.zeros_like(_frozen_joint_pos)

    step = 0
    _c_last = False
    while simulation_app.is_running():
        with torch.inference_mode():
            c_now = _key(Ki.C)
            if c_now and not _c_last:
                _teleport_contact(inner)
            _c_last = c_now

            actions = torch.zeros(env.action_space.shape, device=inner.device)
            _, _, done, trunc, _ = env.step(actions)
            step += 1

            # Freeze robot arm: override joint state and controller target every step.
            inner._robot.write_joint_state_to_sim(_frozen_joint_pos, _frozen_joint_vel)
            inner._robot.set_joint_position_target(_frozen_joint_pos)
            inner._robot.set_joint_effort_target(_frozen_joint_vel)

            _poll_keys_and_move_lid(inner)
            _draw_clip_hole_markers(inner)

            # Compute engaged (clips inside groove) and success (clips fully seated).
            engaged   = inner._get_curr_successes(success_threshold=0.9)
            successes = inner._get_curr_successes(success_threshold=inner.cfg_task.success_threshold)
            _track_success(successes, engaged)

            # Print detailed info every 10 steps.
            if step % 10 == 0:
                _print_success_info(inner, step, successes, engaged)

            if torch.any(done | trunc):
                env.reset()
                step = 0
                _reset_rate_counters()
                print("[INFO] Episode reset — rate counters cleared.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
