"""Visualize and interactively test the RJ45Insert engage/success/keypoint checks.

The male plug is moved directly via keyboard — no robot physics involved.
Use this to verify:
  1. Keypoint positions (GREEN=plug body, YELLOW=socket body) converge when origins align
  2. Engage fires when plug tip is within 5mm XY, <40mm above opening, yaw<20 deg, tilt<15 deg
  3. Success fires when tip is >=7mm inside socket AND xy<3mm

Keyboard controls:
  Arrow Up/Down    — plug +Y/-Y  (socket local frame)
  Arrow Left/Right — plug -X/+X  (socket local frame)
  Q / E            — plug +Z/-Z  (socket local frame)
  I / K            — pitch +/-
  J / L            — yaw   +/-
  U / O            — roll  +/-
  Hold Shift       — 5x speed
  C                — CONTACT init: random r/p/y + random male/female contact points
  R                — ORIGIN-COINCIDE: place plug origin at socket origin
                     (keypoints should overlap if origins coincide at success)
  F                — ENGAGE position: tip 30 mm above socket opening, aligned
  G                — SUCCESS position: tip at success-threshold depth inside socket

Coloured sphere markers (updated every frame):
  BLUE   -- connector tip  (plug local [0,0,-3mm])
  ORANGE -- socket opening centre  (socket local [0,0,0] = socket origin)
  GREEN  -- per-episode uniformly sampled kp_rj45_male_local in plug frame   (keypoints_held)
  YELLOW -- per-episode uniformly sampled kp_rj45_female_local in socket frame (keypoints_fixed)
  WHITE  -- adjustable female rear-edge guide (female local frame)
  CYAN   -- adjustable male bottom patch (male local frame)
  MAGENTA-- success target (get_target_held_base_pose in socket local frame)
  RED    -- +X axis (both male & female local frames)
  GREEN  -- +Y axis (both male & female local frames, pure-green dotted line)
  BLUE   -- +Z axis (both male & female local frames, pure-blue dotted line)
  GREEN/YELLOW overlap -> keypoint_dist = 0 -> plug at full insertion target

Usage
-----
./isaaclab.sh -p scripts/environments/visualize_rj45_insert.py --num_envs 1 --freeze_socket

Flags
-----
--freeze_socket   Disable socket randomisation for a clean axis-aligned view.
--num_envs N      Number of parallel environments (default: 1).
--use_kuka        Use the Kuka variant (default: Franka).
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Interactive RJ45Insert visualizer (no robot).")
parser.add_argument("--num_envs",     type=int, default=1)
parser.add_argument("--freeze_socket", action="store_true", default=False)
parser.add_argument("--use_kuka",     action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math

import carb
import carb.input
import omni.appwindow
import torch
import gymnasium as gym

import isaacsim.core.utils.torch as torch_utils
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_conjugate, quat_from_euler_xyz, quat_mul

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


# ---------------------------------------------------------------------------
# RJ45 geometry constants (metres, post-STL->USD 0.001 scale).
# New STL geometry: both origins at mating interface.
#   Female origin = opening face  → _OPENING_LOCAL = (0, 0, 0)
#   Male origin   = connector mating face → physical tip is 3 mm below origin
# ---------------------------------------------------------------------------
_TIP_LOCAL      = (0.0, 0.0, -0.003)   # connector tip in plug USD frame
_OPENING_LOCAL  = (0.0, 0.0,  0.000)   # socket opening centre = socket USD origin

# Teleport offsets: computed from task config in _init_teleport_offsets().
# Kept as module globals so _teleport / _poll_keys_and_move_plug can use them.
_XY_OFFSET_ORIGINS = (0.0, 0.0)
_Z_OFFSET_ENGAGE  = 0.0
_Z_OFFSET_SUCCESS = 0.0
_Z_OFFSET_ORIGINS = 0.0


def _init_teleport_offsets(cfg_task):
    """Derive teleport positions from the task config (single source of truth).

    cfg_task.socket_target_y_local  — cavity centre Y offset in socket frame
    cfg_task.socket_target_z_local  — tip Z at full insertion in socket frame
    held_base_z_offset              — tip offset below plug USD origin (from RJ45MaleCfg)
    """
    global _XY_OFFSET_ORIGINS, _Z_OFFSET_ENGAGE, _Z_OFFSET_SUCCESS, _Z_OFFSET_ORIGINS

    target_y = cfg_task.socket_target_y_local
    target_z = cfg_task.socket_target_z_local
    held_base_z_offset = cfg_task.held_asset_cfg.base_height  # -0.003 for RJ45

    # Plug-origin Z at full insertion = target_tip_Z - held_base_z_offset
    plug_origin_z_insert = target_z - held_base_z_offset

    _XY_OFFSET_ORIGINS = (0.0, target_y)
    _Z_OFFSET_ORIGINS  = plug_origin_z_insert
    _Z_OFFSET_ENGAGE   = plug_origin_z_insert + 0.030 + abs(held_base_z_offset)
    # Success boundary: tip at height_threshold depth
    height_threshold = cfg_task.fixed_asset_cfg.height + cfg_task.success_threshold
    success_tip_z = target_z + height_threshold
    _Z_OFFSET_SUCCESS = success_tip_z - held_base_z_offset


# ---------------------------------------------------------------------------
# Keyboard input
# ---------------------------------------------------------------------------
MOVE_STEP = 0.001    # metres per frame
ROT_STEP  = 0.01     # radians per frame
SHIFT_MUL = 5.0

Ki = carb.input.KeyboardInput

_key_states: dict = {}
_key_sub    = None


def _on_keyboard_event(event, *args, **kwargs):
    if event.type == carb.input.KeyboardEventType.KEY_PRESS:
        _key_states[event.input] = True
    elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
        _key_states[event.input] = False
    return True


def _init_keyboard():
    global _key_sub
    kbi = carb.input.acquire_input_interface()
    kbw = omni.appwindow.get_default_app_window().get_keyboard()
    _key_sub = kbi.subscribe_to_keyboard_events(kbw, _on_keyboard_event)
    print("[keyboard] subscribed -- click the viewport once to give it focus.")


def _key(k) -> bool:
    return _key_states.get(k, False)


# ---------------------------------------------------------------------------
# Plug pose state (offset from socket origin in socket local frame).
# R/F/G snap these to preset positions.
# ---------------------------------------------------------------------------
_pos_z_offset: float = _Z_OFFSET_ENGAGE   # initial position: engage zone
_pos_xy_offset = [0.0, 0.0]               # [x, y] additional offset
_euler_offset  = [0.0, 0.0, 0.0]          # [roll, pitch, yaw]
_BOTTOM_PATCH_X_SAMPLES = 5
_BOTTOM_PATCH_Y_SAMPLES = 4
_REAR_EDGE_MARKER_SAMPLES = 11
_AXIS_LENGTH  = 0.020   # 20 mm total axis display length
_AXIS_SAMPLES = 3       # dotted spheres per axis


def _teleport(z_off: float, label: str, xy_off=(0.0, 0.0)):
    global _pos_z_offset, _pos_xy_offset, _euler_offset
    _pos_z_offset  = z_off
    _pos_xy_offset = [xy_off[0], xy_off[1]]
    _euler_offset  = [0.0, 0.0, 0.0]
    print(f"[TELEPORT -> {label}]  plug_origin = socket_origin + X={xy_off[0]*1000:.1f} Y={xy_off[1]*1000:.1f} Z={z_off*1000:.1f} mm")


def _teleport_contact(inner):
    global _pos_z_offset, _pos_xy_offset, _euler_offset

    device = inner.device
    num_envs = inner.num_envs
    socket_pos_w = inner._fixed_asset.data.root_pos_w.clone()
    socket_quat_w = inner._fixed_asset.data.root_quat_w.clone()
    ident_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0).expand(num_envs, -1)

    roll = math.radians(float(torch.empty(1).uniform_(*inner.cfg_task.contact_init_roll_range_deg).item()))
    pitch = math.radians(float(torch.empty(1).uniform_(*inner.cfg_task.contact_init_pitch_range_deg).item()))
    yaw = math.radians(float(torch.empty(1).uniform_(*inner.cfg_task.contact_init_yaw_range_deg).item()))
    _euler_offset = [roll, pitch, yaw]
    delta_q = quat_from_euler_xyz(
        torch.tensor([roll], device=device),
        torch.tensor([pitch], device=device),
        torch.tensor([yaw], device=device),
    ).expand(num_envs, -1)
    plug_quat_w = quat_mul(socket_quat_w, delta_q)

    female_point_local = torch.zeros((num_envs, 3), device=device)
    female_point_local[:, 0] = float(torch.empty(1).uniform_(*inner.cfg_task.female_rear_edge_x_range_local).item())
    female_point_local[:, 1] = float(inner.cfg_task.female_rear_edge_y_local)
    female_point_local[:, 2] = float(inner.cfg_task.female_rear_edge_z_local)
    _, female_point_w = torch_utils.tf_combine(socket_quat_w, socket_pos_w, ident_q, female_point_local)

    male_point_local = torch.zeros((num_envs, 3), device=device)
    male_point_local[:, 0] = float(torch.empty(1).uniform_(*inner.cfg_task.male_bottom_patch_x_range_local).item())
    male_point_local[:, 1] = float(torch.empty(1).uniform_(*inner.cfg_task.male_bottom_patch_y_range_local).item())
    male_point_local[:, 2] = float(inner.cfg_task.male_bottom_patch_z_local)

    male_point_offset_w = torch_utils.quat_rotate(plug_quat_w, male_point_local)
    plug_pos_w = female_point_w - male_point_offset_w
    offset_w = plug_pos_w[0] - socket_pos_w[0]

    _pos_xy_offset = [offset_w[0].item(), offset_w[1].item()]
    _pos_z_offset = offset_w[2].item()

    eul_deg = [math.degrees(v) for v in _euler_offset]
    female_dbg = female_point_local[0].tolist()
    male_dbg = male_point_local[0].tolist()
    print(
        "[TELEPORT -> CONTACT init]  "
        f"plug_origin = socket_origin + X={_pos_xy_offset[0]*1000:.1f} "
        f"Y={_pos_xy_offset[1]*1000:.1f} Z={_pos_z_offset*1000:.1f} mm  "
        f"rpy=[{eul_deg[0]:+.1f}, {eul_deg[1]:+.1f}, {eul_deg[2]:+.1f}] deg  "
        f"female_local=[{female_dbg[0]*1000:+.1f}, {female_dbg[1]*1000:+.1f}, {female_dbg[2]*1000:+.1f}] mm  "
        f"male_local=[{male_dbg[0]*1000:+.1f}, {male_dbg[1]*1000:+.1f}, {male_dbg[2]*1000:+.1f}] mm"
    )


def _poll_keys_and_move_plug(inner):
    global _pos_z_offset, _pos_xy_offset, _euler_offset

    mul = SHIFT_MUL if (_key(Ki.LEFT_SHIFT) or _key(Ki.RIGHT_SHIFT)) else 1.0
    ms  = MOVE_STEP * mul
    rs  = ROT_STEP  * mul

    if _key(Ki.UP):    _pos_xy_offset[1] += ms
    if _key(Ki.DOWN):  _pos_xy_offset[1] -= ms
    if _key(Ki.LEFT):  _pos_xy_offset[0] -= ms
    if _key(Ki.RIGHT): _pos_xy_offset[0] += ms
    if _key(Ki.Q):     _pos_z_offset     += ms
    if _key(Ki.E):     _pos_z_offset     -= ms
    if _key(Ki.I):     _euler_offset[1]  += rs
    if _key(Ki.K):     _euler_offset[1]  -= rs
    if _key(Ki.J):     _euler_offset[2]  -= rs
    if _key(Ki.L):     _euler_offset[2]  += rs
    if _key(Ki.U):     _euler_offset[0]  += rs
    if _key(Ki.O):     _euler_offset[0]  -= rs

    device   = inner.device
    num_envs = inner.num_envs

    socket_pos_w  = inner._fixed_asset.data.root_pos_w.clone()   # (N,3) world
    socket_quat_w = inner._fixed_asset.data.root_quat_w.clone()  # (N,4) wxyz

    # Compose offset in socket (female) local frame, then rotate to world.
    off_local = torch.tensor(
        [_pos_xy_offset[0], _pos_xy_offset[1], _pos_z_offset],
        device=device, dtype=torch.float32,
    ).unsqueeze(0).expand(num_envs, -1)
    off_world = torch_utils.quat_rotate(socket_quat_w, off_local)
    plug_pos_w = socket_pos_w + off_world

    # Orientation: socket_quat + euler delta.
    r, p, y = _euler_offset
    delta_q = quat_from_euler_xyz(
        torch.tensor([r], device=device),
        torch.tensor([p], device=device),
        torch.tensor([y], device=device),
    ).expand(num_envs, -1)
    plug_quat_w = quat_mul(socket_quat_w, delta_q)

    pose_w   = torch.cat([plug_pos_w, plug_quat_w], dim=-1)
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
            prim_path="/Visuals/RJ45InsertMarkers",
            markers={
                "tip":      sim_utils.SphereCfg(  # plug tip — BLUE
                    radius=0.004,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.4, 1.0)),
                ),
                "opening":  sim_utils.SphereCfg(  # socket opening — ORANGE
                    radius=0.004,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.0)),
                ),
                "kp_held":  sim_utils.SphereCfg(  # plug body keypoints — GREEN
                    radius=0.003,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.9, 0.2)),
                ),
                "kp_target": sim_utils.SphereCfg( # socket body keypoints — YELLOW
                    radius=0.003,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.8, 0.0)),
                ),
                "rear_edge": sim_utils.SphereCfg(  # adjustable female rear-edge guide — WHITE
                    radius=0.0024,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0)),
                ),
                "bottom_patch": sim_utils.SphereCfg(  # adjustable male bottom patch — CYAN
                    radius=0.0026,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.95, 0.95)),
                ),
                "success_target": sim_utils.SphereCfg(  # success target — MAGENTA
                    radius=0.004,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.2, 0.9)),
                ),
                "axis_x": sim_utils.SphereCfg(       # +X axis — RED
                    radius=0.0015,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
                "axis_y": sim_utils.SphereCfg(       # +Y axis — pure GREEN
                    radius=0.0015,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                ),
                "axis_z": sim_utils.SphereCfg(       # +Z axis — pure BLUE
                    radius=0.0015,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
                ),
            },
        )
        _markers = VisualizationMarkers(cfg)
    return _markers


def _draw_markers(inner):
    markers  = _get_markers()
    device   = inner.device
    num_envs = inner.num_envs

    held_pos   = inner._held_asset.data.root_pos_w  - inner.scene.env_origins
    held_quat  = inner._held_asset.data.root_quat_w
    fixed_pos  = inner._fixed_asset.data.root_pos_w - inner.scene.env_origins
    fixed_quat = inner._fixed_asset.data.root_quat_w
    env_orig   = inner.scene.env_origins

    ident_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0).expand(num_envs, -1)

    # Connector tip (plug local) and socket opening.
    tip_loc     = torch.tensor(_TIP_LOCAL,     device=device).expand(num_envs, -1).clone()
    opening_loc = torch.tensor(_OPENING_LOCAL, device=device).expand(num_envs, -1).clone()

    _, tip_env     = torch_utils.tf_combine(held_quat,  held_pos,  ident_q, tip_loc)
    _, opening_env = torch_utils.tf_combine(fixed_quat, fixed_pos, ident_q, opening_loc)

    # Success target from get_target_held_base_pose (MAGENTA).
    from isaaclab_tasks.direct.factory import factory_utils as _fu
    target_pos, _ = _fu.get_target_held_base_pose(
        fixed_pos, fixed_quat, inner.cfg_task.name,
        inner.cfg_task.fixed_asset_cfg, num_envs, device,
        task_cfg=inner.cfg_task,
    )

    translations_list   = [tip_env + env_orig, opening_env + env_orig, target_pos + env_orig]
    marker_indices_list = [
        torch.zeros(num_envs, dtype=torch.int32, device=device),   # tip=0 (BLUE)
        torch.ones( num_envs, dtype=torch.int32, device=device),   # opening=1 (ORANGE)
        torch.full((num_envs,), 6, dtype=torch.int32, device=device),  # success_target=6 (MAGENTA)
    ]

    # Per-episode body keypoints — use held_base / target_held_base so they
    # coincide at the calibrated full-insertion position (not raw USD origins).
    # if hasattr(inner, "kp_rj45_local"):
    #     from isaaclab_tasks.direct.factory import factory_utils
    #     held_base_pos, held_base_quat = factory_utils.get_held_base_pose(
    #         held_pos, held_quat, inner.cfg_task.name,
    #         inner.cfg_task.fixed_asset_cfg, num_envs, device,
    #     )
    #     target_base_pos, target_base_quat = factory_utils.get_target_held_base_pose(
    #         fixed_pos, fixed_quat, inner.cfg_task.name,
    #         inner.cfg_task.fixed_asset_cfg, num_envs, device,
    #     )
    #     n_kp = inner.kp_rj45_local.shape[1]
    #     for i in range(n_kp):
    #         _, kp_h = torch_utils.tf_combine(held_base_quat,   held_base_pos,   ident_q, inner.kp_rj45_local[:, i])
    #         _, kp_t = torch_utils.tf_combine(target_base_quat, target_base_pos, ident_q, inner.kp_rj45_local[:, i])
    #         translations_list.append(kp_h + env_orig)
    #         translations_list.append(kp_t + env_orig)
    #         marker_indices_list.append(torch.full((num_envs,), 2, dtype=torch.int32, device=device))  # GREEN
    #         marker_indices_list.append(torch.full((num_envs,), 3, dtype=torch.int32, device=device))  # YELLOW
    if hasattr(inner, "kp_rj45_male_local"):
        n_kp = inner.kp_rj45_male_local.shape[1]
        for i in range(n_kp):
            _, kp_h = torch_utils.tf_combine(
                inner.held_quat, inner.held_pos, ident_q, inner.kp_rj45_male_local[:, i]
            )
            translations_list.append(kp_h + env_orig)
            marker_indices_list.append(torch.full((num_envs,), 2, dtype=torch.int32, device=device))  # GREEN

    if hasattr(inner, "kp_rj45_female_local"):
        n_kp = inner.kp_rj45_female_local.shape[1]
        for i in range(n_kp):
            _, kp_f = torch_utils.tf_combine(
                fixed_quat, fixed_pos, ident_q, inner.kp_rj45_female_local[:, i]
            )
            translations_list.append(kp_f + env_orig)
            marker_indices_list.append(torch.full((num_envs,), 3, dtype=torch.int32, device=device))  # YELLOW

    # Adjustable female rear-edge guide, defined directly in the female local frame.
    x_lo, x_hi = inner.cfg_task.female_rear_edge_x_range_local
    rear_edge_local = torch.zeros((_REAR_EDGE_MARKER_SAMPLES, 3), device=device)
    rear_edge_local[:, 0] = torch.linspace(float(x_lo), float(x_hi), _REAR_EDGE_MARKER_SAMPLES, device=device)
    rear_edge_local[:, 1] = float(inner.cfg_task.female_rear_edge_y_local)
    rear_edge_local[:, 2] = float(inner.cfg_task.female_rear_edge_z_local)
    for i in range(_REAR_EDGE_MARKER_SAMPLES):
        _, rear_edge_env = torch_utils.tf_combine(
            fixed_quat, fixed_pos, ident_q, rear_edge_local[i].unsqueeze(0).expand(num_envs, -1)
        )
        translations_list.append(rear_edge_env + env_orig)
        marker_indices_list.append(torch.full((num_envs,), 4, dtype=torch.int32, device=device))  # WHITE

    # Adjustable male bottom patch, defined directly in the male local frame.
    x_lo, x_hi = inner.cfg_task.male_bottom_patch_x_range_local
    y_lo, y_hi = inner.cfg_task.male_bottom_patch_y_range_local
    patch_x = torch.linspace(float(x_lo), float(x_hi), _BOTTOM_PATCH_X_SAMPLES, device=device)
    patch_y = torch.linspace(float(y_lo), float(y_hi), _BOTTOM_PATCH_Y_SAMPLES, device=device)
    patch_xy = torch.cartesian_prod(patch_x, patch_y)
    bottom_patch_local = torch.zeros((patch_xy.shape[0], 3), device=device)
    bottom_patch_local[:, 0:2] = patch_xy
    bottom_patch_local[:, 2] = float(inner.cfg_task.male_bottom_patch_z_local)
    for i in range(bottom_patch_local.shape[0]):
        _, bottom_patch_env = torch_utils.tf_combine(
            held_quat, held_pos, ident_q, bottom_patch_local[i].unsqueeze(0).expand(num_envs, -1)
        )
        translations_list.append(bottom_patch_env + env_orig)
        marker_indices_list.append(torch.full((num_envs,), 5, dtype=torch.int32, device=device))  # CYAN

    # XYZ axis indicators for male (held) and female (fixed) local frames.
    # Dotted spheres along each positive axis: RED=+X, GREEN=+Y, BLUE=+Z.
    axis_dirs = torch.eye(3, device=device)
    for frame_quat, frame_pos in [(held_quat, held_pos), (fixed_quat, fixed_pos)]:
        for axis_idx in range(3):
            for si in range(_AXIS_SAMPLES):
                dist = _AXIS_LENGTH * (si + 1) / _AXIS_SAMPLES
                local_pt = (axis_dirs[axis_idx] * dist).unsqueeze(0).expand(num_envs, -1)
                _, axis_pt_env = torch_utils.tf_combine(
                    frame_quat, frame_pos, ident_q, local_pt,
                )
                translations_list.append(axis_pt_env + env_orig)
                marker_indices_list.append(
                    torch.full((num_envs,), 7 + axis_idx, dtype=torch.int32, device=device)
                )

    translations   = torch.cat(translations_list,   dim=0)
    marker_indices = torch.cat(marker_indices_list, dim=0)
    identity_q     = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(len(translations), -1)
    markers.visualize(translations=translations, orientations=identity_q, marker_indices=marker_indices)


# ---------------------------------------------------------------------------
# Diagnostic print
# ---------------------------------------------------------------------------
def _print_status(step, successes, engaged):
    """One-liner per step: instantaneous counts."""
    n = successes.shape[0]
    n_eng  = engaged.sum().item()
    n_succ = successes.sum().item()
    state0 = "SUCCESS" if successes[0].item() else ("ENGAGED" if engaged[0].item() else "waiting")
    print(f"[step {step:5d}]  engaged={n_eng}/{n}  success={n_succ}/{n}  env0={state0}")


def _print_geometry(inner, step, successes, engaged):
    """Detailed geometry dump every N steps."""
    device   = inner.device
    num_envs = inner.num_envs

    held_pos   = inner._held_asset.data.root_pos_w  - inner.scene.env_origins
    held_quat  = inner._held_asset.data.root_quat_w
    fixed_pos  = inner._fixed_asset.data.root_pos_w - inner.scene.env_origins
    fixed_quat = inner._fixed_asset.data.root_quat_w

    ident_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0).expand(num_envs, -1)

    tip_loc = torch.tensor(_TIP_LOCAL, device=device).expand(num_envs, -1).clone()
    _, tip_w = torch_utils.tf_combine(held_quat, held_pos, ident_q, tip_loc)

    opening_loc = torch.tensor(_OPENING_LOCAL, device=device).expand(num_envs, -1).clone()
    _, opening_w = torch_utils.tf_combine(fixed_quat, fixed_pos, ident_q, opening_loc)

    delta   = (tip_w[0] - opening_w[0]).cpu()
    xy_dist = delta[:2].norm().item() * 1000
    z_disp  = delta[2].item() * 1000

    orig_delta = (held_pos[0] - fixed_pos[0]).cpu()
    orig_xy    = orig_delta[:2].norm().item() * 1000
    orig_z     = orig_delta[2].item() * 1000

    kp_dist_mm = float("nan")
    # if hasattr(inner, "kp_rj45_local"):
    #     n_kp = inner.kp_rj45_local.shape[1]
    #     kp_dists = []
    #     for i in range(n_kp):
    #         _, kp_h = torch_utils.tf_combine(held_quat,  held_pos,  ident_q, inner.kp_rj45_local[:, i])
    #         _, kp_t = torch_utils.tf_combine(fixed_quat, fixed_pos, ident_q, inner.kp_rj45_local[:, i])
    #         kp_dists.append((kp_h[0] - kp_t[0]).norm().item())
    #     kp_dist_mm = sum(kp_dists) / len(kp_dists) * 1000
    if hasattr(inner, "kp_rj45_male_local") and hasattr(inner, "kp_rj45_female_local"):
        n_kp = inner.kp_rj45_male_local.shape[1]
        kp_dists = []
        for i in range(n_kp):
            _, kp_h = torch_utils.tf_combine(
                held_quat, held_pos, ident_q, inner.kp_rj45_male_local[:, i]
            )
            _, kp_f = torch_utils.tf_combine(
                fixed_quat, fixed_pos, ident_q, inner.kp_rj45_female_local[:, i]
            )
            kp_dists.append((kp_h[0] - kp_f[0]).norm().item())
        kp_dist_mm = sum(kp_dists) / len(kp_dists) * 1000

    off_mm  = [_pos_xy_offset[0]*1000, _pos_xy_offset[1]*1000, _pos_z_offset*1000]
    eul_deg = [math.degrees(x) for x in _euler_offset]

    print(
        f"  offset from socket: X={off_mm[0]:+.1f} Y={off_mm[1]:+.1f} Z={off_mm[2]:+.1f} mm"
        f"  euler=[{eul_deg[0]:+.1f},{eul_deg[1]:+.1f},{eul_deg[2]:+.1f}] deg\n"
        f"  tip -> opening: XY={xy_dist:.2f} mm  Z={z_disp:+.2f} mm (<0=inside socket)\n"
        f"  origin -> origin: XY={orig_xy:.2f} mm  Z={orig_z:+.2f} mm\n"
        f"  mean kp_dist: {kp_dist_mm:.2f} mm\n"
        f"  ENGAGE: |XY|<5mm, Z<40mm, |yaw|<20 deg, tilt<15 deg  |  "
        f"SUCCESS: Z<{(inner.cfg_task.fixed_asset_cfg.height + inner.cfg_task.success_threshold) * 1000:.1f} mm AND |XY|<3mm"
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
    env_id = "Isaac-Forge-RJ45Insert-Kuka-Direct-v0" if args_cli.use_kuka else "Isaac-Forge-RJ45Insert-Direct-v0"

    env_cfg = parse_env_cfg(env_id, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True)

    if args_cli.freeze_socket:
        env_cfg.task.fixed_asset_init_pos_noise    = [0.0, 0.0, 0.0]
        env_cfg.task.fixed_asset_init_orn_range_deg = 0.0

    # Park arm well out of the way — plug will be moved directly.
    env_cfg.task.hand_init_pos       = [0.0, 0.0, 0.5]
    env_cfg.task.hand_init_pos_noise = [0.0, 0.0, 0.0]
    env_cfg.task.hand_init_orn_noise = [0.0, 0.0, 0.0]
    env_cfg.task.held_asset_pos_noise = [0.0, 0.0, 0.0]

    env   = gym.make(env_id, cfg=env_cfg)
    inner = env.unwrapped

    _init_teleport_offsets(env_cfg.task)
    global _pos_z_offset
    _pos_z_offset = _Z_OFFSET_ENGAGE

    env.reset()
    _init_keyboard()

    print(
        f"\n[ENV] {env_id}\n"
        "[CONTROLS]  Arrow=XY  Q/E=Z  I/K=pitch  J/L=yaw  U/O=roll  Shift=5x\n"
        "[TELEPORT]  C=contact-init  R=origin-coincide  F=engage  G=success\n"
        "[MARKERS]   BLUE=tip  ORANGE=opening  GREEN=kp_held  YELLOW=kp_socket  WHITE=rear-edge  CYAN=bottom-patch\n"
        "[AXES]      RED=+X  GREEN=+Y  BLUE=+Z  (dotted lines on both male & female frames)\n"
        f"[TARGET]  socket_target_y={env_cfg.task.socket_target_y_local:+.4f}  "
        f"socket_target_z={env_cfg.task.socket_target_z_local:+.4f}  (from config)\n"
        f"[TELEPORT Z]  R(origins)={_Z_OFFSET_ORIGINS*1000:.1f}mm  "
        f"F(engage)={_Z_OFFSET_ENGAGE*1000:.1f}mm  G(success)={_Z_OFFSET_SUCCESS*1000:.1f}mm\n"
        f"[REAR EDGE] female local frame: x={tuple(env_cfg.task.female_rear_edge_x_range_local)}  "
        f"y={env_cfg.task.female_rear_edge_y_local:+.4f}  z={env_cfg.task.female_rear_edge_z_local:+.4f}\n"
        f"[BOTTOM PATCH] male local frame: x={tuple(env_cfg.task.male_bottom_patch_x_range_local)}  "
        f"y={tuple(env_cfg.task.male_bottom_patch_y_range_local)}  z={env_cfg.task.male_bottom_patch_z_local:+.4f}\n"
        f"[CONTACT RPY] roll={tuple(env_cfg.task.contact_init_roll_range_deg)}  "
        f"pitch={tuple(env_cfg.task.contact_init_pitch_range_deg)}  "
        f"yaw={tuple(env_cfg.task.contact_init_yaw_range_deg)}\n"
        "  GREEN+YELLOW overlap -> kp_dist=0 -> plug at full insertion target\n"
    )

    # Freeze robot arm every step.
    _frozen_joint_pos = inner._robot.data.default_joint_pos.clone()
    _frozen_joint_pos[:, :7] = torch.tensor(
        inner.cfg.ctrl.reset_joints, device=inner.device
    ).unsqueeze(0).expand(inner.num_envs, -1)
    _frozen_joint_vel = torch.zeros_like(_frozen_joint_pos)

    step = 0
    _c_last = _r_last = _f_last = _g_last = False

    while simulation_app.is_running():
        with torch.inference_mode():
            c_now = _key(Ki.C)
            r_now = _key(Ki.R)
            f_now = _key(Ki.F)
            g_now = _key(Ki.G)

            # Leading-edge teleports.
            if c_now and not _c_last:
                _teleport_contact(inner)
            if r_now and not _r_last:
                _teleport(_Z_OFFSET_ORIGINS, "FULL-INSERT (calibrated)", _XY_OFFSET_ORIGINS)
            if f_now and not _f_last:
                _teleport(_Z_OFFSET_ENGAGE,  "ENGAGE (tip 30 mm above cavity)", _XY_OFFSET_ORIGINS)
            if g_now and not _g_last:
                _teleport(_Z_OFFSET_SUCCESS, "SUCCESS threshold depth", _XY_OFFSET_ORIGINS)
            _c_last, _r_last, _f_last, _g_last = c_now, r_now, f_now, g_now

            actions = torch.zeros(env.action_space.shape, device=inner.device)
            _, _, done, trunc, _ = env.step(actions)
            step += 1

            # Keep robot frozen.
            inner._robot.write_joint_state_to_sim(_frozen_joint_pos, _frozen_joint_vel)
            inner._robot.set_joint_position_target(_frozen_joint_pos)
            inner._robot.set_joint_effort_target(_frozen_joint_vel)

            _poll_keys_and_move_plug(inner)
            _draw_markers(inner)

            engaged   = inner._get_curr_successes(success_threshold=inner.cfg_task.engage_threshold)
            successes = inner._get_curr_successes(success_threshold=inner.cfg_task.success_threshold)

            _print_status(step, successes, engaged)
            if step % 10 == 0:
                _print_geometry(inner, step, successes, engaged)

            if torch.any(done | trunc):
                env.reset()
                step = 0
                print("[INFO] Episode reset.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
