"""Visualize and interactively test the BNCSmallInsert engage/success/keypoint checks.

The male plug is moved directly via keyboard — no robot physics involved.
Use this to verify:
  1. Keypoint positions (standard Z-axis keypoints) move toward target at socket opening
  2. Engage: |XY|<4 mm, Z per cfg, π-sym yaw (~10°), tilt<15°
  3. Success: tip vs opening Z per cfg AND |XY|<3 mm (no yaw)

BNC Small geometry (metres, post-STL->USD 0.001 scale, Z-up no rotation):
  Female (socket): USD origin 35mm above base, opening at USD_origin + Z = +0.025 m
  Male (plug):     USD origin 36.235mm BELOW tip, tip at USD_origin + Z = +0.036235 m

Keyboard controls:
  Arrow Up/Down    -- plug +Y/-Y  (world frame)
  Arrow Left/Right -- plug -X/+X
  Q / E            -- plug +Z/-Z  (up / down)
  I / K            -- pitch +/-
  J / L            -- yaw   +/-
  U / O            -- roll  +/-
  Hold Shift       -- 5x speed
  F                -- ENGAGE position: tip 30 mm above socket opening, aligned
  G                -- SUCCESS position: tip at effective success z_disp (cfg thresholds)
  R                -- ORIGIN-COINCIDE: place plug origin at socket origin

Coloured sphere markers (updated every frame):
  BLUE   -- connector tip  (plug local [0,0,+0.036235])
  ORANGE -- socket opening centre  (socket local [0,0,+0.025])
Thin box markers (contact-init sampling ranges in each body frame, from task cfg):
  YELLOW  -- bnc_contact_init_female_*  (Z fixed, XY ranges) in socket USD frame
  MAGENTA -- bnc_contact_init_male_*  (Z at plug tip +Z, XY on mating face) in plug USD frame

Usage
-----
./isaaclab.sh -p scripts/environments/visualize_bnc_small_insert.py --num_envs 1 --freeze_socket

Flags
-----
--freeze_socket   Disable socket randomisation for a clean axis-aligned view.
--num_envs N      Number of parallel environments (default: 1).
--use_kuka        Use the Kuka variant (default: Franka).
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Interactive BNCSmallInsert visualizer (no robot).")
parser.add_argument("--num_envs",      type=int, default=1)
parser.add_argument("--freeze_socket", action="store_true", default=False)
parser.add_argument("--use_kuka",      action="store_true", default=False)
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
from isaaclab_tasks.direct.factory import factory_utils


# ---------------------------------------------------------------------------
# BNC Small geometry constants (metres, post-STL->USD 0.001 scale, Z-up).
# ---------------------------------------------------------------------------
_TIP_LOCAL     = (0.0, 0.0,  0.036235)  # connector tip in plug USD frame
_OPENING_LOCAL = (0.0, 0.0,  0.025)     # socket opening centre in socket USD frame

# Z offset from socket origin to place plug origin at different positions:
#   F (ENGAGE)  = tip 30 mm above socket opening
#     plug_origin_z = socket_opening_z + 0.030 - 0.036235
#                   = socket_origin_z  + 0.025 + 0.030 - 0.036235 = socket_origin_z + 0.018765
#   G (SUCCESS) = tip z_disp = min(height*success_threshold, -bnc_success_min_depth_m) vs opening (see cfg);
#     plug_origin Z offset from socket origin = z_disp - (tip_local_z - opening_local_z)
#   R (ORIGINS) = origins coincide: plug_origin_z = socket_origin_z -> offset = 0
_Z_OFFSET_ENGAGE  =  0.018765   # F key: 30mm above socket opening
_Z_OFFSET_ORIGINS =  0.00000    # R key: plug origin = socket origin


def _bnc_tip_minus_opening_z_m(cfg_task) -> float:
    """Aligned bodies: tip Z − opening Z = plug_origin_z_offset + (tip_local_z − opening_local_z)."""
    return float(cfg_task.held_asset_cfg.base_height) - float(cfg_task.fixed_asset_cfg.height)


def _bnc_success_z_disp_threshold_m(cfg_task) -> float:
    """Same as FactoryEnv BNC SUCCESS branch: z_disp must be <= this (m; negative = inside opening)."""
    h = float(cfg_task.fixed_asset_cfg.height)
    st = float(cfg_task.success_threshold)
    ht = h * st
    md = float(getattr(cfg_task, "bnc_success_min_depth_m", 0.0))
    if md > 0.0:
        ht = min(ht, -md)
    return ht


def _bnc_plug_origin_z_offset_for_tip_z_disp_m(cfg_task, z_disp_m: float) -> float:
    """Socket-origin Z offset for plug root so tip−opening z_disp equals z_disp_m (aligned Z)."""
    return z_disp_m - _bnc_tip_minus_opening_z_m(cfg_task)


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
# Plug pose state (offset from socket origin in world frame).
# ---------------------------------------------------------------------------
_pos_z_offset: float = _Z_OFFSET_ENGAGE
_pos_xy_offset = [0.0, 0.0]
_euler_offset  = [0.0, 0.0, 0.0]


def _teleport(z_off: float, label: str):
    global _pos_z_offset, _pos_xy_offset, _euler_offset
    _pos_z_offset  = z_off
    _pos_xy_offset = [0.0, 0.0]
    _euler_offset  = [0.0, 0.0, 0.0]
    print(f"[TELEPORT -> {label}]  plug_origin_z = socket_origin_z + {z_off*1000:.1f} mm")


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

    socket_pos_w  = inner._fixed_asset.data.root_pos_w.clone()
    socket_quat_w = inner._fixed_asset.data.root_quat_w.clone()

    off = torch.tensor(
        [_pos_xy_offset[0], _pos_xy_offset[1], _pos_z_offset],
        device=device, dtype=torch.float32,
    ).unsqueeze(0)
    plug_pos_w = socket_pos_w + off

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
_contact_range_markers: VisualizationMarkers | None = None

_SLAB_DZ = 0.00035  # thin slab thickness (m) for range overlays


def _get_contact_range_markers() -> VisualizationMarkers:
    global _contact_range_markers
    if _contact_range_markers is None:
        _contact_range_markers = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/BNCSmallContactInitRanges",
                markers={
                    "female_slab": sim_utils.CuboidCfg(
                        size=(1.0, 1.0, 1.0),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.95, 0.85, 0.15),
                        ),
                    ),
                    "male_slab": sim_utils.CuboidCfg(
                        size=(1.0, 1.0, 1.0),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.85, 0.15, 0.75),
                        ),
                    ),
                },
            )
        )
    return _contact_range_markers


def _draw_contact_init_ranges(inner):
    """Draw axis-aligned XY ranges: female at opening Z, male at plug tip Z (+Z, insertion end)."""
    cfg = inner.cfg_task
    fx = getattr(cfg, "bnc_contact_init_female_x_range", [-0.008, 0.008])
    fy = getattr(cfg, "bnc_contact_init_female_y_range", [-0.008, 0.008])
    fz = getattr(cfg, "bnc_contact_init_female_z_local", 0.025)
    mx = getattr(cfg, "bnc_contact_init_male_x_range", [-0.01035, 0.01035])
    my = getattr(cfg, "bnc_contact_init_male_y_range", [-0.00915, 0.00915])
    mz = getattr(cfg, "bnc_contact_init_male_z_local", 0.036235)

    device = inner.device
    num_envs = inner.num_envs
    m = _get_contact_range_markers()

    held_pos = inner._held_asset.data.root_pos_w - inner.scene.env_origins
    held_quat = inner._held_asset.data.root_quat_w
    fixed_pos = inner._fixed_asset.data.root_pos_w - inner.scene.env_origins
    fixed_quat = inner._fixed_asset.data.root_quat_w
    env_orig = inner.scene.env_origins
    ident_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0).expand(num_envs, -1)

    cxf = 0.5 * (float(fx[0]) + float(fx[1]))
    cyf = 0.5 * (float(fy[0]) + float(fy[1]))
    cmx = 0.5 * (float(mx[0]) + float(mx[1]))
    cmy = 0.5 * (float(my[0]) + float(my[1]))
    local_f = torch.tensor([cxf, cyf, float(fz)], device=device, dtype=torch.float32).unsqueeze(0).expand(
        num_envs, -1
    )
    local_m = torch.tensor([cmx, cmy, float(mz)], device=device, dtype=torch.float32).unsqueeze(0).expand(
        num_envs, -1
    )
    _, pos_f = torch_utils.tf_combine(fixed_quat, fixed_pos, ident_q, local_f)
    _, pos_m = torch_utils.tf_combine(held_quat, held_pos, ident_q, local_m)

    sxf, syf = float(fx[1]) - float(fx[0]), float(fy[1]) - float(fy[0])
    sxm, sym = float(mx[1]) - float(mx[0]), float(my[1]) - float(my[0])
    d = _SLAB_DZ
    scale_f = (
        torch.tensor([sxf, syf, d], device=device, dtype=torch.float32)
        .unsqueeze(0)
        .expand(num_envs, -1)
    )
    scale_m = (
        torch.tensor([sxm, sym, d], device=device, dtype=torch.float32)
        .unsqueeze(0)
        .expand(num_envs, -1)
    )

    trans = torch.cat([pos_f + env_orig, pos_m + env_orig], dim=0)
    quat = torch.cat([fixed_quat, held_quat], dim=0)
    scales = torch.cat([scale_f, scale_m], dim=0)
    marker_indices = torch.cat(
        [
            torch.zeros(num_envs, dtype=torch.int32, device=device),
            torch.ones(num_envs, dtype=torch.int32, device=device),
        ],
        dim=0,
    )
    m.visualize(translations=trans, orientations=quat, scales=scales, marker_indices=marker_indices)


def _get_markers() -> VisualizationMarkers:
    global _markers
    if _markers is None:
        cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/BNCSmallInsertMarkers",
            markers={
                "tip":     sim_utils.SphereCfg(
                    radius=0.004,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.4, 1.0)),
                ),
                "opening": sim_utils.SphereCfg(
                    radius=0.004,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.0)),
                ),
                "kp_plug": sim_utils.SphereCfg(
                    radius=0.003,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.9, 0.1)),
                ),
                "kp_target": sim_utils.SphereCfg(
                    radius=0.003,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.1)),
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

    tip_loc     = torch.tensor(_TIP_LOCAL,     device=device).expand(num_envs, -1).clone()
    opening_loc = torch.tensor(_OPENING_LOCAL, device=device).expand(num_envs, -1).clone()

    _, tip_env     = torch_utils.tf_combine(held_quat,  held_pos,  ident_q, tip_loc)
    _, opening_env = torch_utils.tf_combine(fixed_quat, fixed_pos, ident_q, opening_loc)

    translations   = [tip_env + env_orig, opening_env + env_orig]
    marker_indices = [
        torch.zeros(num_envs, dtype=torch.int32, device=device),   # tip
        torch.ones( num_envs, dtype=torch.int32, device=device),   # opening
    ]

    # --- Keypoint markers (green=plug, red=target) ---
    if hasattr(inner, "kp_bnc_local"):
        held_base_pos, held_base_quat = factory_utils.get_held_base_pose(
            held_pos, held_quat, inner.cfg_task.name,
            inner.cfg_task.fixed_asset_cfg, num_envs, device,
        )
        target_base_pos, target_base_quat = factory_utils.get_target_held_base_pose(
            fixed_pos, fixed_quat, inner.cfg_task.name,
            inner.cfg_task.fixed_asset_cfg, num_envs, device,
        )
        n_kp = inner.kp_bnc_local.shape[1]
        kp_t_loc = getattr(inner, "kp_bnc_fixed_local", inner.kp_bnc_local)
        for i in range(n_kp):
            _, kp_plug_env   = torch_utils.tf_combine(
                held_base_quat,   held_base_pos,   ident_q, inner.kp_bnc_local[:, i]
            )
            _, kp_target_env = torch_utils.tf_combine(
                target_base_quat, target_base_pos, ident_q, kp_t_loc[:, i]
            )
            translations.append(kp_plug_env   + env_orig)
            translations.append(kp_target_env + env_orig)
            marker_indices.append(torch.full((num_envs,), 2, dtype=torch.int32, device=device))  # kp_plug
            marker_indices.append(torch.full((num_envs,), 3, dtype=torch.int32, device=device))  # kp_target

    translations   = torch.cat(translations,   dim=0)
    marker_indices = torch.cat(marker_indices, dim=0)
    identity_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(len(translations), -1)
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

    # Must match FactoryEnv._get_curr_successes (bnc_insert): held_base vs opening at fixed_cfg.height.
    held_base_pos, _held_base_quat = factory_utils.get_held_base_pose(
        held_pos, held_quat, inner.cfg_task.name,
        inner.cfg_task.fixed_asset_cfg, num_envs, device,
    )
    socket_opening_local = torch.zeros((num_envs, 3), device=device)
    socket_opening_local[:, 2] = inner.cfg_task.fixed_asset_cfg.height
    _, socket_opening_world = torch_utils.tf_combine(
        fixed_quat, fixed_pos, ident_q, socket_opening_local
    )
    bdelta  = (held_base_pos[0] - socket_opening_world[0]).cpu()
    xy_dist = bdelta[:2].norm().item() * 1000
    z_disp  = bdelta[2].item() * 1000

    # Marker mesh tip vs held_base (BNC: should coincide).
    tip_loc = torch.tensor(_TIP_LOCAL, device=device).expand(num_envs, -1).clone()
    _, tip_w = torch_utils.tf_combine(held_quat, held_pos, ident_q, tip_loc)
    tip_skew_mm = (tip_w[0] - held_base_pos[0]).norm().item() * 1000

    orig_delta = (held_pos[0] - fixed_pos[0]).cpu()
    orig_xy    = orig_delta[:2].norm().item() * 1000
    orig_z     = orig_delta[2].item() * 1000

    off_mm  = [_pos_xy_offset[0]*1000, _pos_xy_offset[1]*1000, _pos_z_offset*1000]
    eul_deg = [math.degrees(x) for x in _euler_offset]

    _zmax_m = float(getattr(inner.cfg_task, "bnc_engage_z_max_above_opening", 0.040))
    _zmin_m = float(getattr(inner.cfg_task, "bnc_engage_min_depth_m", 0.0))
    _zmax = _zmax_m * 1000
    _engage_z = (
        f"Z: tip ≤{_zmax:.0f}mm above opening AND ≥{_zmin_m*1000:.1f}mm inside"
        if _zmin_m > 0.0
        else f"Z: tip ≤{_zmax:.0f}mm above opening (no min depth)"
    )

    _succ_z_m = _bnc_success_z_disp_threshold_m(inner.cfg_task)
    _succ_z_mm = _succ_z_m * 1000
    _base_succ_mm = inner.cfg_task.fixed_asset_cfg.height * inner.cfg_task.success_threshold * 1000
    _succ_md_mm = float(getattr(inner.cfg_task, "bnc_success_min_depth_m", 0.0)) * 1000
    _succ_line = (
        f"tip Z ≤ {_succ_z_mm:.1f} mm vs opening (stricter of {_base_succ_mm:.1f} mm & −{_succ_md_mm:.1f} mm depth)"
        if _succ_md_mm > 0.0
        else f"tip Z ≤ {_succ_z_mm:.1f} mm vs opening (height×success_threshold)"
    )

    z_disp_m = z_disp / 1000.0
    xy_dist_m = xy_dist / 1000.0
    eng_z_ok = z_disp_m < _zmax_m and (_zmin_m <= 0.0 or z_disp_m <= -_zmin_m)
    _succ_eps = float(getattr(inner.cfg_task, "bnc_success_z_eps_m", 1e-5))
    succ_z_ok = z_disp_m <= _succ_z_m + _succ_eps
    xy_eng_ok = xy_dist_m < 0.004
    xy_succ_ok = xy_dist_m < 0.003

    _, _, plug_yaw = torch_utils.get_euler_xyz(held_quat[0:1])
    _, _, sock_yaw = torch_utils.get_euler_xyz(fixed_quat[0:1])
    yaw_raw = ((plug_yaw - sock_yaw + math.pi) % (2 * math.pi) - math.pi).item()
    yaw_sym = min(abs(yaw_raw), math.pi - abs(yaw_raw))
    yaw_ok = yaw_sym < 0.175
    _ep0 = hasattr(inner, "ep_succeeded") and inner.ep_succeeded[0].item() > 0
    yaw_gate = yaw_ok or _ep0

    plug_z_loc = torch.zeros((1, 3), device=device)
    plug_z_loc[0, 2] = -1.0
    plug_z_w = torch_utils.quat_rotate(held_quat[0:1], plug_z_loc)
    tilt_ok = (-plug_z_w[0, 2]).item() > 0.966

    engage_all = eng_z_ok and xy_eng_ok and yaw_gate and tilt_ok
    success_all = succ_z_ok and xy_succ_ok

    env_eng = bool(engaged[0].item())
    env_suc = bool(successes[0].item())
    mismatch = (engage_all != env_eng) or (success_all != env_suc)
    mismatch_note = (
        f"\n  [!] instant gates != inner._get_curr_successes  env ENGAGE={env_eng} SUCCESS={env_suc}"
        if mismatch
        else ""
    )
    skew_note = (
        f"\n  mesh tip vs held_base skew: {tip_skew_mm:.4f} mm"
        if tip_skew_mm > 0.05
        else ""
    )

    print(
        f"  offset from socket: X={off_mm[0]:+.1f} Y={off_mm[1]:+.1f} Z={off_mm[2]:+.1f} mm"
        f"  euler=[{eul_deg[0]:+.1f},{eul_deg[1]:+.1f},{eul_deg[2]:+.1f}] deg\n"
        f"  held_base->opening (Factory parity): XY={xy_dist:.2f} mm  Z={z_disp:+.2f} mm (<0=inside)\n"
        f"  origin -> origin: XY={orig_xy:.2f} mm  Z={orig_z:+.2f} mm\n"
        f"  ENGAGE: |XY|<4mm, {_engage_z}, π-sym yaw<~10°, tilt<15°  |  "
        f"SUCCESS: {_succ_line} AND |XY|<3mm (no yaw)\n"
        f"  instant (env0): ENGAGE={engage_all} (Z:{eng_z_ok} XY:{xy_eng_ok} yaw:{yaw_gate} tilt:{tilt_ok})  "
        f"SUCCESS={success_all} (Z:{succ_z_ok} XY:{xy_succ_ok})"
        f"{skew_note}{mismatch_note}"
    )

    # --- Keypoint distances ---
    if hasattr(inner, "kp_bnc_local"):
        held_base_pos_kp, held_base_quat_kp = factory_utils.get_held_base_pose(
            held_pos, held_quat, inner.cfg_task.name,
            inner.cfg_task.fixed_asset_cfg, num_envs, device,
        )
        target_base_pos_kp, target_base_quat_kp = factory_utils.get_target_held_base_pose(
            fixed_pos, fixed_quat, inner.cfg_task.name,
            inner.cfg_task.fixed_asset_cfg, num_envs, device,
        )
        n_kp = inner.kp_bnc_local.shape[1]
        kp_t_loc = getattr(inner, "kp_bnc_fixed_local", inner.kp_bnc_local)
        dists_mm = []
        for i in range(n_kp):
            _, kp_h = torch_utils.tf_combine(
                held_base_quat_kp, held_base_pos_kp, ident_q, inner.kp_bnc_local[:, i]
            )
            _, kp_t = torch_utils.tf_combine(
                target_base_quat_kp, target_base_pos_kp, ident_q, kp_t_loc[:, i]
            )
            dists_mm.append((kp_h[0] - kp_t[0]).norm().item() * 1000)
        avg_dist = sum(dists_mm) / len(dists_mm)
        max_dist = max(dists_mm)
        print(f"  keypoint_dist: avg={avg_dist:.2f} mm  max={max_dist:.2f} mm  (n_kp={n_kp})")

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
    env_id = "Isaac-Forge-BNCSmallInsert-Kuka-Direct-v0" if args_cli.use_kuka else "Isaac-Forge-BNCSmallInsert-Direct-v0"

    env_cfg = parse_env_cfg(env_id, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True)

    if args_cli.freeze_socket:
        env_cfg.task.fixed_asset_init_pos_noise     = [0.0, 0.0, 0.0]
        env_cfg.task.fixed_asset_init_orn_range_deg = 0.0

    env_cfg.task.hand_init_pos       = [0.0, 0.0, 0.5]
    env_cfg.task.hand_init_pos_noise = [0.0, 0.0, 0.0]
    env_cfg.task.hand_init_orn_noise = [0.0, 0.0, 0.0]
    env_cfg.task.held_asset_pos_noise = [0.0, 0.0, 0.0]

    env   = gym.make(env_id, cfg=env_cfg)
    inner = env.unwrapped

    env.reset()
    _init_keyboard()

    t = inner.cfg_task
    cinit = ""
    if getattr(t, "bnc_contact_init_female_z_local", None) is not None:
        fxl, fxh = t.bnc_contact_init_female_x_range
        fyl, fyh = t.bnc_contact_init_female_y_range
        mxl, mxh = t.bnc_contact_init_male_x_range
        myl, myh = t.bnc_contact_init_male_y_range
        cinit = (
            f"\n[contact_init cfg — tune in forge_tasks_cfg.py: ForgeBNCSmallInsert]\n"
            f"  female: z_local={t.bnc_contact_init_female_z_local*1000:.3f} mm  "
            f"x=[{fxl*1000:.2f},{fxh*1000:.2f}] mm  y=[{fyl*1000:.2f},{fyh*1000:.2f}] mm\n"
            f"  male:   z_local={t.bnc_contact_init_male_z_local*1000:.3f} mm  "
            f"x=[{mxl*1000:.2f},{mxh*1000:.2f}] mm  y=[{myl*1000:.2f},{myh*1000:.2f}] mm\n"
            f"  slabs: YELLOW=female  MAGENTA=male  (axis-aligned box, Z fixed in each body frame)\n"
        )
    print(
        f"\n[ENV] {env_id}\n"
        "[CONTROLS]  Arrow=XY  Q/E=Z  I/K=pitch  J/L=yaw  U/O=roll  Shift=5x\n"
        "[TELEPORT]  R=origin-coincide  F=engage  G=success\n"
        "[MARKERS]   BLUE=tip  ORANGE=opening\n"
        f"  TIP_LOCAL={[x*1000 for x in _TIP_LOCAL]} mm  OPENING_LOCAL={[x*1000 for x in _OPENING_LOCAL]} mm"
        f"{cinit}"
    )

    _frozen_joint_pos = inner._robot.data.default_joint_pos.clone()
    _frozen_joint_pos[:, :7] = torch.tensor(
        inner.cfg.ctrl.reset_joints, device=inner.device
    ).unsqueeze(0).expand(inner.num_envs, -1)
    _frozen_joint_vel = torch.zeros_like(_frozen_joint_pos)

    step = 0
    _r_last = _f_last = _g_last = False

    while simulation_app.is_running():
        with torch.inference_mode():
            r_now = _key(Ki.R)
            f_now = _key(Ki.F)
            g_now = _key(Ki.G)

            if r_now and not _r_last:
                _teleport(_Z_OFFSET_ORIGINS, "ORIGIN-COINCIDE")
            if f_now and not _f_last:
                _teleport(_Z_OFFSET_ENGAGE,  "ENGAGE (tip 30mm above opening)")
            if g_now and not _g_last:
                zt = _bnc_success_z_disp_threshold_m(inner.cfg_task)
                off = _bnc_plug_origin_z_offset_for_tip_z_disp_m(inner.cfg_task, zt)
                _teleport(off, f"SUCCESS (tip−opening z_disp={zt*1000:.1f} mm)")
            _r_last, _f_last, _g_last = r_now, f_now, g_now

            actions = torch.zeros(env.action_space.shape, device=inner.device)
            _, _, done, trunc, _ = env.step(actions)
            step += 1

            inner._robot.write_joint_state_to_sim(_frozen_joint_pos, _frozen_joint_vel)
            inner._robot.set_joint_position_target(_frozen_joint_pos)
            inner._robot.set_joint_effort_target(_frozen_joint_vel)

            _poll_keys_and_move_plug(inner)
            _draw_markers(inner)
            _draw_contact_init_ranges(inner)

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
