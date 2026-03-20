"""Visualize and interactively test the RJ45Insert engage/success/keypoint checks.

The male plug is moved directly via keyboard — no robot physics involved.
Use this to verify:
  1. Keypoint positions (GREEN=plug body, YELLOW=socket body) converge when origins align
  2. Engage fires when plug tip is within 5mm XY, <40mm above opening, yaw<20 deg, tilt<15 deg
  3. Success fires when tip is >=7mm inside socket AND xy<3mm

Keyboard controls:
  Arrow Up/Down    — plug +Y/-Y  (world frame)
  Arrow Left/Right — plug -X/+X
  Q / E            — plug +Z/-Z  (up / down)
  I / K            — pitch +/-
  J / L            — yaw   +/-
  U / O            — roll  +/-
  Hold Shift       — 5x speed
  R                — ORIGIN-COINCIDE: place plug origin at socket origin
                     (keypoints should overlap if origins coincide at success)
  F                — ENGAGE position: tip 30 mm above socket opening, aligned
  G                — SUCCESS position: tip at success-threshold depth inside socket

Coloured sphere markers (updated every frame):
  BLUE   -- connector tip  (plug local [0,0,-13.98mm])
  ORANGE -- socket opening centre  (socket local [0,0,+29.22mm])
  GREEN  -- per-episode kp_rj45_local in plug frame  (keypoints_held)
  YELLOW -- per-episode kp_rj45_local in socket frame (keypoints_fixed)
  GREEN/YELLOW overlap -> keypoint_dist = 0 -> origins aligned

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
from collections import deque

import carb
import carb.input
import omni.appwindow
import torch
import gymnasium as gym

import isaacsim.core.utils.torch as torch_utils
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


# ---------------------------------------------------------------------------
# RJ45 geometry constants (metres, post-STL->USD 0.001 scale, -90 Y rotation).
# ---------------------------------------------------------------------------
_TIP_LOCAL      = (0.0, 0.0, -0.01398)   # connector tip in plug USD frame
_OPENING_LOCAL  = (0.0, 0.0,  0.02922)   # socket opening centre in socket USD frame

# Z offset from socket origin to place plug origin at different positions:
#   F (ENGAGE)  = tip 30 mm above socket opening
#     plug_origin_z = socket_opening_z + 0.030 + 0.01398
#                   = socket_origin_z  + 0.02922 + 0.030 + 0.01398 = socket_origin_z + 0.07320
#   G (SUCCESS) = tip at success threshold depth = height * success_threshold = 0.02922 * (-0.24) = -0.007
#     plug_origin_z = socket_opening_z - 0.007 + 0.01398
#                   = socket_origin_z  + 0.02922 - 0.007 + 0.01398 = socket_origin_z + 0.03620
#   R (ORIGINS) = origins coincide: plug_origin_z = socket_origin_z -> offset = 0
_Z_OFFSET_ENGAGE  =  0.07320   # F key: 30mm above socket opening
_Z_OFFSET_SUCCESS =  0.03620   # G key: at success threshold depth
_Z_OFFSET_ORIGINS =  0.00000   # R key: plug origin = socket origin


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
# R/F/G snap these to preset positions.
# ---------------------------------------------------------------------------
_pos_z_offset: float = _Z_OFFSET_ENGAGE   # initial position: engage zone
_pos_xy_offset = [0.0, 0.0]               # [x, y] additional offset
_euler_offset  = [0.0, 0.0, 0.0]          # [roll, pitch, yaw]


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

    socket_pos_w  = inner._fixed_asset.data.root_pos_w.clone()   # (N,3) world
    socket_quat_w = inner._fixed_asset.data.root_quat_w.clone()  # (N,4) wxyz

    # Compose offset: XY in world frame, Z along world Z.
    off = torch.tensor(
        [_pos_xy_offset[0], _pos_xy_offset[1], _pos_z_offset],
        device=device, dtype=torch.float32,
    ).unsqueeze(0)
    plug_pos_w = socket_pos_w + off

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

    translations_list   = [tip_env + env_orig, opening_env + env_orig]
    marker_indices_list = [
        torch.zeros(num_envs, dtype=torch.int32, device=device),   # tip=0 (BLUE)
        torch.ones( num_envs, dtype=torch.int32, device=device),   # opening=1 (ORANGE)
    ]

    # Per-episode body keypoints.
    if hasattr(inner, "kp_rj45_local"):
        n_kp = inner.kp_rj45_local.shape[1]
        for i in range(n_kp):
            _, kp_h = torch_utils.tf_combine(held_quat,  held_pos,  ident_q, inner.kp_rj45_local[:, i])
            _, kp_t = torch_utils.tf_combine(fixed_quat, fixed_pos, ident_q, inner.kp_rj45_local[:, i])
            translations_list.append(kp_h + env_orig)
            translations_list.append(kp_t + env_orig)
            marker_indices_list.append(torch.full((num_envs,), 2, dtype=torch.int32, device=device))  # GREEN
            marker_indices_list.append(torch.full((num_envs,), 3, dtype=torch.int32, device=device))  # YELLOW

    translations   = torch.cat(translations_list,   dim=0)
    marker_indices = torch.cat(marker_indices_list, dim=0)
    identity_q     = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(len(translations), -1)
    markers.visualize(translations=translations, orientations=identity_q, marker_indices=marker_indices)


# ---------------------------------------------------------------------------
# Success / engaged tracking
# ---------------------------------------------------------------------------
_success_count  = 0
_engaged_count  = 0
_total_count    = 0
_recent_success: deque = deque(maxlen=100)
_recent_engaged: deque = deque(maxlen=100)


def _track(successes, engaged):
    global _success_count, _engaged_count, _total_count
    _total_count   += 1
    _success_count += int(successes.any().item())
    _engaged_count += int(engaged.any().item())
    _recent_success.append(successes.float().mean().item())
    _recent_engaged.append(engaged.float().mean().item())


def _reset_counters():
    global _success_count, _engaged_count, _total_count
    _success_count = _engaged_count = _total_count = 0
    _recent_success.clear()
    _recent_engaged.clear()


# ---------------------------------------------------------------------------
# Diagnostic print
# ---------------------------------------------------------------------------
def _print_info(inner, step, successes, engaged):
    device   = inner.device
    num_envs = inner.num_envs

    held_pos   = inner._held_asset.data.root_pos_w  - inner.scene.env_origins
    held_quat  = inner._held_asset.data.root_quat_w
    fixed_pos  = inner._fixed_asset.data.root_pos_w - inner.scene.env_origins
    fixed_quat = inner._fixed_asset.data.root_quat_w

    ident_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0).expand(num_envs, -1)

    # Connector tip (world frame).
    tip_loc = torch.tensor(_TIP_LOCAL, device=device).expand(num_envs, -1).clone()
    _, tip_w = torch_utils.tf_combine(held_quat, held_pos, ident_q, tip_loc)

    # Socket opening (world frame).
    opening_loc = torch.tensor(_OPENING_LOCAL, device=device).expand(num_envs, -1).clone()
    _, opening_w = torch_utils.tf_combine(fixed_quat, fixed_pos, ident_q, opening_loc)

    delta   = (tip_w[0] - opening_w[0]).cpu()
    xy_dist = delta[:2].norm().item() * 1000     # mm
    z_disp  = delta[2].item() * 1000             # mm  (<0 = tip inside socket)

    # Origin-to-origin delta.
    orig_delta = (held_pos[0] - fixed_pos[0]).cpu()
    orig_xy    = orig_delta[:2].norm().item() * 1000
    orig_z     = orig_delta[2].item() * 1000

    # Keypoint distance (mean over all kps, env 0).
    kp_dist_mm = float("nan")
    if hasattr(inner, "kp_rj45_local"):
        n_kp = inner.kp_rj45_local.shape[1]
        kp_dists = []
        for i in range(n_kp):
            _, kp_h = torch_utils.tf_combine(held_quat,  held_pos,  ident_q, inner.kp_rj45_local[:, i])
            _, kp_t = torch_utils.tf_combine(fixed_quat, fixed_pos, ident_q, inner.kp_rj45_local[:, i])
            kp_dists.append((kp_h[0] - kp_t[0]).norm().item())
        kp_dist_mm = sum(kp_dists) / len(kp_dists) * 1000

    state_str = "SUCCESS" if successes[0].item() else ("ENGAGED" if engaged[0].item() else "WAITING")

    cr = (_success_count / _total_count * 100) if _total_count > 0 else 0.0
    er = (_engaged_count / _total_count * 100) if _total_count > 0 else 0.0
    rs = (sum(_recent_success) / len(_recent_success) * 100) if _recent_success else 0.0
    re = (sum(_recent_engaged) / len(_recent_engaged) * 100) if _recent_engaged else 0.0

    off_mm  = [_pos_xy_offset[0]*1000, _pos_xy_offset[1]*1000, _pos_z_offset*1000]
    eul_deg = [math.degrees(x) for x in _euler_offset]

    print(
        f"[step {step:5d}] env0={state_str}"
        f"  engaged={_engaged_count}/{_total_count}({er:.0f}%)"
        f"  success={_success_count}/{_total_count}({cr:.0f}%)"
        f"  recent-engaged={re:.0f}%  recent-success={rs:.0f}%\n"
        f"  offset from socket: X={off_mm[0]:+.1f} Y={off_mm[1]:+.1f} Z={off_mm[2]:+.1f} mm"
        f"  euler=[{eul_deg[0]:+.1f},{eul_deg[1]:+.1f},{eul_deg[2]:+.1f}] deg\n"
        f"  tip -> opening: XY={xy_dist:.2f} mm  Z={z_disp:+.2f} mm (<0=inside socket)\n"
        f"  origin -> origin: XY={orig_xy:.2f} mm  Z={orig_z:+.2f} mm\n"
        f"  mean kp_dist: {kp_dist_mm:.2f} mm  (0=origins aligned)\n"
        f"  ENGAGE fires when: |XY|<5mm, Z<40mm, |yaw|<20 deg, tilt<15 deg\n"
        f"  SUCCESS fires when: Z<{inner.cfg_task.fixed_asset_cfg.height * inner.cfg_task.success_threshold * 1000:.1f} mm AND |XY|<3mm"
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

    env.reset()
    _init_keyboard()

    print(
        f"\n[ENV] {env_id}\n"
        "[CONTROLS]  Arrow=XY  Q/E=Z  I/K=pitch  J/L=yaw  U/O=roll  Shift=5x\n"
        "[TELEPORT]  R=origin-coincide  F=engage  G=success\n"
        "[MARKERS]   BLUE=tip  ORANGE=opening  GREEN=kp_held  YELLOW=kp_socket\n"
        "  GREEN+YELLOW overlap -> kp_dist=0 -> origins aligned (R position)\n"
    )

    # Freeze robot arm every step.
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

            # Leading-edge teleports.
            if r_now and not _r_last:
                _teleport(_Z_OFFSET_ORIGINS, "ORIGIN-COINCIDE (kp should overlap)")
            if f_now and not _f_last:
                _teleport(_Z_OFFSET_ENGAGE,  "ENGAGE (tip 30 mm above opening)")
            if g_now and not _g_last:
                _teleport(_Z_OFFSET_SUCCESS, "SUCCESS threshold depth")
            _r_last, _f_last, _g_last = r_now, f_now, g_now

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
            _track(successes, engaged)

            if step % 10 == 0:
                _print_info(inner, step, successes, engaged)

            if torch.any(done | trunc):
                env.reset()
                step = 0
                _reset_counters()
                print("[INFO] Episode reset.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
