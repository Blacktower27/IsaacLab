"""Visualize and interactively test the BNCSmallInsert engage/success/keypoint checks.

The male plug is moved directly via keyboard — no robot physics involved.
Use this to verify:
  1. Keypoint positions (standard Z-axis keypoints) move toward target at socket opening
  2. Engage fires when plug tip is within 4mm XY, <40mm above opening, yaw<20 deg, tilt<15 deg
  3. Success fires when tip is >=7.5mm inside socket AND xy<3mm

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
  G                -- SUCCESS position: tip at success-threshold depth inside socket
  R                -- ORIGIN-COINCIDE: place plug origin at socket origin

Coloured sphere markers (updated every frame):
  BLUE   -- connector tip  (plug local [0,0,+0.036235])
  ORANGE -- socket opening centre  (socket local [0,0,+0.025])

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
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

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
#   G (SUCCESS) = tip at success threshold depth = height * success_threshold = 0.025 * (-0.3) = -0.0075
#     plug_origin_z = socket_opening_z - 0.0075 - 0.036235
#                   = socket_origin_z  + 0.025 - 0.0075 - 0.036235 = socket_origin_z - 0.018735
#   R (ORIGINS) = origins coincide: plug_origin_z = socket_origin_z -> offset = 0
_Z_OFFSET_ENGAGE  =  0.018765   # F key: 30mm above socket opening
_Z_OFFSET_SUCCESS = -0.018735   # G key: at success threshold depth
_Z_OFFSET_ORIGINS =  0.00000    # R key: plug origin = socket origin


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
        for i in range(n_kp):
            _, kp_plug_env   = torch_utils.tf_combine(
                held_base_quat,   held_base_pos,   ident_q, inner.kp_bnc_local[:, i]
            )
            _, kp_target_env = torch_utils.tf_combine(
                target_base_quat, target_base_pos, ident_q, inner.kp_bnc_local[:, i]
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

    tip_loc     = torch.tensor(_TIP_LOCAL,     device=device).expand(num_envs, -1).clone()
    opening_loc = torch.tensor(_OPENING_LOCAL, device=device).expand(num_envs, -1).clone()
    _, tip_w     = torch_utils.tf_combine(held_quat,  held_pos,  ident_q, tip_loc)
    _, opening_w = torch_utils.tf_combine(fixed_quat, fixed_pos, ident_q, opening_loc)

    delta   = (tip_w[0] - opening_w[0]).cpu()
    xy_dist = delta[:2].norm().item() * 1000
    z_disp  = delta[2].item() * 1000

    orig_delta = (held_pos[0] - fixed_pos[0]).cpu()
    orig_xy    = orig_delta[:2].norm().item() * 1000
    orig_z     = orig_delta[2].item() * 1000

    off_mm  = [_pos_xy_offset[0]*1000, _pos_xy_offset[1]*1000, _pos_z_offset*1000]
    eul_deg = [math.degrees(x) for x in _euler_offset]

    print(
        f"  offset from socket: X={off_mm[0]:+.1f} Y={off_mm[1]:+.1f} Z={off_mm[2]:+.1f} mm"
        f"  euler=[{eul_deg[0]:+.1f},{eul_deg[1]:+.1f},{eul_deg[2]:+.1f}] deg\n"
        f"  tip -> opening: XY={xy_dist:.2f} mm  Z={z_disp:+.2f} mm (<0=inside socket)\n"
        f"  origin -> origin: XY={orig_xy:.2f} mm  Z={orig_z:+.2f} mm\n"
        f"  ENGAGE: |XY|<4mm, Z<40mm, |yaw|<20 deg, tilt<15 deg  |  "
        f"SUCCESS: Z<{inner.cfg_task.fixed_asset_cfg.height * inner.cfg_task.success_threshold * 1000:.1f} mm AND |XY|<3mm"
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

    print(
        f"\n[ENV] {env_id}\n"
        "[CONTROLS]  Arrow=XY  Q/E=Z  I/K=pitch  J/L=yaw  U/O=roll  Shift=5x\n"
        "[TELEPORT]  R=origin-coincide  F=engage  G=success\n"
        "[MARKERS]   BLUE=tip  ORANGE=opening\n"
        f"  TIP_LOCAL={[x*1000 for x in _TIP_LOCAL]} mm  OPENING_LOCAL={[x*1000 for x in _OPENING_LOCAL]} mm\n"
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
                _teleport(_Z_OFFSET_SUCCESS, "SUCCESS threshold depth")
            _r_last, _f_last, _g_last = r_now, f_now, g_now

            actions = torch.zeros(env.action_space.shape, device=inner.device)
            _, _, done, trunc, _ = env.step(actions)
            step += 1

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
