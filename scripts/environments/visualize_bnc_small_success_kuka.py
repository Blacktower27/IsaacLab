"""Visualize and interactively test the ForgeBNCSmallInsert success check — Kuka variant.

FORGE action convention (forge_env._apply_action):
  actions[:, 0:3] = normalised pos offset from bolt-action-frame
                    physical target = action_frame + actions[0:3] * pos_action_bounds
  actions[:, 4]   = pitch target in bolt frame (rad / rot_action_bounds[1])
  actions[:, 5]   = yaw target in bolt frame, normalised so [-1,1] -> [-180°, 90°]
  actions[:, 3,6] = roll (forced 0 in env) / gripper (unused for Kuka)

The script maintains a *virtual target* EE pose that is updated by keyboard input
and converted to FORGE actions each frame.

Keyboard controls (click the viewport once to give it focus):
  Arrow Up / Down    -- move EE +Y / -Y  (forward / back)
  Arrow Left / Right -- move EE -X / +X  (left / right)
  Q / E              -- move EE +Z / -Z  (up / down)
  I / K              -- pitch EE +/-
  J / L              -- yaw   EE +/-
  Hold Shift         -- 5x speed
  R                  -- reset arm to default joint pose
  F                  -- teleport virtual target to ENGAGED position
                        (plug tip ~30 mm above socket opening, aligned for insertion)
  G                  -- teleport virtual target to SUCCESS center position
                        (connector tip exactly at socket opening level)

Coloured sphere markers (updated every frame):
  BLUE   -- BNC connector tip  (plug-local Z = +0.036235 m)
  ORANGE -- BNC socket opening centre  (socket-local Z = +0.025 m)
  GREEN  -- 5 keypoints on plug tip face  (held frame)
  YELLOW -- 5 keypoints on socket opening  (fixed frame)

Usage
-----
./isaaclab.sh -p scripts/environments/visualize_bnc_small_success_kuka.py --num_envs 1 --freeze_socket

Flags
-----
--freeze_socket   Disable socket randomisation for a clean axis-aligned view.
--num_envs N      Number of parallel environments (default: 1).
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Interactive ForgeBNCSmallInsert Kuka success visualizer.")
parser.add_argument("--num_envs",      type=int, default=1)
parser.add_argument("--freeze_socket", action="store_true", default=False)
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
from isaaclab.utils.math import euler_xyz_from_quat

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


# ---------------------------------------------------------------------------
# BNC Small geometry reference points (metres, post-STL->USD 0.001 scale, Z-up).
# ---------------------------------------------------------------------------
# BNC male (plug): STL Z in [+36.235, +107.201] mm; origin 36.235 mm BELOW tip.
#   Connector tip at plug-local Z = +0.036235 m.
#   Body is cylindrical, outer diameter ~41 mm.
#   5 keypoints on the tip face (centre + 4 cardinal edge points at r=15 mm).
_R_TIP = 0.015   # keypoint ring radius on tip face (m)
_BNC_TIP_KPS = [
    (  0.0,      0.0,    0.036235),  # tip centre
    ( _R_TIP,    0.0,    0.036235),  # +X edge
    (-_R_TIP,    0.0,    0.036235),  # -X edge
    (  0.0,    _R_TIP,   0.036235),  # +Y edge
    (  0.0,   -_R_TIP,   0.036235),  # -Y edge
]

# BNC female (socket): STL Z in [-35, +25] mm; origin 35 mm above base.
#   Socket opening at socket-local Z = +0.025 m.
#   Inner bore diameter ~16 mm (BNC spec), use r=8 mm for keypoints.
_R_SOCK = 0.008
_SOCKET_KPS = [
    (  0.0,      0.0,    0.025),  # opening centre
    ( _R_SOCK,   0.0,    0.025),  # +X edge
    (-_R_SOCK,   0.0,    0.025),  # -X edge
    (  0.0,    _R_SOCK,  0.025),  # +Y edge
    (  0.0,   -_R_SOCK,  0.025),  # -Y edge
]

# Geometry constants (metres).
_BNC_BASE_HEIGHT = 0.036235  # BNCSmallMaleCfg.base_height  (tip ABOVE link_bnc origin)
_GRASP_Z_OFF     = 0.060     # joint_bnc z-offset along link_tcp Z
                              #   → link_bnc is 60 mm "forward" of link_tcp (in world -Z)
                              #   when EE points straight down.

# EE Z offset above socket opening to place tip exactly at socket opening:
#   tip_world_z = link_bnc_z + _BNC_BASE_HEIGHT
#   link_bnc_z  = link_tcp_z - _GRASP_Z_OFF    (Rx(pi) maps bnc +Z -> tcp -Z)
#   tip_world_z = link_tcp_z - _GRASP_Z_OFF + _BNC_BASE_HEIGHT
#               = EE_z - 0.060 + 0.036235 = EE_z - 0.023765
# => for tip AT socket opening: EE_z = socket_opening_z + 0.023765
_EE_ABOVE_OPENING = _GRASP_Z_OFF - _BNC_BASE_HEIGHT   # = 0.060 - 0.036235 = 0.023765


# ---------------------------------------------------------------------------
# Keyboard input
# ---------------------------------------------------------------------------
CART_SPEED = 0.002   # 2 mm per step
ANG_SPEED  = 0.15    # ~0.9 deg per step
SHIFT_MUL  = 5.0

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
# Virtual target state — persists across frames, updated by keyboard.
# ---------------------------------------------------------------------------
_vtgt_pos:          torch.Tensor | None = None  # (num_envs, 3) env-local
_vtgt_pitch_action: float = 0.0
_vtgt_yaw_action:   float = 0.0

_TELEPORT_ENGAGED_Z_ABOVE: float = 0.030   # 30 mm above socket opening


def _teleport_vtgt_to(inner, label: str, z_above_success: float = 0.0) -> None:
    """Snap virtual target (OSC EE target) to the position implied by the task success target.

    Chain:
      target_held_base_pos  (= socket opening in world/env frame)
      -> link_bnc_z  = socket_opening_z - _BNC_BASE_HEIGHT  (tip above origin)
      -> link_tcp_z  = link_bnc_z + _GRASP_Z_OFF
      -> EE target_z = link_tcp_z + z_above_success
      Combined: EE_z = socket_opening_z + _EE_ABOVE_OPENING + z_above_success
    """
    from isaaclab_tasks.direct.factory import factory_utils as _futils

    global _vtgt_pos
    device   = inner.device
    num_envs = inner.num_envs

    target_held_base_pos, _ = _futils.get_target_held_base_pose(
        inner.fixed_pos, inner.fixed_quat,
        inner.cfg_task.name, inner.cfg_task.fixed_asset_cfg,
        num_envs, device,
    )

    ee_pos = target_held_base_pos.clone()
    ee_pos[:, 2] += _EE_ABOVE_OPENING + z_above_success
    _vtgt_pos = ee_pos.clone()

    # Zero dead-zone so OSC can converge to mm-level precision.
    inner.dead_zone_thresholds = torch.zeros((inner.num_envs, 6), device=inner.device)

    print(
        f"[TELEPORT -> {label}]"
        f"  EE target (env-local): X={ee_pos[0,0].item()*1000:.1f}"
        f"  Y={ee_pos[0,1].item()*1000:.1f}"
        f"  Z={ee_pos[0,2].item()*1000:.1f} mm"
    )


def _init_virtual_target(inner) -> None:
    global _vtgt_pos, _vtgt_pitch_action, _vtgt_yaw_action
    _vtgt_pos          = inner.fingertip_midpoint_pos.clone()
    _vtgt_pitch_action = inner.actions[0, 4].item()
    _vtgt_yaw_action   = inner.actions[0, 5].item()


def _compute_actions(inner) -> torch.Tensor:
    global _vtgt_pos, _vtgt_pitch_action, _vtgt_yaw_action

    device   = inner.device
    num_envs = inner.num_envs

    mul = SHIFT_MUL if (_key(Ki.LEFT_SHIFT) or _key(Ki.RIGHT_SHIFT)) else 1.0
    cs  = CART_SPEED * mul
    as_ = ANG_SPEED  * mul

    dp = torch.zeros(3, device=device, dtype=torch.float32)
    if _key(Ki.LEFT):  dp[0] -= cs
    if _key(Ki.RIGHT): dp[0] += cs
    if _key(Ki.UP):    dp[1] += cs
    if _key(Ki.DOWN):  dp[1] -= cs
    if _key(Ki.Q):     dp[2] += cs
    if _key(Ki.E):     dp[2] -= cs

    _vtgt_pos = _vtgt_pos + dp.unsqueeze(0)

    pos_bounds = torch.tensor(inner.cfg.ctrl.pos_action_bounds, device=device)
    _vtgt_pos  = torch.clamp(
        _vtgt_pos,
        inner.fingertip_midpoint_pos - pos_bounds,
        inner.fingertip_midpoint_pos + pos_bounds,
    )

    frame      = inner.fixed_pos_obs_frame + inner.init_fixed_pos_obs_noise
    pos_action = (_vtgt_pos - frame) / pos_bounds

    rot_bounds  = inner.cfg.ctrl.rot_action_bounds
    pitch_delta = as_ / rot_bounds[1]
    yaw_delta   = as_ * 2.0 / (math.pi * 1.5)

    if _key(Ki.I): _vtgt_pitch_action += pitch_delta
    if _key(Ki.K): _vtgt_pitch_action -= pitch_delta
    if _key(Ki.J): _vtgt_yaw_action   -= yaw_delta
    if _key(Ki.L): _vtgt_yaw_action   += yaw_delta

    _vtgt_pitch_action = max(-1.0, min(1.0, _vtgt_pitch_action))
    _vtgt_yaw_action   = max(-1.0, min(1.0, _vtgt_yaw_action))

    action = torch.zeros(7, device=device, dtype=torch.float32)
    action[0:3] = pos_action[0]
    action[4]   = _vtgt_pitch_action
    action[5]   = _vtgt_yaw_action

    return action.unsqueeze(0).expand(num_envs, -1).clone()


# ---------------------------------------------------------------------------
# Visualisation markers
# ---------------------------------------------------------------------------
_markers: VisualizationMarkers | None = None


def _get_markers() -> VisualizationMarkers:
    global _markers
    if _markers is None:
        cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/BNCSmallMarkers",
            markers={
                "tip_plug":   sim_utils.SphereCfg(   # connector tip (BLUE)
                    radius=0.004,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.4, 1.0)),
                ),
                "tip_socket": sim_utils.SphereCfg(   # socket opening centre (ORANGE)
                    radius=0.004,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.0)),
                ),
                "kp_held":    sim_utils.SphereCfg(   # plug tip keypoints (GREEN)
                    radius=0.002,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.9, 0.2)),
                ),
                "kp_target":  sim_utils.SphereCfg(   # socket opening keypoints (YELLOW)
                    radius=0.002,
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

    held_pos   = inner.held_pos                                               # (N,3) env-local
    held_quat  = inner.held_quat
    fixed_pos  = inner._fixed_asset.data.root_pos_w - inner.scene.env_origins
    fixed_quat = inner._fixed_asset.data.root_quat_w
    env_orig   = inner.scene.env_origins

    ident_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0).expand(num_envs, -1)

    # Slot 0: plug tip centre (BLUE).  Slot 1: socket opening centre (ORANGE).
    tip_loc  = torch.tensor(_BNC_TIP_KPS[0],  device=device).expand(num_envs, -1).clone()
    sock_loc = torch.tensor(_SOCKET_KPS[0],   device=device).expand(num_envs, -1).clone()
    _, tip_env  = torch_utils.tf_combine(held_quat,  held_pos,  ident_q, tip_loc)
    _, sock_env = torch_utils.tf_combine(fixed_quat, fixed_pos, ident_q, sock_loc)

    translations_list   = [tip_env + env_orig, sock_env + env_orig]
    marker_indices_list = [
        torch.zeros(num_envs, dtype=torch.int32, device=device),   # BLUE
        torch.ones( num_envs, dtype=torch.int32, device=device),   # ORANGE
    ]

    # Slots 2+: ring keypoints on plug tip (GREEN) and socket opening (YELLOW).
    # At success pose (tip at socket opening, XY aligned) GREEN and YELLOW overlap.
    for kp_plug, kp_sock in zip(_BNC_TIP_KPS[1:], _SOCKET_KPS[1:]):
        plug_loc = torch.tensor(kp_plug, device=device).expand(num_envs, -1).clone()
        sock_kp  = torch.tensor(kp_sock, device=device).expand(num_envs, -1).clone()
        _, kp_h = torch_utils.tf_combine(held_quat,  held_pos,  ident_q, plug_loc)
        _, kp_t = torch_utils.tf_combine(fixed_quat, fixed_pos, ident_q, sock_kp)
        translations_list.append(kp_h + env_orig)
        translations_list.append(kp_t + env_orig)
        marker_indices_list.append(torch.full((num_envs,), 2, dtype=torch.int32, device=device))
        marker_indices_list.append(torch.full((num_envs,), 3, dtype=torch.int32, device=device))

    translations   = torch.cat(translations_list,   dim=0)
    marker_indices = torch.cat(marker_indices_list, dim=0)
    identity_q     = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(len(translations), -1)
    markers.visualize(translations=translations, orientations=identity_q, marker_indices=marker_indices)


# ---------------------------------------------------------------------------
# Success / failure rate tracking
# ---------------------------------------------------------------------------
_success_count  = 0
_engaged_count  = 0
_total_count    = 0
_recent_success: deque = deque(maxlen=100)
_recent_engaged: deque = deque(maxlen=100)


def _track_success(successes: torch.Tensor, engaged: torch.Tensor):
    global _success_count, _engaged_count, _total_count
    _total_count   += 1
    _success_count += int(successes.any().item())
    _engaged_count += int(engaged.any().item())
    _recent_success.append(successes.float().mean().item())
    _recent_engaged.append(engaged.float().mean().item())


def _reset_rate_counters():
    global _success_count, _engaged_count, _total_count
    _success_count = _engaged_count = _total_count = 0
    _recent_success.clear()
    _recent_engaged.clear()


# ---------------------------------------------------------------------------
# Diagnostic print
# ---------------------------------------------------------------------------
def _print_info(inner, step, successes: torch.Tensor, engaged: torch.Tensor):
    device   = inner.device
    num_envs = inner.num_envs

    held_pos   = inner.held_pos
    held_quat  = inner.held_quat
    fixed_pos  = inner._fixed_asset.data.root_pos_w - inner.scene.env_origins
    fixed_quat = inner._fixed_asset.data.root_quat_w

    ident_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0).expand(num_envs, -1)

    # Connector tip in env-local frame.
    tip_loc  = torch.tensor(_BNC_TIP_KPS[0], device=device).expand(num_envs, -1).clone()
    sock_loc = torch.tensor(_SOCKET_KPS[0],  device=device).expand(num_envs, -1).clone()
    _, tip_env  = torch_utils.tf_combine(held_quat,  held_pos,  ident_q, tip_loc)
    _, sock_env = torch_utils.tf_combine(fixed_quat, fixed_pos, ident_q, sock_loc)

    tip0  = tip_env[0].cpu()
    sock0 = sock_env[0].cpu()
    delta = tip0 - sock0

    xy_dist_mm = delta[:2].norm().item() * 1000
    z_gap_mm   = delta[2].item() * 1000   # <0 = tip has entered socket

    # Plug yaw vs socket yaw.
    _, _, plug_yaw = euler_xyz_from_quat(held_quat[0:1])
    _, _, sock_yaw = euler_xyz_from_quat(fixed_quat[0:1])

    # EE orientation.
    tcp_quat = inner.fingertip_midpoint_quat[0:1]
    r, p, y  = euler_xyz_from_quat(tcp_quat)
    r_deg, p_deg, y_deg = math.degrees(r.item()), math.degrees(p.item()), math.degrees(y.item())

    fingertip_pos = inner.fingertip_midpoint_pos[0].cpu()

    def _fmt(v):
        return f"X={v[0]*1000:+6.1f} Y={v[1]*1000:+6.1f} Z={v[2]*1000:+6.1f} mm"

    # Yaw diagnostic only (Factory BNC engage/success do not gate on yaw).
    yaw_diff_raw = float(((plug_yaw - sock_yaw + math.pi) % (2 * math.pi) - math.pi).item())
    yaw_diff_sym_deg = math.degrees(min(abs(yaw_diff_raw), math.pi - abs(yaw_diff_raw)))

    state_str = "SUCCESS" if successes[0].item() else ("ENGAGED" if engaged[0].item() else "WAITING")

    cr = (_success_count / _total_count * 100) if _total_count > 0 else 0.0
    er = (_engaged_count / _total_count * 100) if _total_count > 0 else 0.0
    rs = (sum(_recent_success) / len(_recent_success) * 100) if _recent_success else 0.0
    re = (sum(_recent_engaged) / len(_recent_engaged) * 100) if _recent_engaged else 0.0

    print(
        f"[step {step:5d}] env0={state_str}"
        f"  engaged={_engaged_count}/{_total_count}({er:.1f}%)"
        f"  success={_success_count}/{_total_count}({cr:.1f}%)"
        f"  recent-engaged={re:.1f}%  recent-success={rs:.1f}%\n"
        f"  EE (link_tcp): {_fmt(fingertip_pos)}\n"
        f"  link_tcp euler (world): roll={r_deg:+.1f} pitch={p_deg:+.1f} yaw={y_deg:+.1f} deg\n"
        f"  link_bnc pos (env-local): {_fmt(held_pos[0].cpu())}\n"
        f"  connector tip (env-local): {_fmt(tip0)}\n"
        f"  socket opening (env-local): {_fmt(sock0)}\n"
        f"  XY dist tip->socket: {xy_dist_mm:.2f} mm  |  Z gap (tip-socket): {z_gap_mm:+.2f} mm"
        f"  (<0 = inside socket)\n"
        f"  plug yaw vs socket yaw (diagnostic): {math.degrees(yaw_diff_raw):+.1f} deg  "
        f"| yaw_sym={yaw_diff_sym_deg:.1f} deg  (not used for engage/success)\n"
        f"  ENGAGE: |XY|<4mm, Z per cfg (bnc_engage_*), π-sym yaw, tilt<15deg\n"
        f"  SUCCESS: Z_gap vs opening per cfg (success_threshold & bnc_success_min_depth_m)"
        f"  AND |XY|<3mm, no yaw\n"
        f"  EE_ABOVE_OPENING = {_EE_ABOVE_OPENING*1000:.2f} mm"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    env_cfg = parse_env_cfg(
        "Isaac-Forge-BNCSmallInsert-Kuka-Direct-v0",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=True,
    )

    if args_cli.freeze_socket:
        env_cfg.task.fixed_asset_init_pos_noise     = [0.0, 0.0, 0.0]
        env_cfg.task.fixed_asset_init_orn_range_deg = 0.0

    # Start arm above socket with zero noise.
    env_cfg.task.hand_init_pos       = [0.0, 0.0, 0.10]
    env_cfg.task.hand_init_pos_noise = [0.0, 0.0, 0.0]
    env_cfg.task.hand_init_orn_noise = [0.0, 0.0, 0.0]

    env   = gym.make("Isaac-Forge-BNCSmallInsert-Kuka-Direct-v0", cfg=env_cfg)
    inner = env.unwrapped

    env.reset()
    _init_keyboard()

    print(
        "\n[ENV] Isaac-Forge-BNCSmallInsert-Kuka-Direct-v0\n"
        "[CONTROLS]  Arrow=XY  Q/E=Z  I/K=pitch  J/L=yaw  Shift=5x speed  R=reset arm\n"
        "[TELEPORT]  F=ENGAGED (30 mm above opening)  G=SUCCESS (tip at opening level)\n"
        "[MARKERS]   BLUE=tip  ORANGE=opening  GREEN=plug-kps  YELLOW=socket-kps\n"
        f"  BNC tip offset from link_bnc origin: +{_BNC_BASE_HEIGHT*1000:.2f} mm (above origin)\n"
        f"  joint_bnc z-offset in link_tcp:       {_GRASP_Z_OFF*1000:.1f} mm\n"
        f"  EE must be {_EE_ABOVE_OPENING*1000:.2f} mm above socket opening for tip to align\n"
    )

    _reset_joint_pos = inner._robot.data.default_joint_pos.clone()
    _reset_joint_pos[:, :7] = torch.tensor(
        inner.cfg.ctrl.reset_joints, device=inner.device
    ).unsqueeze(0).expand(inner.num_envs, -1)
    _reset_joint_vel = torch.zeros_like(_reset_joint_pos)

    _init_virtual_target(inner)

    step = 0
    _r_last = _f_last = _g_last = False
    _diag_steps = 0

    while simulation_app.is_running():
        with torch.inference_mode():

            r_now = _key(Ki.R)
            f_now = _key(Ki.F)
            g_now = _key(Ki.G)

            if r_now:
                inner._robot.write_joint_state_to_sim(_reset_joint_pos, _reset_joint_vel)
                inner._robot.set_joint_position_target(_reset_joint_pos)

            if f_now and not _f_last:
                _teleport_vtgt_to(inner, "ENGAGED", z_above_success=_TELEPORT_ENGAGED_Z_ABOVE)
                _diag_steps = 100
            if g_now and not _g_last:
                _teleport_vtgt_to(inner, "SUCCESS")
                _diag_steps = 100
            _f_last, _g_last = f_now, g_now

            actions = _compute_actions(inner)
            _, _, done, trunc, _ = env.step(actions)
            step += 1

            if r_now and not _r_last:
                _init_virtual_target(inner)
            _r_last = r_now

            _draw_markers(inner)

            # Diagnostic: after F/G teleport, print tip-to-socket gap each step.
            if _diag_steps > 0:
                from isaaclab_tasks.direct.factory import factory_utils as _futils
                tgt, _ = _futils.get_target_held_base_pose(
                    inner.fixed_pos, inner.fixed_quat,
                    inner.cfg_task.name, inner.cfg_task.fixed_asset_cfg,
                    inner.num_envs, inner.device,
                )
                # held_base = link_bnc + [0, 0, _BNC_BASE_HEIGHT]  (tip above origin)
                tip_z = inner.held_pos[0, 2].item() + _BNC_BASE_HEIGHT
                tgt_z = tgt[0, 2].item()
                ee_z  = inner.fingertip_midpoint_pos[0, 2].item()
                vtgt_z = _vtgt_pos[0, 2].item() if _vtgt_pos is not None else ee_z
                print(
                    f"  [diag {_diag_steps:3d}]"
                    f"  tip Z={tip_z*1000:+7.2f} mm  socket_opening Z={tgt_z*1000:+7.2f} mm"
                    f"  gap={(tip_z - tgt_z)*1000:+6.2f} mm"
                    f"  | EE Z={ee_z*1000:+7.2f}  vtgt Z={vtgt_z*1000:+7.2f}"
                )
                _diag_steps -= 1

            engaged   = inner._get_curr_successes(success_threshold=inner.cfg_task.engage_threshold)
            successes = inner._get_curr_successes(success_threshold=inner.cfg_task.success_threshold)
            _track_success(successes, engaged)

            if step % 10 == 0:
                _print_info(inner, step, successes, engaged)

            if torch.any(done | trunc):
                env.reset()
                _init_virtual_target(inner)
                step = 0
                _reset_rate_counters()
                print("[INFO] Episode reset -- rate counters cleared.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
