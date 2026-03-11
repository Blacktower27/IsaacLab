"""Visualize and interactively test the ForgeBoxLidInsert success check — Kuka variant.

FORGE action convention (forge_env._apply_action):
  actions[:, 0:3] = normalised pos offset from box-action-frame
                    physical target = box_action_frame + actions[0:3] * pos_action_bounds
  actions[:, 4]   = pitch target in bolt frame (rad / rot_action_bounds[1])
  actions[:, 5]   = yaw target in bolt frame, normalised so [-1,1] → [-180°, 90°]
  actions[:, 3,6] = roll (forced 0 in env) / gripper (unused for Kuka)

The script maintains a *virtual target* EE pose that is updated by keyboard input
and converted to FORGE actions each frame.

Keyboard controls (click the viewport once to give it focus):
  Arrow Up / Down    — move EE +Y / -Y  (forward / back)
  Arrow Left / Right — move EE -X / +X  (left / right)
  Q / E              — move EE +Z / -Z  (up / down)
  I / K              — pitch EE +/-
  J / L              — yaw   EE +/-
  Hold Shift         — 5× speed
  R                  — reset arm to default joint pose

Coloured sphere markers (updated every frame):
  BLUE   — left  clip (lid) + left  hole (box)  [matched pair]
  ORANGE — right clip (lid) + right hole (box)  [matched pair]
  GREEN  — per-episode keypoints on lid (kp_lid_local)
  YELLOW — per-episode keypoints on box (kp_box_local)

Usage
-----
./isaaclab.sh -p scripts/environments/visualize_box_lid_success_kuka.py --num_envs 1 --freeze_box

Flags
-----
--freeze_box   Disable box randomisation for a clean axis-aligned view.
--num_envs N   Number of parallel environments (default: 1).
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Interactive ForgeBoxLidInsert Kuka success visualizer.")
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
from isaaclab.utils.math import quat_apply_inverse, quat_from_euler_xyz, quat_mul

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


# ---------------------------------------------------------------------------
# Snap-fit reference points (from STL analysis, metres).
# ---------------------------------------------------------------------------
# Lid STL: clip tooth outer face Y=-44.4mm, centre X=±25mm, top Z=28.9mm
_LEFT_CLIP_LID_LOCAL  = (-0.025218, -0.0444, 0.0289)
_RIGHT_CLIP_LID_LOCAL = ( 0.024250, -0.0444, 0.0289)

# Box STL: pocket centre Z = midpoint of [0.021, 0.029] = 0.025m
_LEFT_HOLE_BOX_LOCAL  = (-0.025218, -0.0444, 0.0289)
_RIGHT_HOLE_BOX_LOCAL = ( 0.024250, -0.0444, 0.0289)
_LEFT_HOLE_TARGET     = (-0.025218, -0.0444, 0.025)
_RIGHT_HOLE_TARGET    = ( 0.024250, -0.0444, 0.025)


# ---------------------------------------------------------------------------
# Keyboard input (event-subscription approach)
# ---------------------------------------------------------------------------
CART_SPEED = 0.03   # 3 mm per step
ANG_SPEED  = 0.15   # ~0.9 deg per step
SHIFT_MUL  = 3.0

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
    print("[keyboard] subscribed — click the viewport once to give it focus.")


def _key(k) -> bool:
    return _key_states.get(k, False)


# ---------------------------------------------------------------------------
# Virtual target state — persists across frames, updated by keyboard.
#
# FORGE actions are absolute targets relative to the box action frame, NOT
# incremental EE deltas. We maintain a virtual target and convert each frame:
#   pos action  = (vtgt_pos - fixed_pos_action_frame) / pos_action_bounds
#   pitch action = vtgt_pitch / rot_action_bounds[1]
#   yaw action   = 2*(vtgt_yaw - (-pi)) / (pi*1.5) - 1   [maps [-pi, pi/2]]
# ---------------------------------------------------------------------------
_vtgt_pos:          torch.Tensor | None = None  # (num_envs, 3) env-local
_vtgt_pitch_action: float = 0.0   # action[4], range ≈ [-1, 1]
_vtgt_yaw_action:   float = 0.0   # action[5], range  [-1, 1]


def _init_virtual_target(inner) -> None:
    """Initialise virtual target from current EE state (call after env step/reset)."""
    global _vtgt_pos, _vtgt_pitch_action, _vtgt_yaw_action
    _vtgt_pos          = inner.fingertip_midpoint_pos.clone()
    _vtgt_pitch_action = inner.actions[0, 4].item()
    _vtgt_yaw_action   = inner.actions[0, 5].item()


def _compute_actions(inner) -> torch.Tensor:
    """Return a (num_envs, 7) FORGE action tensor from the current key state."""
    global _vtgt_pos, _vtgt_pitch_action, _vtgt_yaw_action

    device   = inner.device
    num_envs = inner.num_envs

    mul = SHIFT_MUL if (_key(Ki.LEFT_SHIFT) or _key(Ki.RIGHT_SHIFT)) else 1.0
    cs  = CART_SPEED * mul
    as_ = ANG_SPEED  * mul

    # ---- Position --------------------------------------------------------
    dp = torch.zeros(3, device=device, dtype=torch.float32)
    if _key(Ki.LEFT):  dp[0] -= cs
    if _key(Ki.RIGHT): dp[0] += cs
    if _key(Ki.UP):    dp[1] += cs
    if _key(Ki.DOWN):  dp[1] -= cs
    if _key(Ki.Q):     dp[2] += cs
    if _key(Ki.E):     dp[2] -= cs

    _vtgt_pos = _vtgt_pos + dp.unsqueeze(0)  # (num_envs, 3)

    fixed_pos_action_frame = inner.fixed_pos_obs_frame + inner.init_fixed_pos_obs_noise  # (N,3)
    pos_action_bounds = torch.tensor(inner.cfg.ctrl.pos_action_bounds, device=device)    # (3,)
    pos_action = (_vtgt_pos - fixed_pos_action_frame) / pos_action_bounds                # (N,3)

    # ---- Rotation --------------------------------------------------------
    # pitch: action[4] * rot_action_bounds[1] = pitch_target (rad, bolt frame).
    # yaw:   forge remaps action[5] → [-180°, 90°] via
    #          yaw = deg2rad(-180) + deg2rad(270) * (action[5]+1)/2
    #        so delta in action space ≈ delta_yaw * 2 / (pi*1.5)
    rot_bounds = inner.cfg.ctrl.rot_action_bounds          # list [rx, ry, rz]
    pitch_delta = as_ / rot_bounds[1]
    yaw_delta   = as_ * 2.0 / (math.pi * 1.5)

    if _key(Ki.I): _vtgt_pitch_action += pitch_delta
    if _key(Ki.K): _vtgt_pitch_action -= pitch_delta
    if _key(Ki.J): _vtgt_yaw_action   -= yaw_delta
    if _key(Ki.L): _vtgt_yaw_action   += yaw_delta

    _vtgt_pitch_action = max(-1.0, min(1.0, _vtgt_pitch_action))
    _vtgt_yaw_action   = max(-1.0, min(1.0, _vtgt_yaw_action))

    # ---- Assemble 7-dim action -------------------------------------------
    action = torch.zeros(7, device=device, dtype=torch.float32)
    action[0:3] = pos_action[0]           # env-0 pos (broadcast below)
    action[4]   = _vtgt_pitch_action
    action[5]   = _vtgt_yaw_action
    # action[3] = roll (env forces to 0),  action[6] = gripper (Kuka: fixed)

    return action.unsqueeze(0).expand(num_envs, -1).clone()


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
                "left": sim_utils.SphereCfg(          # left clip + left hole
                    radius=0.005,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.4, 1.0)),
                ),
                "right": sim_utils.SphereCfg(         # right clip + right hole
                    radius=0.005,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.0)),
                ),
                "kp_held": sim_utils.SphereCfg(
                    radius=0.003,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.9, 0.2)),
                ),
                "kp_target": sim_utils.SphereCfg(
                    radius=0.003,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.8, 0.0)),
                ),
            },
        )
        _markers = VisualizationMarkers(cfg)
    return _markers


def _draw_clip_hole_markers(inner):
    markers  = _get_markers()
    device   = inner.device
    num_envs = inner.num_envs

    ident_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0).expand(num_envs, -1)

    # held_pos/held_quat already computed by _compute_intermediate_values (env-local).
    held_pos   = inner.held_pos                                           # (N, 3) env-local
    held_quat  = inner.held_quat                                          # (N, 4) wxyz
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

    # Group by matched pair: [left clip, left hole] share index 0 (blue),
    #                         [right clip, right hole] share index 1 (orange).
    translations_list   = [lc_env + env_orig, lh_env + env_orig, rc_env + env_orig, rh_env + env_orig]
    marker_indices_list = [
        torch.zeros(2 * num_envs, dtype=torch.int32),   # left  pair → "left"  (blue)
        torch.ones( 2 * num_envs, dtype=torch.int32),   # right pair → "right" (orange)
    ]

    # Per-episode keypoints: kp_held (green=2), kp_target (yellow=3).
    if hasattr(inner, "kp_lid_local"):
        n = inner.kp_lid_local.shape[1]
        for i in range(n):
            _, kp_h = torch_utils.tf_combine(held_quat,  held_pos,  ident_q, inner.kp_lid_local[:, i])
            _, kp_t = torch_utils.tf_combine(fixed_quat, fixed_pos, ident_q, inner.kp_box_local[:, i])
            translations_list.append(kp_h + env_orig)
            translations_list.append(kp_t + env_orig)
            marker_indices_list.append(torch.full((num_envs,), 2, dtype=torch.int32))
            marker_indices_list.append(torch.full((num_envs,), 3, dtype=torch.int32))

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
_recent_window:         deque = deque(maxlen=100)
_recent_engaged_window: deque = deque(maxlen=100)


def _track_success(successes: torch.Tensor, engaged: torch.Tensor):
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
    device   = inner.device
    num_envs = inner.num_envs
    ident_q  = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0).expand(num_envs, -1)

    held_pos   = inner.held_pos                                           # env-local
    held_quat  = inner.held_quat
    fixed_pos  = inner._fixed_asset.data.root_pos_w - inner.scene.env_origins
    fixed_quat = inner._fixed_asset.data.root_quat_w

    lc_loc = torch.tensor(_LEFT_CLIP_LID_LOCAL,  device=device).expand(num_envs, -1).clone()
    rc_loc = torch.tensor(_RIGHT_CLIP_LID_LOCAL, device=device).expand(num_envs, -1).clone()

    _, lc_w = torch_utils.tf_combine(held_quat, held_pos, ident_q, lc_loc)
    _, rc_w = torch_utils.tf_combine(held_quat, held_pos, ident_q, rc_loc)

    lc_box = quat_apply_inverse(fixed_quat, lc_w - fixed_pos)
    rc_box = quat_apply_inverse(fixed_quat, rc_w - fixed_pos)

    lh_tgt = torch.tensor(_LEFT_HOLE_TARGET,  device=device)
    rh_tgt = torch.tensor(_RIGHT_HOLE_TARGET, device=device)

    lc0 = lc_box[0].cpu()
    rc0 = rc_box[0].cpu()
    lh0 = lh_tgt.cpu()
    rh0 = rh_tgt.cpu()

    dl  = lc0 - lh0
    dr  = rc0 - rh0
    dl_norm = dl.norm().item() * 1000
    dr_norm = dr.norm().item() * 1000

    # EE (link_tcp) position for display.
    fingertip_pos = inner.fingertip_midpoint_pos[0].cpu()

    def _fmt_pos(v):
        return f"X={v[0]*1000:+6.1f} Y={v[1]*1000:+6.1f} Z={v[2]*1000:+6.1f} mm"

    def _fmt_dist(d, norm):
        return f"ΔX={d[0]*1000:+5.1f} ΔY={d[1]*1000:+5.1f} ΔZ={d[2]*1000:+5.1f} |d|={norm:5.2f} mm"

    state_str = "SUCCESS" if successes[0].item() else ("ENGAGED" if engaged[0].item() else "FAILURE")

    cumul_success_rate  = (_success_count / _total_count * 100) if _total_count > 0 else 0.0
    cumul_engaged_rate  = (_engaged_count / _total_count * 100) if _total_count > 0 else 0.0
    recent_success_frac = (sum(_recent_window) / len(_recent_window) * 100) if _recent_window else 0.0
    recent_engaged_frac = (sum(_recent_engaged_window) / len(_recent_engaged_window) * 100) if _recent_engaged_window else 0.0

    # Orientation of link_tcp in world frame — verify roll≈π, pitch≈0 for OSC compatibility.
    import math as _math
    from isaaclab.utils.math import euler_xyz_from_quat as _euler_xyz
    tcp_quat = inner.fingertip_midpoint_quat[0:1]
    r, p, y = _euler_xyz(tcp_quat)
    r_deg = _math.degrees(r.item())
    p_deg = _math.degrees(p.item())
    y_deg = _math.degrees(y.item())

    print(
        f"[step {step:5d}] env0={state_str}"
        f"  engaged={_engaged_count}/{_total_count}({cumul_engaged_rate:.1f}%)"
        f"  success={_success_count}/{_total_count}({cumul_success_rate:.1f}%)"
        f"  recent-engaged={recent_engaged_frac:.1f}%  recent-success={recent_success_frac:.1f}%\n"
        f"  EE (link_tcp): {_fmt_pos(fingertip_pos)}\n"
        f"  link_tcp euler (world): roll={r_deg:+.1f}° pitch={p_deg:+.1f}° yaw={y_deg:+.1f}°"
        f"  [OSC needs roll≈±180°, pitch≈0°]\n"
        f"  lid  pos (env-local): {_fmt_pos(inner.held_pos[0].cpu())}\n"
        f"  left  clip pos (box-local): {_fmt_pos(lc0)}  dist-to-hole: {_fmt_dist(dl, dl_norm)}\n"
        f"  right clip pos (box-local): {_fmt_pos(rc0)}  dist-to-hole: {_fmt_dist(dr, dr_norm)}\n"
        f"  (hole targets — left: X=-25.2 Y=-44.4 Z=+25.0 mm | right: X=+24.3 Y=-44.4 Z=+25.0 mm)"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    env_cfg = parse_env_cfg(
        "Isaac-Forge-BoxLidInsert-Kuka-Direct-v0",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=True,
    )

    if args_cli.freeze_box:
        env_cfg.task.fixed_asset_init_pos_noise = [0.0, 0.0, 0.0]
        env_cfg.task.fixed_asset_init_orn_range_deg = 0.0

    # Start arm above box with a small noise so OSC has a non-degenerate state.
    env_cfg.task.hand_init_pos       = [0.0, 0.0, 0.15]
    env_cfg.task.hand_init_pos_noise = [0.0, 0.0, 0.0]
    env_cfg.task.hand_init_orn_noise = [0.0, 0.0, 0.0]

    env   = gym.make("Isaac-Forge-BoxLidInsert-Kuka-Direct-v0", cfg=env_cfg)
    inner = env.unwrapped

    env.reset()
    _init_keyboard()

    print(
        "\n[CONTROLS]  Arrow=XY  Q/E=Z  I/K=pitch  J/L=yaw  "
        "Shift=3×speed  R=reset arm\n"
        "[INFO] Virtual-target control — keys move a target pose; FORGE OSC tracks it.\n"
        "[INFO] Roll is fixed by the controller; U/O have no effect.\n"
    )

    # Pre-build reset joint state for the R key.
    _reset_joint_pos = inner._robot.data.default_joint_pos.clone()
    _reset_joint_pos[:, :7] = torch.tensor(
        inner.cfg.ctrl.reset_joints, device=inner.device
    ).unsqueeze(0).expand(inner.num_envs, -1)
    _reset_joint_vel = torch.zeros_like(_reset_joint_pos)

    # Initialise virtual target from the post-reset EE pose.
    _init_virtual_target(inner)

    step = 0
    _r_pressed_last_frame = False

    while simulation_app.is_running():
        with torch.inference_mode():

            r_now = _key(Ki.R)

            if r_now:
                inner._robot.write_joint_state_to_sim(_reset_joint_pos, _reset_joint_vel)
                inner._robot.set_joint_position_target(_reset_joint_pos)

            actions = _compute_actions(inner)

            _, _, done, trunc, _ = env.step(actions)
            step += 1

            # Reinit virtual target AFTER the step so buffers are fresh.
            if r_now and not _r_pressed_last_frame:
                # Leading edge of R — reinit once after the teleport step.
                _init_virtual_target(inner)
            _r_pressed_last_frame = r_now

            _draw_clip_hole_markers(inner)

            engaged   = inner._get_curr_successes(success_threshold=0.9)
            successes = inner._get_curr_successes(success_threshold=inner.cfg_task.success_threshold)
            _track_success(successes, engaged)

            if step % 10 == 0:
                _print_success_info(inner, step, successes, engaged)

            if torch.any(done | trunc):
                env.reset()
                _init_virtual_target(inner)
                step = 0
                _reset_rate_counters()
                print("[INFO] Episode reset — rate counters cleared.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
