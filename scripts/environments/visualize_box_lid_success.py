"""Visualize and interactively test the ForgeBoxLidInsert success check.

Keyboard controls (click the viewport once to give it focus):
  Arrow Up / Down   — move lid +Y / -Y  (forward / back)
  Arrow Left / Right— move lid -X / +X  (left / right)
  Q / E             — move lid +Z / -Z  (up / down)
  I / K             — pitch lid +/-
  J / L             — yaw   lid +/-
  U / O             — roll  lid +/-
  Hold Shift        — 5× speed
  R                 — reset lid to success position (offsets → 0)

Coloured sphere markers (updated every frame):
  BLUE  — clip tooth positions (lid frame → world)
  RED   — hole target positions (box frame → world)

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
from isaaclab.utils.math import quat_apply_inverse, quat_from_euler_xyz, quat_mul

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
MOVE_STEP = 0.0002   # metres per frame (×5 when Shift held)
ROT_STEP  = 0.003    # radians per frame
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
# Interactive lid pose state (relative to the box, expressed in box local).
# Reset (R key) brings these back to zero = exact success position.
# ---------------------------------------------------------------------------
_pos_offset  = [0.0, 0.0, 0.0]   # XYZ offset in world frame, metres
_euler_offset = [0.0, 0.0, 0.0]  # [roll, pitch, yaw] offset, radians


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

    # Reset to success position
    if _key(Ki.R):
        _pos_offset   = [0.0, 0.0, 0.0]
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
            },
        )
        _markers = VisualizationMarkers(cfg)
    return _markers


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

    translations = torch.cat(
        [lc_env + env_orig, rc_env + env_orig, lh_env + env_orig, rh_env + env_orig], dim=0
    )
    identity_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(4 * num_envs, -1)
    marker_indices = torch.cat([
        torch.zeros(2 * num_envs, dtype=torch.int32),
        torch.ones( 2 * num_envs, dtype=torch.int32),
    ])

    markers.visualize(translations=translations, orientations=identity_q, marker_indices=marker_indices)


# ---------------------------------------------------------------------------
# Success / failure rate tracking
# ---------------------------------------------------------------------------
_success_count  = 0   # cumulative steps where ANY env succeeded
_total_count    = 0   # cumulative steps checked
_recent_window: deque = deque(maxlen=100)  # rolling 100-step success fractions


def _track_success(successes: torch.Tensor):
    """Call every step to update success-rate counters."""
    global _success_count, _total_count
    frac = successes.float().mean().item()   # fraction of envs in success this step
    _total_count   += 1
    _success_count += int(successes.any().item())
    _recent_window.append(frac)


def _reset_rate_counters():
    global _success_count, _total_count
    _success_count = 0
    _total_count   = 0
    _recent_window.clear()


# ---------------------------------------------------------------------------
# Success info print
# ---------------------------------------------------------------------------

def _print_success_info(inner, step, successes: torch.Tensor):
    """Print clip positions, distances to hole targets, and success rates."""
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

    # Success rate stats.
    cumul_rate   = (_success_count / _total_count * 100) if _total_count > 0 else 0.0
    recent_frac  = (sum(_recent_window) / len(_recent_window) * 100) if _recent_window else 0.0
    success_str  = "SUCCESS" if successes[0].item() else "FAILURE"

    off_mm  = [x * 1000 for x in _pos_offset]
    eul_deg = [math.degrees(x) for x in _euler_offset]

    print(
        f"[step {step:5d}] env0={success_str}"
        f"  cumul={_success_count}/{_total_count}({cumul_rate:.1f}%)"
        f"  recent-100={recent_frac:.1f}%"
        f"  pos=[{off_mm[0]:+.1f},{off_mm[1]:+.1f},{off_mm[2]:+.1f}]mm"
        f"  euler=[{eul_deg[0]:+.1f},{eul_deg[1]:+.1f},{eul_deg[2]:+.1f}]°\n"
        f"  left  clip pos: {_fmt_pos(lc0)}  →  dist-to-hole: {_fmt_dist(dl, dl_norm)}\n"
        f"  right clip pos: {_fmt_pos(rc0)}  →  dist-to-hole: {_fmt_dist(dr, dr_norm)}\n"
        f"  (hole targets — left: X=-25.2 Y=-44.4 Z=+25.0 mm | right: X=+24.3 Y=-44.4 Z=+25.0 mm)"
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

    env_cfg.task.hand_init_pos_noise  = [0.0, 0.0, 0.0]
    env_cfg.task.hand_init_orn_noise  = [0.0, 0.0, 0.0]
    env_cfg.task.held_asset_pos_noise = [0.0, 0.0, 0.0]

    env   = gym.make("Isaac-Forge-BoxLidInsert-Direct-v0", cfg=env_cfg)
    inner = env.unwrapped

    env.reset()
    _init_keyboard()

    print(
        "\n[CONTROLS]  Arrow=XY  Q/E=Z  I/K=pitch  J/L=yaw  U/O=roll  "
        "Shift=5×speed  R=reset\n"
    )

    step = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=inner.device)
            _, _, done, trunc, _ = env.step(actions)
            step += 1

            _poll_keys_and_move_lid(inner)
            _draw_clip_hole_markers(inner)

            # Compute success every step for rate tracking.
            successes = inner._get_curr_successes(success_threshold=inner.cfg_task.success_threshold)
            _track_success(successes)

            # Print detailed info every 10 steps (3× more frequent than before).
            if step % 10 == 0:
                _print_success_info(inner, step, successes)

            if torch.any(done | trunc):
                env.reset()
                step = 0
                _reset_rate_counters()
                print("[INFO] Episode reset — rate counters cleared.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
