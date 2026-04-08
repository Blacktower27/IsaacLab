"""Interactive Franka teleoperation — keyboard controls the real robot via the action space.

The robot moves through the same IK pipeline as training, so joint limits are respected.
Use this to find which yaw angles are reachable and where to place the dead zone.

Keyboard controls
-----------------
  Arrow Up/Down    — action +Y/-Y  (socket local frame)
  Arrow Left/Right — action -X/+X  (socket local frame)
  Q / E            — action +Z/-Z
  J / L            — yaw action -/+  (moves Franka EE yaw, same mapping as training)
  Hold Shift       — 5x speed

Yaw arc visualization
---------------------
  GREEN arc   — current valid yaw range [yaw_min, yaw_min + 270°]
  RED arc     — dead zone (90° gap)
  YELLOW dots — current EE yaw (socket frame)
  [ / ]       — rotate dead zone -5°/+5°  (Shift = 1° fine step)
  P           — print current yaw_min_deg to terminal (copy into forge_env_cfg.py)

Usage
-----
./isaaclab.sh -p scripts/environments/teleop_franka_yaw.py \\
    --task Isaac-Forge-RJ45Insert-Direct-v0 \\
    --freeze_socket
"""

"""Launch Isaac Sim Simulator first."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Franka teleoperation with yaw dead-zone visualization.")
parser.add_argument("--num_envs",      type=int,  default=1)
parser.add_argument("--task",          type=str,  default="Isaac-Forge-RJ45Insert-Direct-v0")
parser.add_argument("--freeze_socket", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import gymnasium as gym
import torch

import carb
import carb.input
import omni.appwindow

import isaacsim.core.utils.torch as torch_utils
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_conjugate, quat_mul

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

# ---------------------------------------------------------------------------
# Dead zone control (adjustable with [ / ] keys)
# ---------------------------------------------------------------------------
_YAW_MIN_DEG: float = 0.0     # start of valid range; dead zone = [yaw_min+270°, yaw_min+360°)

_YAW_ARC_RADIUS  = 0.060
_YAW_ARC_Z_LOW   = 0.010      # bottom ring height
_YAW_ARC_Z_HIGH  = 0.025      # top ring height (second ring for depth)
_YAW_ARC_SAMPLES = 72         # one sphere per 5°
_YAW_IND_SAMPLES = 6


def _in_valid_range(theta_rad: float) -> bool:
    shifted = (theta_rad - math.radians(_YAW_MIN_DEG)) % (2 * math.pi)
    return shifted < math.radians(270.0)


# ---------------------------------------------------------------------------
# Keyboard state
# ---------------------------------------------------------------------------
Ki = carb.input.KeyboardInput
_key_states: dict = {}
_key_sub = None


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
# Action state  (7-dim: [dx, dy, dz, droll, dpitch, dyaw, success])
# ---------------------------------------------------------------------------
_action = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]   # yaw action in [-1, +1]

POS_STEP = 0.005    # action units per frame
YAW_STEP = 0.02    # yaw action units per frame
ARC_STEP_COARSE = 5.0   # degrees per [ / ] press
ARC_STEP_FINE   = 1.0

_bracket_l_last = False
_bracket_r_last = False
_p_last         = False


def _poll_keys(base_env):
    global _action, _YAW_MIN_DEG, _bracket_l_last, _bracket_r_last, _p_last

    shift = _key(Ki.LEFT_SHIFT) or _key(Ki.RIGHT_SHIFT)
    mul   = 5.0 if shift else 1.0
    ps    = POS_STEP * mul
    ys    = YAW_STEP * mul
    arc_s = ARC_STEP_FINE if shift else ARC_STEP_COARSE

    if _key(Ki.UP):    _action[1] = min( 1.0, _action[1] + ps)
    if _key(Ki.DOWN):  _action[1] = max(-1.0, _action[1] - ps)
    if _key(Ki.LEFT):  _action[0] = max(-1.0, _action[0] - ps)
    if _key(Ki.RIGHT): _action[0] = min( 1.0, _action[0] + ps)
    if _key(Ki.Q):     _action[2] = min( 1.0, _action[2] + ps)
    if _key(Ki.E):     _action[2] = max(-1.0, _action[2] - ps)
    if _key(Ki.J):     _action[5] = max(-1.0, _action[5] - ys)
    if _key(Ki.L):     _action[5] = min( 1.0, _action[5] + ys)

    # [ / ] — rotate dead zone (leading-edge)
    bl_now = _key(Ki.LEFT_BRACKET)
    br_now = _key(Ki.RIGHT_BRACKET)
    if bl_now and not _bracket_l_last:
        _YAW_MIN_DEG -= arc_s
    if br_now and not _bracket_r_last:
        _YAW_MIN_DEG += arc_s
    if bl_now and not _bracket_l_last or br_now and not _bracket_r_last:
        dead_start = (_YAW_MIN_DEG + 270.0) % 360.0
        dead_end   = (_YAW_MIN_DEG + 360.0) % 360.0
        print(
            f"[arc]  yaw_min={_YAW_MIN_DEG:+.1f}°  "
            f"valid=[{_YAW_MIN_DEG:+.1f}°, {_YAW_MIN_DEG+270.0:+.1f}°]  "
            f"dead=[{dead_start:+.1f}°, {dead_end:+.1f}°]"
        )
    _bracket_l_last = bl_now
    _bracket_r_last = br_now

    # P — print yaw_min for config
    p_now = _key(Ki.P)
    if p_now and not _p_last:
        ee_quat    = base_env.fingertip_midpoint_quat
        fixed_quat = base_env._fixed_asset.data.root_quat_w
        rel_q      = quat_mul(quat_conjugate(fixed_quat[0:1]), ee_quat[0:1])
        _, _, yaw_t = euler_xyz_from_quat(rel_q)
        curr_yaw   = math.degrees(yaw_t.item())
        in_range   = _in_valid_range(yaw_t.item())
        dead_start = (_YAW_MIN_DEG + 270.0) % 360.0
        print(
            f"\n=== YAW CONFIG SNAPSHOT ===\n"
            f"  franka_yaw_min_deg = {_YAW_MIN_DEG:.1f}    ← copy to ForgeCtrlCfg\n"
            f"  valid range : [{_YAW_MIN_DEG:+.1f}°, {_YAW_MIN_DEG+270.0:+.1f}°]\n"
            f"  dead zone   : [{dead_start:+.1f}°, {_YAW_MIN_DEG+360.0:+.1f}°]\n"
            f"  current EE yaw (socket frame): {curr_yaw:+.1f}°  {'OK' if in_range else '*** IN DEAD ZONE ***'}\n"
            f"=========================\n"
        )
    _p_last = p_now

    return torch.tensor([_action], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Yaw arc markers
# ---------------------------------------------------------------------------
_markers: VisualizationMarkers | None = None


def _get_markers() -> VisualizationMarkers:
    global _markers
    if _markers is None:
        cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/TeleopYawArc",
            markers={
                "arc_ok":   sim_utils.SphereCfg(radius=0.0018,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 1.0, 0.3))),
                "arc_bad":  sim_utils.SphereCfg(radius=0.0018,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.1, 0.1))),
                "curr_yaw": sim_utils.SphereCfg(radius=0.0032,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0))),
            },
        )
        _markers = VisualizationMarkers(cfg)
    return _markers


def _draw_arc(base_env):
    markers  = _get_markers()
    device   = base_env.device
    num_envs = base_env.num_envs

    fixed_pos_w  = base_env._fixed_asset.data.root_pos_w
    fixed_quat   = base_env._fixed_asset.data.root_quat_w
    ee_quat      = base_env.fingertip_midpoint_quat

    ident_q = torch.zeros((num_envs, 4), device=device)
    ident_q[:, 0] = 1.0

    translations   = []
    marker_indices = []

    # Full 360° ring at two heights (low + high) for easy reading.
    for z_level in (_YAW_ARC_Z_LOW, _YAW_ARC_Z_HIGH):
        for i in range(_YAW_ARC_SAMPLES):
            theta    = math.radians(-180.0 + 360.0 * i / _YAW_ARC_SAMPLES)
            in_range = _in_valid_range(theta)
            midx     = 0 if in_range else 1
            local_pt = torch.tensor(
                [_YAW_ARC_RADIUS * math.cos(theta),
                 _YAW_ARC_RADIUS * math.sin(theta),
                 z_level],
                device=device, dtype=torch.float32,
            ).unsqueeze(0).expand(num_envs, -1)
            _, arc_w = torch_utils.tf_combine(fixed_quat, fixed_pos_w, ident_q, local_pt)
            translations.append(arc_w)
            marker_indices.append(torch.full((num_envs,), midx, dtype=torch.int32, device=device))

    # Current EE yaw indicator (yellow).
    rel_q       = quat_mul(quat_conjugate(fixed_quat), ee_quat)
    _, _, rel_y = euler_xyz_from_quat(rel_q)
    cos_y = torch.cos(rel_y)
    sin_y = torch.sin(rel_y)
    for si in range(1, _YAW_IND_SAMPLES + 1):
        r        = _YAW_ARC_RADIUS * si / _YAW_IND_SAMPLES
        local_pt = torch.stack(
            [r * cos_y, r * sin_y,
             torch.full((num_envs,), (_YAW_ARC_Z_LOW + _YAW_ARC_Z_HIGH) / 2, device=device)],
            dim=1,
        )
        _, ind_w = torch_utils.tf_combine(fixed_quat, fixed_pos_w, ident_q, local_pt)
        translations.append(ind_w)
        marker_indices.append(torch.full((num_envs,), 2, dtype=torch.int32, device=device))

    t = torch.cat(translations, dim=0)
    m = torch.cat(marker_indices, dim=0)
    iq = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(len(t), -1)
    markers.visualize(translations=t, orientations=iq, marker_indices=m)


# ---------------------------------------------------------------------------
# Status print
# ---------------------------------------------------------------------------
def _print_status(base_env, step):
    if step % 20 != 0:
        return
    ee_quat    = base_env.fingertip_midpoint_quat
    fixed_quat = base_env._fixed_asset.data.root_quat_w
    rel_q      = quat_mul(quat_conjugate(fixed_quat[0:1]), ee_quat[0:1])
    _, _, yaw_t = euler_xyz_from_quat(rel_q)
    curr_yaw   = math.degrees(yaw_t.item())
    in_range   = _in_valid_range(yaw_t.item())
    yaw_act    = _action[5]
    cmd_yaw    = _YAW_MIN_DEG + 270.0 * (yaw_act + 1.0) / 2.0
    status     = "OK" if in_range else "*** DEAD ZONE ***"
    print(
        f"[step {step:5d}]  "
        f"EE yaw={curr_yaw:+.1f}°  {status}  |  "
        f"yaw_action={yaw_act:+.3f} → cmd={cmd_yaw:+.1f}°  |  "
        f"dead_zone=[{(_YAW_MIN_DEG+270)%360:+.1f}°, {_YAW_MIN_DEG%360:+.1f}°]  "
        f"([ / ] to rotate, P to print config)"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True)

    if args_cli.freeze_socket:
        env_cfg.task.fixed_asset_init_pos_noise     = [0.0, 0.0, 0.0]
        env_cfg.task.fixed_asset_init_orn_range_deg = 0.0

    env   = gym.make(args_cli.task, cfg=env_cfg)
    inner = env.unwrapped

    env.reset()
    _init_keyboard()

    print(
        f"\n[TELEOP]  Task: {args_cli.task}\n"
        "[MOVE]    Arrow=XY  Q/E=Z  J/L=yaw(-/+)\n"
        "[ARC]     [ / ] = rotate dead zone (-5°/+5°)   Shift+[/] = 1° fine\n"
        "          GREEN = reachable   RED = dead zone   YELLOW = current EE yaw\n"
        "[PRINT]   P = print yaw_min_deg snapshot to terminal\n"
        "Note: J/L controls yaw through the REAL Franka IK — joint limits apply.\n"
    )

    step = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = _poll_keys(inner)
            actions = actions.to(inner.device).expand(inner.num_envs, -1)
            env.step(actions)
            _draw_arc(inner)
            _print_status(inner, step)
        step += 1


if __name__ == "__main__":
    main()
    try:
        simulation_app.close()
    except Exception:
        pass
