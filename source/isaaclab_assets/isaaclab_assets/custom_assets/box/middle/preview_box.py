# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Preview Small_Box and Lid_Yellow USD assets together in Isaac Sim.

Scale note
----------
Both USDs were produced by convert_stl_to_usd.py with scale=0.001 (mm→m).
That scale is already baked into each USD as a root Xform op, so do NOT pass
scale again via UsdFileCfg — doing so would stack the two 0.001 ops and shrink
the objects to 1 e-6 of their intended size.

Instanceable-USD coexistence note
----------------------------------
Both Small_Box.usd and Lid_Yellow.usd were converted with make_instanceable=True
into the SAME directory, so they share a single relative reference:

    ./Props/instanceable_meshes.usd

Batch conversion overwrites that file on every run, leaving it with only the
LAST converted asset's visual mesh (alphabetically Small_Box wins, so Lid's
visual is currently broken).  Physics/collision prims are self-contained in
each USD, so the simulation is still physically correct.

To fix the visual as well, choose one of:
  A) Set  make_instanceable: false  in conversion_config.yaml and re-run.
  B) Convert each asset into its own sub-directory so each has a private Props/.

Usage:
    ./isaaclab.sh -p \\
        source/isaaclab_assets/isaaclab_assets/custom_assets/box/middle/preview_box.py
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Preview box assets in Isaac Sim.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- imports after Isaac Sim starts ---
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_BOX_USD = os.path.join(_HERE, "Small_Box.usd")
_LID_USD = os.path.join(_HERE, "Lid_Yellow.usd")

# Approximate half-heights after mm→m conversion (update if your model differs).
# Small_Box:  ~60 mm tall  → half-height 0.030 m
# Lid_Yellow: ~12 mm tall  → half-height 0.006 m
_BOX_HALF_H = 0.030
_LID_HALF_H = 0.006


def design_scene() -> dict:
    # Ground plane — single config object, no double-construction
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/Ground", ground_cfg)

    # Dome light
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9))
    light_cfg.func("/World/Light", light_cfg)

    # Small Box — USD is already in meters; no extra scale.
    # Unique stage path /World/SmallBox prevents any prim-path conflict with lid.
    box_cfg = RigidObjectCfg(
        prim_path="/World/SmallBox",
        spawn=sim_utils.UsdFileCfg(usd_path=_BOX_USD),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, _BOX_HALF_H),  # centre sits at half-height above ground
        ),
    )
    box = RigidObject(cfg=box_cfg)

    # Lid Yellow — unique stage path /World/LidYellow.
    # Spawned above the box so it falls and settles on top (watch the drop in viewer).
    # The shared Props/instanceable_meshes.usd means the lid's VISUAL may show the
    # box mesh; see the module docstring for the permanent fix.
    lid_init_z = _BOX_HALF_H * 2 + _LID_HALF_H + 0.05  # box-top + lid half-h + gap
    lid_cfg = RigidObjectCfg(
        prim_path="/World/LidYellow",
        spawn=sim_utils.UsdFileCfg(usd_path=_LID_USD),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, lid_init_z),
        ),
    )
    lid = RigidObject(cfg=lid_cfg)

    return {"box": box, "lid": lid}


def run_simulator(sim: SimulationContext, entities: dict) -> None:
    box: RigidObject = entities["box"]
    lid: RigidObject = entities["lid"]
    sim_dt = sim.get_physics_dt()
    count = 0

    while simulation_app.is_running():
        # Reset every 300 steps so the lid keeps dropping onto the box
        if count % 300 == 0:
            for obj in (box, lid):
                root_state = obj.data.default_root_state.clone()
                obj.write_root_pose_to_sim(root_state[:, :7])
                obj.write_root_velocity_to_sim(root_state[:, 7:])
                obj.reset()
            if count > 0:
                print(f"[INFO] Step {count}: scene reset.")

        for obj in (box, lid):
            obj.write_data_to_sim()
        sim.step()
        for obj in (box, lid):
            obj.update(sim_dt)

        if count < 5:
            print(
                f"[DEBUG] step={count} | "
                f"box pos={box.data.root_pos_w[0].tolist()} | "
                f"lid pos={lid.data.root_pos_w[0].tolist()}"
            )
        count += 1


def main() -> None:
    sim = SimulationContext(
        sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    )
    # Camera pulled close — objects are ~6 cm tall after mm→m conversion
    sim.set_camera_view(eye=(0.3, 0.3, 0.2), target=(0.0, 0.0, 0.05))

    entities = design_scene()
    sim.reset()
    print("[INFO] Setup complete. Running simulation...")
    run_simulator(sim, entities)


if __name__ == "__main__":
    main()
    simulation_app.close()
