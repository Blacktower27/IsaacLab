# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os as _os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

from isaaclab_tasks.direct.factory.factory_tasks_cfg import (
    FactoryTask,
    FixedAssetCfg,
    GearMesh,
    HeldAssetCfg,
    NutThread,
    PegInsert,
)

# Absolute path to the custom box/middle asset directory.
_MIDDLE_BOX_DIR = _os.path.normpath(
    _os.path.join(
        _os.path.dirname(_os.path.abspath(__file__)),
        "../../../../isaaclab_assets/isaaclab_assets/custom_assets/box/middle",
    )
)


@configclass
class ForgeTask(FactoryTask):
    action_penalty_ee_scale: float = 0.0
    action_penalty_asset_scale: float = 0.001
    action_grad_penalty_scale: float = 0.1
    contact_penalty_scale: float = 0.05
    delay_until_ratio: float = 0.25
    contact_penalty_threshold_range = [5.0, 10.0]


@configclass
class ForgePegInsert(PegInsert, ForgeTask):
    contact_penalty_scale: float = 0.2


@configclass
class ForgeGearMesh(GearMesh, ForgeTask):
    contact_penalty_scale: float = 0.05


@configclass
class ForgeNutThread(NutThread, ForgeTask):
    contact_penalty_scale: float = 0.05


# ---------------------------------------------------------------------------
# Custom box-lid insertion task assets
# ---------------------------------------------------------------------------

@configclass
class SmallBoxCfg(FixedAssetCfg):
    """Small_Box.usd — fixed on table, receives the lid."""

    usd_path: str = f"{_MIDDLE_BOX_DIR}/Small_Box.usd"
    # STL bounding box (mm → m after 0.001 scale):
    #   X: 120 mm  Y: 100 mm  Z: 0→30 mm
    height: float = 0.030  # full box height in metres
    base_height: float = 0.0
    mass: float = 0.1
    friction: float = 0.75


@configclass
class LidYellowCfg(HeldAssetCfg):
    """Lid_Yellow.usd — held by the robot arm, placed onto the box.

    Grip feature geometry (STL coords, mm):
        Handle: X ∈ [-8.5, 8.5], Y ∈ [-0.6, 37.9], Z ∈ [45, 55]
        → width 17 mm (X), top at Z = 55 mm from USD origin.
    Lid body bottom: Z ≈ 18.8 mm from USD origin.
    """

    usd_path: str = f"{_MIDDLE_BOX_DIR}/Lid_Yellow.usd"
    diameter: float = 0.017  # handle width in X — sets gripper opening
    height: float = 0.055  # handle top height from USD origin — used for grasp offset
    mass: float = 0.02
    friction: float = 0.75


# Offset from the lid's USD origin to its bottom face (in metres).
_LID_BOTTOM_Z_OFFSET: float = 0.0188


@configclass
class ForgeBoxLidInsert(ForgeTask):
    """FORGE box-lid insertion task.

    The robot grasps the Lid_Yellow by its top handle and places it onto the
    Small_Box.  Success is defined as the lid bottom face reaching the box
    top face within a small XY/Z tolerance.
    """

    name: str = "box_lid_insert"
    fixed_asset_cfg: SmallBoxCfg = SmallBoxCfg()
    held_asset_cfg: LidYellowCfg = LidYellowCfg()
    asset_size: float = 100.0  # box outer X in mm (informational)
    duration_s: float = 15.0

    # --- Robot initial state (relative to fixed-asset tip = box top) ---
    hand_init_pos: list = [0.0, 0.0, 0.10]  # 100 mm above box top
    hand_init_pos_noise: list = [0.02, 0.02, 0.01]
    hand_init_orn: list = [3.1416, 0.0, 0.0]  # EE pointing down
    hand_init_orn_noise: list = [0.0, 0.0, 0.785]  # ±45° yaw noise

    # --- Fixed asset randomisation ---
    fixed_asset_init_pos_noise: list = [0.05, 0.05, 0.05]
    fixed_asset_init_orn_deg: float = 0.0
    fixed_asset_init_orn_range_deg: float = 360.0

    # --- Held asset in-gripper noise ---
    held_asset_pos_noise: list = [0.003, 0.003, 0.003]
    held_asset_rot_init: float = 90.0  # base yaw around Z (deg)
    # Additional rotation offsets [roll, pitch, yaw] in degrees applied on top of held_asset_rot_init
    held_asset_rot_offset: list = [0.0, 35.0, 0.0]
    # Position offset [x, y, z] in flipped fingertip frame (m)
    held_asset_pos_offset: list = [0.0, 0.02, 0.005]

    # --- Reward shaping (same scale as ForgePegInsert) ---
    contact_penalty_scale: float = 0.2
    keypoint_coef_baseline: list = [5, 4]
    keypoint_coef_coarse: list = [50, 2]
    keypoint_coef_fine: list = [100, 0]
    success_threshold: float = 0.04   # fraction of box height ≈ 1.2 mm
    engage_threshold: float = 0.9

    # --- Scene assets ---
    fixed_asset: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/FixedAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=fixed_asset_cfg.usd_path,
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                fix_root_link=False,  # gravity holds box in place; can be nudged
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=fixed_asset_cfg.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.6, 0.0, 0.05), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )
    held_asset: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/HeldAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=held_asset_cfg.usd_path,
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                fix_root_link=False,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=3666.0,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=192,
                solver_velocity_iteration_count=1,
                max_contact_impulse=1e32,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=held_asset_cfg.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.4, 0.1), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )
