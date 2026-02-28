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

# ---------------------------------------------------------------------------
# [CUSTOM] Absolute path to the custom box/middle asset directory.
#
# Resolved at import time relative to this config file so the path is valid
# regardless of the working directory from which the training script is launched.
# normpath removes redundant ".." segments (e.g. "a/b/../../c" → "c").
# ---------------------------------------------------------------------------
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
# [CUSTOM] Box-lid insertion task — asset configurations
# ---------------------------------------------------------------------------

@configclass
class SmallBoxCfg(FixedAssetCfg):
    """Small_Box.usd — fixed on table, receives the lid.

    STL coordinate system (mm, converted to metres with scale 0.001):
        X ∈ [-60, 60]  →  120 mm wide
        Y ∈ [-50, 50]  →  100 mm deep
        Z ∈ [  0, 30]  →   30 mm tall; USD origin is at the base face (Z=0).

    The USD was converted with collision_approximation=triangleMesh so that the
    inner cavity walls are exact — critical for lid-insertion accuracy.
    (triangleMesh is valid here because the box is a static/fixed body.)
    """

    usd_path: str = f"{_MIDDLE_BOX_DIR}/Small_Box.usd"
    # height = full box height in metres.
    # This value is used by factory_utils.get_target_held_base_pose to set the
    # target Z of the lid bottom face (= box top face = height above box origin).
    height: float = 0.030
    # base_height = distance from the USD prim origin to the bottom of the object.
    # For Small_Box the USD origin is already at the base, so no offset is needed.
    base_height: float = 0.0
    mass: float = 0.1
    friction: float = 0.75


@configclass
class LidYellowCfg(HeldAssetCfg):
    """Lid_Yellow.usd — held by the robot arm, placed onto the box.

    STL coordinate system (mm → metres after 0.001 scale):
        Lid body:  X ∈ [-52.3,  52.3],  Y ∈ [-44.4, 50.4],  Z ∈ [18.8, 55.0]
        Handle:    X ∈ [ -8.5,   8.5],  Y ∈ [ -0.6, 37.9],  Z ∈ [45.0, 55.0]
            → handle width (X) = 17 mm,  handle top (Z) = 55 mm from USD origin.
        Lid body bottom: Z ≈ 18.8 mm from the USD origin.

    The USD was converted with collision_approximation=convexDecomposition with
    tightened precision parameters (error_percentage=1%, hull_vertex_limit=256,
    shrink_wrap=True) because it is a dynamic body (PhysX requires convex shapes
    for dynamic rigid bodies).
    """

    usd_path: str = f"{_MIDDLE_BOX_DIR}/Lid_Yellow.usd"
    # diameter = handle width along X (17 mm).
    # Used by _set_franka_to_default_pose to set the gripper finger separation:
    #   gripper_width = diameter / 2 * 1.25  (25 % wider than the handle for safety).
    diameter: float = 0.017
    # height = Z position of the handle top from the USD prim origin (55 mm).
    # Used by get_handheld_asset_relative_pose to compute the EE offset so that
    # the finger pads are aligned with the top of the handle at grasp time.
    height: float = 0.055
    # base_height = Z distance from the USD prim origin (0,0,0) to the geometric
    # BOTTOM FACE of the lid body (18.8 mm = STL Z_min × 0.001).
    # The lid STL has its origin NOT at its base — the body starts 18.8 mm above
    # the origin.  This offset is used in factory_utils.get_held_base_pos_local
    # to transform the USD root position to the true contact face of the lid,
    # which is the reference point for the keypoint reward and success check.
    # Note: HeldAssetCfg (the upstream base class) does not define base_height;
    # this field is custom to LidYellowCfg.  It is accessed via
    #   self.cfg_task.held_asset_cfg.base_height  if needed at runtime.
    base_height: float = 0.0188
    mass: float = 0.02
    friction: float = 0.75


@configclass
class ForgeBoxLidInsert(ForgeTask):
    """FORGE box-lid insertion task.

    The robot grasps Lid_Yellow by its top handle and places it onto Small_Box.
    Success is defined as the lid bottom face reaching the box top face within
    a tight XY/Z tolerance (see success_threshold below).

    Coordinate conventions used in this config:
        - All positions in metres unless stated otherwise.
        - hand_init_pos / hand_init_orn are relative to the fixed-asset tip
          (= box top face), not to the world origin.
        - held_asset_rot_offset is applied in the FLIPPED fingertip frame
          (see factory_env.get_handheld_asset_relative_pose for details).
    """

    # Task name string: must match the branch keys added to factory_utils.py and
    # factory_env.py (get_held_base_pos_local, get_target_held_base_pose,
    # get_handheld_asset_relative_pose, _get_curr_successes).
    name: str = "box_lid_insert"
    fixed_asset_cfg: SmallBoxCfg = SmallBoxCfg()
    held_asset_cfg: LidYellowCfg = LidYellowCfg()
    # asset_size is informational only (used in some logging/metric paths in
    # upstream Factory code).  Set to the box's outer X dimension in mm.
    asset_size: float = 100.0
    duration_s: float = 15.0

    # --- Robot initial state (relative to fixed-asset tip = box top face) ---
    # hand_init_pos[2] = 0.10 m → EE starts 100 mm above the box top, giving
    # enough clearance to avoid clipping the box during the grasp reset.
    hand_init_pos: list = [0.0, 0.0, 0.10]
    hand_init_pos_noise: list = [0.02, 0.02, 0.01]
    # hand_init_orn = [roll, pitch, yaw] in radians.  π on roll = EE pointing down.
    hand_init_orn: list = [3.1416, 0.0, 0.0]
    # ±45° yaw noise so the policy must learn to handle arbitrary approach angles.
    hand_init_orn_noise: list = [0.0, 0.0, 0.785]

    # --- Fixed asset (box) randomisation ---
    fixed_asset_init_pos_noise: list = [0.05, 0.05, 0.05]
    fixed_asset_init_orn_deg: float = 0.0
    # Full 360° yaw randomisation: the policy must handle the box at any orientation.
    fixed_asset_init_orn_range_deg: float = 360.0

    # --- Held asset (lid) in-gripper noise ---
    # Small positional jitter (3 mm per axis) to mimic real-world grasp uncertainty.
    held_asset_pos_noise: list = [0.003, 0.003, 0.003]
    # held_asset_rot_init = base yaw of the lid in the flipped fingertip frame.
    # 90° aligns the lid's long axis with the robot's approach direction.
    held_asset_rot_init: float = 90.0
    # held_asset_rot_offset = additional [roll, pitch, yaw] in degrees on top of
    # held_asset_rot_init.  pitch=35° tilts the lid slightly forward so the handle
    # clears the finger pads during the closing step.
    held_asset_rot_offset: list = [0.0, 35.0, 0.0]
    # held_asset_pos_offset = fine-tune translation [x, y, z] in the flipped
    # fingertip frame (metres).  y=0.02 shifts the lid 20 mm "inward" so the
    # handle is centred between the finger pads; z=0.005 compensates for the
    # 5 mm height difference between finger pad centre and handle top.
    held_asset_pos_offset: list = [0.0, 0.02, 0.005]

    # --- Reward shaping (same structure as ForgePegInsert) ---
    contact_penalty_scale: float = 0.2
    keypoint_coef_baseline: list = [5, 4]
    keypoint_coef_coarse: list = [50, 2]
    keypoint_coef_fine: list = [100, 0]
    # success_threshold < 0.5  → _get_curr_successes uses snap-fit CLIP ENGAGEMENT check
    #   (success_threshold value itself is unused in that branch).
    # engage_threshold  >= 0.5 → _get_curr_successes uses Z-distance height-fraction check:
    #   height_threshold = fixed_asset.height × engage_threshold = 0.030 × 0.9 = 0.027 m.
    #   curr_engaged fires when lid base is within 27 mm of the assembled Z target AND
    #   XY-centred within 2.5 mm, i.e. approximately when the lid body starts entering
    #   the box cavity. This provides an intermediate reward gradient between the dense
    #   keypoint reward and the sparse clip-engagement success bonus.
    success_threshold: float = 0.04
    engage_threshold: float = 0.9

    # --- Scene assets ---
    fixed_asset: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/FixedAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=fixed_asset_cfg.usd_path,
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                # fix_root_link=False: the box is NOT kinematically frozen; gravity
                # holds it on the table and a strong shove can move it (realistic).
                # Keeps the physics pipeline consistent with other Factory tasks.
                fix_root_link=False,
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
            # [CUSTOM] contact_offset reduced from the Factory default of 0.005 m
            # (5 mm) to 0.001 m (1 mm).
            # contact_offset is the distance at which PhysX starts generating
            # contact constraints.  With the box-lid clearance of only a few mm,
            # the default 5 mm would cause PhysX to push the lid away BEFORE it
            # geometrically reaches the box top, preventing the lid from seating.
            # NOTE: this runtime value overrides whatever is baked into the USD file.
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
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
                # disable_gravity=True: gravity is turned off for the held asset so
                # the lid does not fall out of the gripper during the reset phase
                # before the grasp closes.  Gravity is re-enabled implicitly once
                # the gripper is closed (the finger forces dominate).
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
            # [CUSTOM] Same 1 mm contact_offset as the fixed asset — see comment
            # on the fixed_asset collision_props above for the rationale.
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.4, 0.1), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )
