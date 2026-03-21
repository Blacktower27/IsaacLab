# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os as _os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
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

_RJ45_DIR = _os.path.normpath(
    _os.path.join(
        _os.path.dirname(_os.path.abspath(__file__)),
        "../../../../isaaclab_assets/isaaclab_assets/custom_assets/rj45/medium",
    )
)

_BNC_SMALL_DIR = _os.path.normpath(
    _os.path.join(
        _os.path.dirname(_os.path.abspath(__file__)),
        "../../../../isaaclab_assets/isaaclab_assets/custom_assets/bnc/small",
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
    # hand_init_pos[2] = 0.05 m: lid body bottom sits ~20 mm above box top,
    # clear of box walls. Engage-zone initialization is geometrically infeasible:
    # any clip-in-pocket position requires lid body inside box cavity → penetration.
    # hand_init_pos: list = [0.00, 0.07, 0.043] #for franka
    hand_init_pos: list = [0.00, 0.07, 0.034]  # [x, y, z] box-local; z used as fallback only
    # hand_init_pos_noise: list = [0.02, 0.02, 0.01]
    hand_init_pos_noise: list = [0.0, 0.0, 0.0]
    # hand_init_orn = [roll, pitch, yaw] in radians.  π on roll = EE pointing down.
    hand_init_orn: list = [3.1416, 0.0, 0.0]
    # Yaw is overridden in randomize_initial_state to align with box yaw (so clips face pockets).
    hand_init_orn_noise: list = [0.0, 0.0, 0.0]

    # --- Random initial pose (box_lid_insert) ---
    # XYZ are sampled uniformly within the ranges below (box-local frame, metres).
    # XY is rotated to world frame via box yaw before applying.
    # Z is the EE height above the box top face (fixed_tip_pos.z).
    # Yaw/pitch noise is added on top of the box-aligned yaw.
    hand_init_x_range: list = [-0.05, 0.05]   # X in box-local frame (m)
    hand_init_y_range: list = [0.0,   0.1]   # Y in box-local frame, back half (m)
    hand_init_z_range: list = [0.055, 0.085]  # Z above box top (m)
    # hand_init_yaw_noise_deg: float = 20.0     # ±yaw noise (deg) on top of box-aligned yaw
    # hand_init_pitch_noise_deg: float = 20.0   # ±pitch noise (deg)
    hand_init_yaw_noise_deg: float = 0.0     # ±yaw noise (deg) on top of box-aligned yaw
    hand_init_pitch_noise_deg: float = 0.0   # ±pitch noise (deg)

    # --- Fixed asset (box) randomisation ---
    # fixed_asset_init_pos_noise: list = [0.05, 0.05, 0.05]  # Z=0.05 allows vertical jitter
    fixed_asset_init_pos_noise: list = [0.05, 0.05, 0.0]  # Z fixed to table surface
    fixed_asset_init_orn_deg: float = 0.0
    # Full 360° yaw randomisation: the policy must handle the box at any orientation.
    fixed_asset_init_orn_range_deg: float = 360.0

    # --- Held asset (lid) in-gripper noise ---
    # Small positional jitter (3 mm per axis) to mimic real-world grasp uncertainty.
    held_asset_pos_noise: list = [0.003, 0.003, 0.003]
    # held_asset_rot_init = base yaw of the lid in the flipped fingertip frame.
    # 90° aligns the lid's long axis with the robot's approach direction.
    held_asset_rot_init: float = 90.0# for franka
    # held_asset_rot_init: float = 0.0
    # held_asset_rot_offset = additional [roll, pitch, yaw] in degrees on top of
    # held_asset_rot_init.  pitch=35° tilts the lid slightly forward so the handle
    # clears the finger pads during the closing step.
    held_asset_rot_offset: list = [0.0, 35.0, 0.0]# for franka
    # held_asset_rot_offset: list = [0.0,0, 0.0]
    # held_asset_rot_offset: list = [0.0, 0.0, 0.0]
    # held_asset_pos_offset = fine-tune translation [x, y, z] in the flipped
    # fingertip frame (metres).  y=0.02 shifts the lid 20 mm "inward" so the
    # handle is centred between the finger pads; z=0.005 compensates for the
    # 5 mm height difference between finger pad centre and handle top.
    held_asset_pos_offset: list = [0.0, 0.02, 0.005]
    # held_asset_pos_offset: list = [0.0, 0.0, -0.01]

    # --- Reward shaping (same structure as ForgePegInsert) ---
    # contact_penalty_scale: float = 0.2
    contact_penalty_scale: float = 0
    # Keypoint layout:
    #   index 0,1          = left/right clip positions (always fixed)
    #   index 2..2+nr-1    = extra front-face kps at reset (same Y,Z as clips; X random in lid-front range)
    #   index 2..2+ns-1    = random lid-body kps added on first success (overwrites extra reset kps)
    # Total buffer size = 2 + max(num_reset_extra_kp, num_success_extra_kp)
    # (num_keypoints is unused for box_lid_insert; computed automatically in factory_env.py)
    num_reset_extra_kp: int = 10    # extra front-face keypoints at reset
    num_success_extra_kp: int = 100  # random body keypoints added on first success
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
    # Box is a kinematic RigidObject: PhysX treats it as infinite-mass static body,
    # so contact forces against the lid are physically correct. This avoids the
    # fix_root_link=True incompatibility with FactoryEnv's gpu_max_num_partitions=1.
    fixed_asset: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/FixedAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=fixed_asset_cfg.usd_path,
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
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
                kinematic_enabled=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=fixed_asset_cfg.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # pos=(0.6, 0.0, 0.05),  # 0.05: bolt-fixture height used by other Factory tasks
            pos=(0.6, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)  # 0.0: box base flush with table surface
        ),
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


# ---------------------------------------------------------------------------
# [CUSTOM] RJ45 insertion task — asset configurations
# ---------------------------------------------------------------------------

@configclass
class RJ45FemaleCfg(FixedAssetCfg):
    """Adi_Medium_RJ45_Female_Raw_Colored.usd — fixed socket, receives the male plug.

    STL coordinate system (mm, converted to metres with scale 0.001).
    STL was modelled X-up; conversion applied -90° rotation around Y so that
    STL +X → sim +Z (Z-up world frame).

    In simulation (after -90° Y rotation):
        sim_X ∈ [-37.00, 16.10]  →  53.10 mm wide
        sim_Y ∈ [ -6.88, 24.38]  →  31.25 mm deep
        sim_Z ∈ [-22.23, 29.22]  →  51.45 mm tall

    USD origin is 22.23 mm above the socket base, 29.22 mm below the opening.
    """

    usd_path: str = f"{_RJ45_DIR}/Adi_Medium_RJ45_Female_Raw_Colored.usd"
    # height = distance from USD origin to socket opening (top face, where plug enters).
    # The init_state pos.z = 0.02223 already places the USD origin at the correct world
    # height (socket base flush with table), so no additional base_height offset is needed.
    height: float = 0.02922  # 29.22 mm
    # base_height is added to fixed_tip_pos_local in factory_env.py (line 913).
    # Set to 0.0 since the origin-to-table offset is handled via init_state pos.z.
    base_height: float = 0.0
    mass: float = 0.05
    friction: float = 0.75


@configclass
class RJ45MaleCfg(HeldAssetCfg):
    """Adi_Medium_RJ45_Male_Raw_Colored.usd — held by robot, inserted into socket.

    STL coordinate system (mm, converted to metres with scale 0.001).
    STL was modelled X-up; conversion applied -90° rotation around Y so that
    STL +X → sim +Z (Z-up world frame).

    In simulation (after -90° Y rotation):
        sim_X ∈ [-30.45,  9.55]  →  40.00 mm wide
        sim_Y ∈ [ -5.00, 28.08]  →  33.08 mm deep
        sim_Z ∈ [-13.98, 77.49]  →  91.47 mm tall

    USD origin sits between connector head and cable body.
    Connector tip (insertion end): sim_Z = -13.98 mm  (BELOW origin → negative base_height).
    Cable / grip end:              sim_Z = +77.49 mm  (used as height for EE offset).
    """

    usd_path: str = f"{_RJ45_DIR}/Adi_Medium_RJ45_Male_Raw_Colored.usd"
    # diameter = estimated grip width of the cable body (~12 mm).
    # Used by _set_franka_to_default_pose to set gripper finger separation.
    diameter: float = 0.012
    # height = sim_Z of the cable top from USD origin (grip end, 77.49 mm).
    # Used in get_handheld_asset_relative_pose to compute the EE → asset offset.
    height: float = 0.07749
    # base_height = sim_Z of the connector tip from USD origin (-13.98 mm).
    # Negative because the tip is BELOW the origin in sim_Z.
    # Used in factory_utils.get_held_base_pos_local as held_base_z_offset.
    base_height: float = -0.01398
    mass: float = 0.01
    friction: float = 0.75


@configclass
class ForgeRJ45Insert(ForgeTask):
    """FORGE RJ45 plug-into-socket insertion task.

    The robot holds the RJ45 male plug (by the cable end) and inserts it
    downward into the RJ45 female socket fixed on the table.

    Coordinate conventions:
        - All positions in metres.
        - hand_init_pos is relative to the fixed-asset tip (= socket opening).
        - The female socket opens upward (+Z); insertion direction is -Z.
    """

    name: str = "rj45_insert"
    fixed_asset_cfg: RJ45FemaleCfg = RJ45FemaleCfg()
    held_asset_cfg: RJ45MaleCfg = RJ45MaleCfg()
    # asset_size is informational; set to female socket width in mm.
    asset_size: float = 53.0
    duration_s: float = 10.0

    # --- Robot initial state (relative to socket opening) ---
    hand_init_pos: list = [0.00, 0.00, 0.05]   # 50 mm above socket opening
    hand_init_pos_noise: list = [0.0, 0.0, 0.0]
    hand_init_orn: list = [3.1416, 0.0, 0.0]   # EE pointing down
    hand_init_orn_noise: list = [0.0, 0.0, 0.0]

    # --- Random initial pose ---
    hand_init_x_range: list = [-0.03, 0.03]
    hand_init_y_range: list = [-0.03, 0.03]
    hand_init_z_range: list = [0.04, 0.08]
    hand_init_yaw_noise_deg: float = 0.0
    hand_init_pitch_noise_deg: float = 0.0

    # --- Fixed asset (socket) randomisation ---
    fixed_asset_init_pos_noise: list = [0.05, 0.05, 0.0]
    fixed_asset_init_orn_deg: float = 0.0
    fixed_asset_init_orn_range_deg: float = 360.0

    # --- Held asset (plug) in-gripper noise ---
    held_asset_pos_noise: list = [0.002, 0.002, 0.002]
    held_asset_rot_init: float = 0.0
    held_asset_pos_offset: list = [0.0, 0.0, 0.0]

    # --- Reward shaping ---
    contact_penalty_scale: float = 0.0
    keypoint_coef_baseline: list = [5, 4]
    keypoint_coef_coarse: list = [50, 2]
    keypoint_coef_fine: list = [100, 0]
    # engage_threshold > 1.0 → triggers the XY + yaw + tilt alignment check in
    #   _get_curr_successes.  Z is unconstrained — curr_engaged fires whenever the
    #   plug is XY-aligned (tip < 4 mm from opening centre) and correctly oriented
    #   (yaw < 20°, tilt < 15°), regardless of height above the socket.
    engage_threshold: float = 2.0
    # success_threshold < 0 → "halfway insertion" check.
    #   height_threshold = fixed_cfg.height * success_threshold = 0.02922 × (-0.24) ≈ -7 mm.
    #   curr_success fires once the connector tip is ≥ 7 mm inside the socket opening
    #   (z_disp < -0.007 m) AND XY is within 3 mm.
    #   The male connector head is ~14 mm long, so -0.24 requires roughly half-insertion.
    success_threshold: float = -0.24

    # --- Scene assets (RJ45 Female socket is kinematic RigidObject) ---
    fixed_asset: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/FixedAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=fixed_asset_cfg.usd_path,
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
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
                kinematic_enabled=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=fixed_asset_cfg.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # pos.z = base_height (0.02223 m) places the socket base flush with the
            # table surface.  The USD origin is 22.23 mm above the socket base.
            pos=(0.6, 0.0, 0.02223),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
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
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.4, 0.1), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )


# ---------------------------------------------------------------------------
# [CUSTOM] BNC Small insertion task — asset configurations
# ---------------------------------------------------------------------------

@configclass
class BNCSmallFemaleCfg(FixedAssetCfg):
    """Adi_BNC_Simulation_Small_Female.usd — fixed socket, receives the male plug.

    STL coordinate system (mm, converted to metres with scale 0.001, Z-up, no rotation):
        X in [-22.0, +22.0]  ->  44 mm wide  (symmetric)
        Y in [-22.0, +22.0]  ->  44 mm deep  (symmetric)
        Z in [-35.0, +25.0]  ->  60 mm tall

    USD origin is 35 mm above the socket base and 25 mm below the socket opening.
    init_state.pos.z = 0.035 m so the socket base is flush with the table surface.
    """

    usd_path: str = f"{_BNC_SMALL_DIR}/Adi_BNC_Simulation_Small_Female.usd"
    # height = distance from USD origin to socket opening (where plug enters) = 25 mm.
    height: float = 0.025
    # base_height handled via init_state.pos.z = 0.035 m.
    base_height: float = 0.0
    mass: float = 0.05
    friction: float = 0.75


@configclass
class BNCSmallMaleCfg(HeldAssetCfg):
    """Adi_BNC_Simulation_Small_Male.usd — held by robot, inserted into socket.

    STL coordinate system (mm, converted to metres with scale 0.001, Z-up, no rotation):
        X in [-20.7, +20.7]  ->  41.4 mm wide  (symmetric)
        Y in [-18.3, +18.3]  ->  36.6 mm deep  (symmetric)
        Z in [+36.235, +107.201]  ->  70.966 mm tall

    USD origin is 36.235 mm BELOW the connector tip (part is entirely above origin).
    Connector tip (insertion end): sim_Z = +36.235 mm  (ABOVE origin -> positive base_height).
    Body centre / grip point:      sim_Z = +71.7 mm.
    """

    usd_path: str = f"{_BNC_SMALL_DIR}/Adi_BNC_Simulation_Small_Male.usd"
    # diameter = BNC barrel outer diameter (~20 mm), for gripper finger separation.
    diameter: float = 0.020
    # height = sim_Z of the grip point (body centre at 36.235 + 35.5 = 71.7 mm).
    height: float = 0.0717
    # base_height = sim_Z of connector tip from USD origin (+36.235 mm).
    # Positive because the tip is ABOVE the origin.
    base_height: float = 0.036235
    mass: float = 0.01
    friction: float = 0.75


@configclass
class ForgeBNCSmallInsert(ForgeTask):
    """FORGE BNC Small plug-into-socket insertion task.

    The robot holds the BNC male plug and inserts it downward into the
    BNC female socket fixed on the table.

    Coordinate conventions:
        - All positions in metres.
        - hand_init_pos is relative to the fixed-asset tip (socket opening).
        - The female socket opens upward (+Z); insertion direction is -Z.
    """

    name: str = "bnc_insert"
    fixed_asset_cfg: BNCSmallFemaleCfg = BNCSmallFemaleCfg()
    held_asset_cfg: BNCSmallMaleCfg = BNCSmallMaleCfg()
    asset_size: float = 44.0
    duration_s: float = 10.0

    # --- Robot initial state (relative to socket opening) ---
    hand_init_pos: list = [0.00, 0.00, 0.05]
    hand_init_pos_noise: list = [0.0, 0.0, 0.0]
    hand_init_orn: list = [3.1416, 0.0, 0.0]
    hand_init_orn_noise: list = [0.0, 0.0, 0.0]

    # --- Random initial pose ---
    hand_init_x_range: list = [-0.03, 0.03]
    hand_init_y_range: list = [-0.03, 0.03]
    hand_init_z_range: list = [0.04, 0.08]
    hand_init_yaw_noise_deg: float = 0.0
    hand_init_pitch_noise_deg: float = 0.0

    # --- Fixed asset (socket) randomisation ---
    fixed_asset_init_pos_noise: list = [0.05, 0.05, 0.0]
    fixed_asset_init_orn_deg: float = 0.0
    # BNC is cylindrically symmetric -> full 360 deg yaw randomisation.
    fixed_asset_init_orn_range_deg: float = 360.0

    # --- Held asset (plug) in-gripper noise ---
    held_asset_pos_noise: list = [0.002, 0.002, 0.002]
    held_asset_rot_init: float = 0.0
    held_asset_pos_offset: list = [0.0, 0.0, 0.0]

    # --- Reward shaping ---
    contact_penalty_scale: float = 0.0
    # Standard axis keypoints (4 points along Z, +/-15 mm spread).
    num_keypoints: int = 4
    keypoint_scale: float = 0.03
    keypoint_coef_baseline: list = [5, 4]
    keypoint_coef_coarse: list = [50, 2]
    keypoint_coef_fine: list = [100, 0]
    # engage_threshold > 1.0 -> XY + yaw + tilt alignment check (same as rj45_insert).
    engage_threshold: float = 2.0
    # success_threshold < 0 -> tip must be 7.5 mm inside socket (0.025 x 0.3 = 7.5 mm).
    success_threshold: float = -0.3

    # --- Scene assets (BNC Female socket is kinematic RigidObject) ---
    fixed_asset: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/FixedAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=fixed_asset_cfg.usd_path,
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
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
                kinematic_enabled=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=fixed_asset_cfg.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # pos.z = 0.035 m: socket base (STL Z=-35 mm) flush with table surface.
            pos=(0.6, 0.0, 0.035),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
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
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.001, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.4, 0.1), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
    )
