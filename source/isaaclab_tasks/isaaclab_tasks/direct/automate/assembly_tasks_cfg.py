# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os as _os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/AutoMate"

_CUSTOM_ASSETS_DIR = _os.path.normpath(
    _os.path.join(
        _os.path.dirname(_os.path.abspath(__file__)),
        "../../../../isaaclab_assets/isaaclab_assets/custom_assets",
    )
)
_MIDDLE_BOX_DIR = f"{_CUSTOM_ASSETS_DIR}/box/middle"
_RJ45_DIR = f"{_CUSTOM_ASSETS_DIR}/rj45/medium"
_BNC_SMALL_DIR = f"{_CUSTOM_ASSETS_DIR}/bnc/small"
# Repo-root ``scripts/rj45_ref_traj_automate.json`` (init→insert, Automate ``fingertip_centered_pos``).
_RJ45_REF_TRAJ_AUTOMATE_JSON = _os.path.normpath(
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "../../../../../scripts/rj45_ref_traj_automate.json")
)

OBS_DIM_CFG = {
    "fingertip_pos": 3,
    "fingertip_pos_rel_fixed": 3,
    "fingertip_quat": 4,
    "ee_linvel": 3,
    "ee_angvel": 3,
}

STATE_DIM_CFG = {
    "fingertip_pos": 3,
    "fingertip_pos_rel_fixed": 3,
    "fingertip_quat": 4,
    "ee_linvel": 3,
    "ee_angvel": 3,
    "joint_pos": 7,
    "held_pos": 3,
    "held_pos_rel_fixed": 3,
    "held_quat": 4,
    "fixed_pos": 3,
    "fixed_quat": 4,
    "task_prop_gains": 6,
    "ema_factor": 1,
    "pos_threshold": 3,
    "rot_threshold": 3,
}


@configclass
class FixedAssetCfg:
    usd_path: str = ""
    diameter: float = 0.0
    height: float = 0.0
    base_height: float = 0.0  # Used to compute held asset CoM.
    friction: float = 0.75
    mass: float = 0.05


@configclass
class HeldAssetCfg:
    usd_path: str = ""
    diameter: float = 0.0  # Used for gripper width.
    height: float = 0.0
    friction: float = 0.75
    mass: float = 0.05


@configclass
class RobotCfg:
    robot_usd: str = ""
    franka_fingerpad_length: float = 0.017608
    friction: float = 0.75


@configclass
class AssemblyTask:
    robot_cfg: RobotCfg = RobotCfg()
    name: str = ""
    duration_s = 5.0

    fixed_asset_cfg: FixedAssetCfg = FixedAssetCfg()
    held_asset_cfg: HeldAssetCfg = HeldAssetCfg()
    asset_size: float = 0.0

    # palm_to_finger_dist: float = 0.1034
    palm_to_finger_dist: float = 0.1134

    # Robot
    hand_init_pos: list = [0.0, 0.0, 0.015]  # Relative to fixed asset tip.
    hand_init_pos_noise: list = [0.02, 0.02, 0.01]
    hand_init_orn: list = [3.1416, 0, 2.356]
    hand_init_orn_noise: list = [0.0, 0.0, 1.57]

    # Action
    unidirectional_rot: bool = False

    # Fixed Asset (applies to all tasks)
    fixed_asset_init_pos_noise: list = [0.05, 0.05, 0.05]
    fixed_asset_init_orn_deg: float = 0.0
    fixed_asset_init_orn_range_deg: float = 10.0

    # Held Asset (applies to all tasks)
    # held_asset_pos_noise: list = [0.0, 0.006, 0.003]  # noise level of the held asset in gripper
    held_asset_init_pos_noise: list = [0.01, 0.01, 0.01]
    held_asset_pos_noise: list = [0.0, 0.0, 0.0]
    held_asset_rot_init: float = 0.0

    # Reward
    ee_success_yaw: float = 0.0  # nut_threading task only.
    action_penalty_scale: float = 0.0
    action_penalty_ee_scale: float = 0.0  # Forge-style: L2 penalty on actions (per-env norms in env).
    action_grad_penalty_scale: float = 0.0
    # Reward function details can be found in Appendix B of https://arxiv.org/pdf/2408.04587.
    # Multi-scale keypoints are used to capture different phases of the task.
    # Each reward passes the keypoint distance, x, through a squashing function:
    #     r(x) = 1/(exp(-ax) + b + exp(ax)).
    # Each list defines [a, b] which control the slope and maximum of the squashing function.
    num_keypoints: int = 4
    keypoint_scale: float = 0.15

    # Fixed-asset height fraction for which different bonuses are rewarded (see individual tasks).
    success_threshold: float = 0.04
    engage_threshold: float = 0.9

    # SDF reward
    sdf_rwd_scale: float = 1.0
    num_mesh_sample_points: int = 1000

    # Imitation reward
    imitation_rwd_scale: float = 1.0
    soft_dtw_gamma: float = 0.01  # set to 0 if want to use the original DTW without any smoothing
    num_point_robot_traj: int = 10  # number of waypoints included in the end-effector trajectory
    # When JSON has ``fingertip_centered_pose`` (7 = xyz + wxyz), Soft-DTW uses pose; weights scale pos vs quat in R^7.
    imitation_pose_pos_w: float = 1.0
    imitation_pose_rot_w: float = 0.35

    # SBC
    initial_max_disp: float = 0.01  # max initial downward displacement of plug at beginning of curriculum
    curriculum_success_thresh: float = 0.8  # success rate threshold for increasing curriculum difficulty
    curriculum_failure_thresh: float = 0.5  # success rate threshold for decreasing curriculum difficulty
    curriculum_freespace_range: float = 0.01
    num_curriculum_step: int = 10
    curriculum_height_step: list = [
        -0.005,
        0.003,
    ]  # how much to increase max initial downward displacement after hitting success or failure thresh

    if_sbc: bool = True

    # Logging evaluation results
    if_logging_eval: bool = False
    num_eval_trials: int = 100
    eval_filename: str = "evaluation_00015.h5"


@configclass
class Peg8mm(HeldAssetCfg):
    usd_path = "plug.usd"
    obj_path = "plug.obj"
    diameter = 0.007986
    height = 0.050
    mass = 0.019


@configclass
class Hole8mm(FixedAssetCfg):
    usd_path = "socket.usd"
    obj_path = "socket.obj"
    diameter = 0.0081
    height = 0.050896
    base_height = 0.0


@configclass
class Insertion(AssemblyTask):
    name = "insertion"

    assembly_id = "00015"
    assembly_dir = f"{ASSET_DIR}/{assembly_id}/"

    fixed_asset_cfg = Hole8mm()
    held_asset_cfg = Peg8mm()
    asset_size = 8.0
    duration_s = 10.0

    plug_grasp_json = f"{ASSET_DIR}/plug_grasps.json"
    disassembly_dist_json = f"{ASSET_DIR}/disassembly_dist.json"
    disassembly_path_json = f"{assembly_dir}/disassemble_traj.json"

    # Robot
    hand_init_pos: list = [0.0, 0.0, 0.047]  # Relative to fixed asset tip.
    hand_init_pos_noise: list = [0.02, 0.02, 0.01]
    hand_init_orn: list = [3.1416, 0.0, 0.0]
    hand_init_orn_noise: list = [0.0, 0.0, 0.785]
    hand_width_max: float = 0.080  # maximum opening width of gripper

    # Fixed Asset (applies to all tasks)
    fixed_asset_init_pos_noise: list = [0.05, 0.05, 0.05]
    fixed_asset_init_orn_deg: float = 0.0
    fixed_asset_init_orn_range_deg: float = 10.0
    fixed_asset_z_offset: float = 0.1435

    # Held Asset (applies to all tasks)
    # held_asset_pos_noise: list = [0.003, 0.0, 0.003]  # noise level of the held asset in gripper
    held_asset_init_pos_noise: list = [0.01, 0.01, 0.01]
    held_asset_pos_noise: list = [0.0, 0.0, 0.0]
    held_asset_rot_init: float = 0.0

    # Rewards
    keypoint_coef_baseline: list = [5, 4]
    keypoint_coef_coarse: list = [50, 2]
    keypoint_coef_fine: list = [100, 0]
    # Fraction of socket height.
    success_threshold: float = 0.04
    engage_threshold: float = 0.9
    engage_height_thresh: float = 0.01
    success_height_thresh: float = 0.003
    close_error_thresh: float = 0.015

    fixed_asset: ArticulationCfg = ArticulationCfg(
        # fixed_asset: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/FixedAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{assembly_dir}{fixed_asset_cfg.usd_path}",
            activate_contact_sensors=True,
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
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                fix_root_link=True,  # add this so the fixed asset is set to have a fixed base
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=fixed_asset_cfg.mass),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            # init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.6, 0.0, 0.05),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={},
            joint_vel={},
        ),
        actuators={},
    )
    # held_asset: ArticulationCfg = ArticulationCfg(
    held_asset: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/HeldAsset",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{assembly_dir}{held_asset_cfg.usd_path}",
            activate_contact_sensors=True,
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
        # init_state=ArticulationCfg.InitialStateCfg(
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.4, 0.1),
            rot=(1.0, 0.0, 0.0, 0.0),
            # joint_pos={},
            # joint_vel={}
        ),
        # actuators={}
    )


# ---------------------------------------------------------------------------
# Box-Lid insertion task (ported from forge)
# ---------------------------------------------------------------------------

@configclass
class SmallBoxCfg(FixedAssetCfg):
    usd_path: str = f"{_MIDDLE_BOX_DIR}/Small_Box.usd"
    height: float = 0.030
    base_height: float = 0.0
    mass: float = 0.1
    friction: float = 0.75


@configclass
class LidYellowCfg(HeldAssetCfg):
    usd_path: str = f"{_MIDDLE_BOX_DIR}/Lid_Yellow.usd"
    diameter: float = 0.017
    height: float = 0.055
    base_height: float = 0.0188
    mass: float = 0.02
    friction: float = 0.75


@configclass
class BoxLidInsertion(AssemblyTask):
    name = "box_lid_insert"
    fixed_asset_cfg = SmallBoxCfg()
    held_asset_cfg = LidYellowCfg()
    asset_size = 100.0
    duration_s = 30.0

    # No external data files — these features are disabled.
    assembly_id = ""
    assembly_dir = ""
    plug_grasp_json = ""
    disassembly_dist_json = ""
    disassembly_path_json = ""

    # Simple grasp: grip the lid handle top, identity orientation (xyzw).
    simple_grasp_pos_local: list = [0.0, 0.0, 0.055]
    simple_grasp_quat_local: list = [0.0, 0.0, 0.0, 1.0]
    default_disassembly_dist: float = 0.030
    gripper_open_width_override: float = 0.014

    # SDF unavailable (no OBJ) → keypoint-distance reward used instead at this scale.
    sdf_rwd_scale: float = 1.0
    imitation_rwd_scale: float = 0.0
    if_sbc: bool = False

    # Robot initial state (relative to fixed-asset tip = box top face)
    hand_init_pos: list = [0.00, 0.07, 0.034]
    hand_init_pos_noise: list = [0.0, 0.0, 0.0]
    hand_init_orn: list = [3.1416, 0.0, 0.0]
    hand_init_orn_noise: list = [0.0, 0.0, 0.0]
    hand_width_max: float = 0.080

    # Fixed asset
    fixed_asset_init_pos_noise: list = [0.05, 0.05, 0.0]
    fixed_asset_init_orn_deg: float = 0.0
    fixed_asset_init_orn_range_deg: float = 360.0
    fixed_asset_z_offset: float = 0.0

    # Held asset
    held_asset_init_pos_noise: list = [0.003, 0.003, 0.003]
    held_asset_pos_noise: list = [0.0, 0.0, 0.0]
    held_asset_rot_init: float = 90.0
    held_asset_rot_offset: list = [0.0, 35.0, 0.0]
    held_asset_pos_offset: list = [0.0, 0.02, 0.005]

    # Init (Forge: near / far / mixed)
    init_mode: str = "far"
    near_init_prob: float = 0.5
    hand_init_x_range: list = [-0.05, 0.05]
    hand_init_y_range: list = [0.0, 0.1]
    hand_init_z_range: list = [0.055, 0.085]
    hand_init_yaw_noise_deg: float = 20.0
    hand_init_pitch_noise_deg: float = 0.0

    # Keypoints & reward (Forge multi-scale + frame rotation)
    num_reset_extra_kp: int = 10
    num_success_extra_kp: int = 100
    rot_weight: float = 0.05
    action_penalty_ee_scale: float = 0.0

    # Reward
    keypoint_coef_baseline: list = [5, 4]
    keypoint_coef_coarse: list = [50, 2]
    keypoint_coef_fine: list = [100, 0]
    success_threshold: float = 0.04
    engage_threshold: float = 0.9
    close_error_thresh: float = 0.015

    # Kinematic box (RigidObject), lid as Articulation — matches Forge (avoids fix_root_link + gpu_max_num_partitions).
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
            pos=(0.6, 0.0, 0.0),
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
            pos=(0.0, 0.4, 0.1),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={},
            joint_vel={},
        ),
        actuators={},
    )


# ---------------------------------------------------------------------------
# RJ45 insertion task (ported from forge)
# ---------------------------------------------------------------------------

@configclass
class RJ45FemaleCfg(FixedAssetCfg):
    usd_path: str = f"{_RJ45_DIR}/rs_female_rj45.usd"
    height: float = 0.0
    base_height: float = 0.0
    mass: float = 0.05
    friction: float = 0.75


@configclass
class RJ45MaleCfg(HeldAssetCfg):
    usd_path: str = f"{_RJ45_DIR}/rs_male_rj45.usd"
    diameter: float = 0.012
    height: float = 0.08847
    base_height: float = -0.003
    mass: float = 0.01
    friction: float = 0.75


@configclass
class RJ45Insertion(AssemblyTask):
    name = "rj45_insert"
    fixed_asset_cfg = RJ45FemaleCfg()
    held_asset_cfg = RJ45MaleCfg()
    asset_size = 53.0
    duration_s = 30.0

    assembly_id = ""
    assembly_dir = ""
    plug_grasp_json = ""
    disassembly_dist_json = ""
    # Soft-DTW imitation: default to bundled Automate-format demos; override with Hydra if you use your own file.
    disassembly_path_json: str = _RJ45_REF_TRAJ_AUTOMATE_JSON

    simple_grasp_pos_local: list = [0.0, 0.0, 0.08847]
    simple_grasp_quat_local: list = [0.0, 0.0, 0.0, 1.0]
    default_disassembly_dist: float = 0.050
    gripper_open_width_override: float = 0.010

    sdf_rwd_scale: float = 1.0
    imitation_rwd_scale: float = 1.0
    if_sbc: bool = False

    hand_init_pos: list = [0.00, 0.00, 0.091]
    hand_init_pos_noise: list = [0.0, 0.0, 0.0]
    hand_init_orn: list = [3.1416, 0.0, 0.0]
    hand_init_orn_noise: list = [0.0, 0.0, 0.0]
    hand_width_max: float = 0.080

    fixed_asset_init_pos_noise: list = [0.05, 0.05, 0.0]
    fixed_asset_init_orn_deg: float = 0.0
    fixed_asset_init_orn_range_deg: float = 360.0
    fixed_asset_z_offset: float = 0.0

    held_asset_init_pos_noise: list = [0.0, 0.0, 0.0]
    held_asset_pos_noise: list = [0.0, 0.0, 0.0]
    held_asset_rot_init: float = 180.0
    held_asset_rot_offset: list = [0.0, 0.0, 0.0]
    held_asset_pos_offset: list = [0.0, 0.0, -0.06]

    socket_target_y_local: float = -0.006
    socket_target_z_local: float = 0.014

    init_mode: str = "contact"
    near_init_prob: float = 0.5
    hand_init_x_range: list = [-0.03, 0.01]
    hand_init_y_range: list = [-0.02, 0.02]
    hand_init_z_range: list = [0.081, 0.091]
    hand_init_yaw_noise_deg: float = 10.0
    hand_init_pitch_noise_deg: float = 0.0
    hand_init_yaw_offset_deg: float = 0.0

    female_rear_edge_x_range_local: list = [-0.020, 0.020]
    female_rear_edge_y_local: float = -0.0186
    female_rear_edge_z_local: float = 0.0
    male_bottom_patch_x_range_local: list = [-0.018, 0.018]
    male_bottom_patch_y_range_local: list = [-0.006, 0.009]
    male_bottom_patch_z_local: float = -0.06
    contact_init_roll_range_deg: list = [0.0, 0.0]
    contact_init_pitch_range_deg: list = [0.0, 0.0]
    contact_init_yaw_range_deg: list = [-10.0, 10.0]

    num_reset_kp: int = 64
    num_success_kp: int = 10
    kp_advance_threshold: float = 0.005
    kp_advance_step: float = 0.0002
    kp_advance_z_limit: float = -0.060
    rot_weight: float = 1.0
    action_penalty_ee_scale: float = 0.0

    keypoint_coef_baseline: list = [5, 4]
    keypoint_coef_coarse: list = [50, 2]
    keypoint_coef_fine: list = [100, 0]
    success_threshold: float = 0.027
    # Unused by Automate reward (no Forge engage term); kept for cfg parity / external tools.
    engage_threshold: float = 2.0
    close_error_thresh: float = 0.015

    # Female socket: kinematic RigidObject; male plug: Articulation (matches Forge).
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
            pos=(0.6, 0.0, 0.05145),
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
            pos=(0.0, 0.4, 0.1),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={},
            joint_vel={},
        ),
        actuators={},
    )


# ---------------------------------------------------------------------------
# BNC Small insertion task (ported from forge)
# ---------------------------------------------------------------------------

@configclass
class BNCSmallFemaleCfg(FixedAssetCfg):
    usd_path: str = f"{_BNC_SMALL_DIR}/Adi_BNC_Simulation_Small_Female.usd"
    height: float = 0.025
    base_height: float = 0.0
    mass: float = 0.05
    friction: float = 0.75


@configclass
class BNCSmallMaleCfg(HeldAssetCfg):
    usd_path: str = f"{_BNC_SMALL_DIR}/Adi_BNC_Simulation_Small_Male.usd"
    diameter: float = 0.020
    height: float = 0.0717
    base_height: float = 0.036235
    mass: float = 0.01
    friction: float = 0.75


@configclass
class BNCSmallInsertion(AssemblyTask):
    name = "bnc_insert"
    fixed_asset_cfg = BNCSmallFemaleCfg()
    held_asset_cfg = BNCSmallMaleCfg()
    asset_size = 44.0
    duration_s = 30.0

    assembly_id = ""
    assembly_dir = ""
    plug_grasp_json = ""
    disassembly_dist_json = ""
    disassembly_path_json = ""

    simple_grasp_pos_local: list = [0.0, 0.0, 0.0717]
    simple_grasp_quat_local: list = [0.0, 0.0, 0.0, 1.0]
    default_disassembly_dist: float = 0.060
    gripper_open_width_override: float = 0.016

    sdf_rwd_scale: float = 1.0
    imitation_rwd_scale: float = 0.0
    if_sbc: bool = False

    hand_init_pos: list = [0.00, 0.00, 0.07]
    hand_init_pos_noise: list = [0.0, 0.0, 0.0]
    hand_init_orn: list = [3.1416, 0.0, 0.0]
    hand_init_orn_noise: list = [0.0, 0.0, 0.0]
    hand_width_max: float = 0.080

    fixed_asset_init_pos_noise: list = [0.05, 0.05, 0.0]
    fixed_asset_init_orn_deg: float = 0.0
    fixed_asset_init_orn_range_deg: float = 360.0
    fixed_asset_z_offset: float = 0.0

    held_asset_init_pos_noise: list = [0.002, 0.002, 0.002]
    held_asset_pos_noise: list = [0.0, 0.0, 0.0]
    held_asset_rot_init: float = 0.0
    held_asset_rot_offset: list = [0.0, 0.0, 0.0]
    held_asset_pos_offset: list = [0.0, 0.0, 0.0]

    init_mode: str = "far"
    near_init_prob: float = 0.5
    hand_init_x_range: list = [-0.06, 0.06]
    hand_init_y_range: list = [-0.06, 0.06]
    hand_init_z_range: list = [0.07, 0.10]
    hand_init_yaw_noise_deg: float = 20.0
    hand_init_pitch_noise_deg: float = 0.0

    num_reset_kp: int = 4
    num_success_kp: int = 10
    rot_weight: float = 0.05
    action_penalty_ee_scale: float = 0.0

    keypoint_coef_baseline: list = [5, 4]
    keypoint_coef_coarse: list = [50, 2]
    keypoint_coef_fine: list = [100, 0]
    success_threshold: float = -0.3
    # Unused by Automate reward (no Forge engage term); kept for cfg parity / external tools.
    engage_threshold: float = 2.0
    close_error_thresh: float = 0.015

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
            pos=(0.0, 0.4, 0.1),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={},
            joint_vel={},
        ),
        actuators={},
    )
