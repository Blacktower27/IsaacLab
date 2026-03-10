# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os as _os

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.direct.factory.factory_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG, CtrlCfg, FactoryEnvCfg, ObsRandCfg

_KUKA_URDF = _os.path.normpath(
    _os.path.join(
        _os.path.dirname(_os.path.abspath(__file__)),
        "../../../../isaaclab_assets/isaaclab_assets/custom_assets/robots"
        "/lbr_description/urdf/kuka_blue/kuka_blue.urdf",
    )
)

from .forge_events import randomize_dead_zone
from .forge_tasks_cfg import ForgeBoxLidInsert, ForgeGearMesh, ForgeNutThread, ForgePegInsert, ForgeTask

OBS_DIM_CFG.update({"force_threshold": 1, "ft_force": 3})

STATE_DIM_CFG.update({"force_threshold": 1, "ft_force": 3})


@configclass
class ForgeCtrlCfg(CtrlCfg):
    ema_factor_range = [0.025, 0.1]
    default_task_prop_gains = [565.0, 565.0, 565.0, 28.0, 28.0, 28.0]
    task_prop_gains_noise_level = [0.41, 0.41, 0.41, 0.41, 0.41, 0.41]
    pos_threshold_noise_level = [0.25, 0.25, 0.25]
    rot_threshold_noise_level = [0.29, 0.29, 0.29]
    default_dead_zone = [5.0, 5.0, 5.0, 1.0, 1.0, 1.0]


@configclass
class ForgeObsRandCfg(ObsRandCfg):
    fingertip_pos = 0.00025
    fingertip_rot_deg = 0.1
    ft_force = 1.0


@configclass
class EventCfg:
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("held_asset"),
            "mass_distribution_params": (-0.005, 0.005),
            "operation": "add",
            "distribution": "uniform",
        },
    )

    held_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("held_asset"),
            "static_friction_range": (0.75, 0.75),
            "dynamic_friction_range": (0.75, 0.75),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )

    fixed_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("fixed_asset"),
            "static_friction_range": (0.25, 1.25),  # TODO: Set these values based on asset type.
            "dynamic_friction_range": (0.25, 0.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 128,
        },
    )

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.75, 0.75),
            "dynamic_friction_range": (0.75, 0.75),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )

    dead_zone_thresholds = EventTerm(
        func=randomize_dead_zone,
        mode="interval",
        interval_range_s=(2.0, 2.0),  # (0.25, 0.25)
    )


@configclass
class ForgeEnvCfg(FactoryEnvCfg):
    action_space: int = 7
    obs_rand: ForgeObsRandCfg = ForgeObsRandCfg()
    ctrl: ForgeCtrlCfg = ForgeCtrlCfg()
    task: ForgeTask = ForgeTask()
    events: EventCfg = EventCfg()

    ft_smoothing_factor: float = 0.25

    obs_order: list = [
        "fingertip_pos_rel_fixed",
        "fingertip_quat",
        "ee_linvel",
        "ee_angvel",
        # "ft_force",
        # "force_threshold",
    ]
    state_order: list = [
        "fingertip_pos",
        "fingertip_quat",
        "ee_linvel",
        "ee_angvel",
        "joint_pos",
        "held_pos",
        "held_pos_rel_fixed",
        "held_quat",
        "fixed_pos",
        "fixed_quat",
        "task_prop_gains",
        "ema_factor",
        # "ft_force",
        "pos_threshold",
        "rot_threshold",
        # "force_threshold",
    ]


@configclass
class ForgeTaskPegInsertCfg(ForgeEnvCfg):
    task_name = "peg_insert"
    task = ForgePegInsert()
    episode_length_s = 10.0


@configclass
class ForgeTaskGearMeshCfg(ForgeEnvCfg):
    task_name = "gear_mesh"
    task = ForgeGearMesh()
    episode_length_s = 20.0


@configclass
class ForgeTaskNutThreadCfg(ForgeEnvCfg):
    task_name = "nut_thread"
    task = ForgeNutThread()
    episode_length_s = 30.0


# ---------------------------------------------------------------------------
# [CUSTOM] Environment config for the box-lid insertion task.
#
# Registered as gym ID: Isaac-Forge-BoxLidInsert-Direct-v0
# (see forge/__init__.py).
#
# episode_length_s = 15 s matches ForgeBoxLidInsert.duration_s.  The lid
# insertion is a shorter-horizon task than nut threading (30 s) but needs
# more time than peg insertion (10 s) because the box can be randomly rotated
# up to 360° and the policy must first align the lid before descending.
# ---------------------------------------------------------------------------
@configclass
class ForgeTaskBoxLidInsertCfg(ForgeEnvCfg):
    task_name = "box_lid_insert"
    task = ForgeBoxLidInsert()
    episode_length_s = 15.0


# ---------------------------------------------------------------------------
# [CUSTOM] Kuka iiwa7 variant of the box-lid insertion task.
#
# The lid (Lid_Yellow) is embedded directly in the URDF as link_lid (fixed
# joint to link_tcp), so the arm physically feels insertion resistance.
# There is no separate held_asset in the scene.
#
# Registered as gym ID: Isaac-Forge-BoxLidInsert-Kuka-Direct-v0
# ---------------------------------------------------------------------------
@configclass
class ForgeKukaCtrlCfg(ForgeCtrlCfg):
    # Kuka body names — merge_fixed_joints=False keeps link_tcp as the gripper tip.
    fingertip_body_name: str = "link_tcp"
    left_finger_body_name: str = "link_tcp"
    right_finger_body_name: str = "link_tcp"
    force_sensor_body_name: str = "link_ee"
    # Home pose: arm roughly above the workspace.
    reset_joints: list = [0.0, 0.3, 0.0, -1.5, 0.0, 1.2, 0.0]


@configclass
class ForgeKukaEventCfg(EventCfg):
    # Disable held-asset mass randomisation — the lid is part of the robot URDF.
    object_scale_mass = None


@configclass
class ForgeKukaBoxLidInsertCfg(ForgeTaskBoxLidInsertCfg):
    task_name = "box_lid_insert"
    ctrl: ForgeKukaCtrlCfg = ForgeKukaCtrlCfg()
    events: ForgeKukaEventCfg = ForgeKukaEventCfg()

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=_KUKA_URDF,
            fix_base=True,               # Lock the base link to the world frame.
            merge_fixed_joints=False,    # Keep all fixed joints as separate bodies (preserves link_ee, link_tcp).
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                # Disable default PD gains from URDF — OSC controller outputs pure torques instead.
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None)
            ),
            activate_contact_sensors=True,  # Enable contact sensors to read contact forces.
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,               # Disable gravity on all links; the OSC controller compensates for it.
                max_depenetration_velocity=5.0,     # Max velocity (m/s) used to resolve collision penetration.
                linear_damping=0.0,                 # Linear velocity damping; 0 = no artificial damping.
                angular_damping=0.0,                # Angular velocity damping; 0 = no artificial damping.
                max_linear_velocity=1000.0,         # Velocity clamp (m/s) to prevent numerical explosion.
                max_angular_velocity=3666.0,        # Angular velocity clamp (rad/s), ~350 rpm.
                enable_gyroscopic_forces=True,      # Enable gyroscopic forces for accurate high-speed dynamics.
                solver_position_iteration_count=192, # PhysX position solver iterations; higher = more stable but slower.
                solver_velocity_iteration_count=1,  # PhysX velocity solver iterations.
                max_contact_impulse=1e32,           # Contact impulse cap; large value = effectively unlimited.
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,       # Disable self-collision detection for performance.
                solver_position_iteration_count=192, # Articulation-level position solver iterations.
                solver_velocity_iteration_count=1,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005,  # Distance (m) at which contact is detected before actual touch.
                rest_offset=0.0,       # Resting gap (m) between surfaces; 0 = surfaces can fully touch.
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            # Initial joint angles (rad) at reset; should match reset_joints in ForgeKukaCtrlCfg.
            joint_pos={
                "A1": 0.0,
                "A2": 0.3,
                "A3": 0.0,
                "A4": -1.5,
                "A5": 0.0,
                "A6": 1.2,
                "A7": 0.0,
            },
            pos=(0.0, 0.0, 0.0),        # Base position in world frame.
            rot=(1.0, 0.0, 0.0, 0.0),   # Base orientation in world frame (quaternion, no rotation).
        ),
        actuators={
            "kuka_arm": ImplicitActuatorCfg(
                joint_names_expr=["A[1-7]"],  # Apply to all 7 arm joints.
                stiffness=0.0,         # P gain; 0 = pure torque control (OSC computes torques directly).
                damping=0.0,           # D gain; 0 = no damping (OSC handles it).
                friction=0.0,          # Joint friction; 0 = ignored (can be tuned later).
                armature=0.0,          # Rotor inertia; 0 = ignored.
                effort_limit_sim=200.0,   # Torque limit (Nm) in simulation; rough estimate, varies per joint on real iiwa7.
                velocity_limit_sim=3.15,  # Joint velocity limit (rad/s), ~180 deg/s.
            ),
        },
    )
