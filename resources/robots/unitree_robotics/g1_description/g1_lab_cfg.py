"""Isaac Lab actuator config for Unitree G1 12-DOF legs."""
from isaaclab.actuators import DCMotorCfg

# Effort/velocity from g1_12dof.urdf; stiffness/damping applied in _compute_torques
G1_12DOF_ACTUATOR_CFG = {
    "hip_pitch": DCMotorCfg(
        joint_names_expr=[".*_hip_pitch_joint"],
        effort_limit_sim=88.0,
        saturation_effort=88.0,
        velocity_limit_sim=32.0,
        stiffness=0.0,
        damping=0.0,
        armature=0.01,
    ),
    "hip_roll": DCMotorCfg(
        joint_names_expr=[".*_hip_roll_joint"],
        effort_limit_sim=139.0,
        saturation_effort=139.0,
        velocity_limit_sim=20.0,
        stiffness=0.0,
        damping=0.0,
        armature=0.01,
    ),
    "hip_yaw": DCMotorCfg(
        joint_names_expr=[".*_hip_yaw_joint"],
        effort_limit_sim=88.0,
        saturation_effort=88.0,
        velocity_limit_sim=32.0,
        stiffness=0.0,
        damping=0.0,
        armature=0.01,
    ),
    "knee": DCMotorCfg(
        joint_names_expr=[".*_knee_joint"],
        effort_limit_sim=139.0,
        saturation_effort=139.0,
        velocity_limit_sim=20.0,
        stiffness=0.0,
        damping=0.0,
        armature=0.01,
    ),
    "ankle": DCMotorCfg(
        joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
        effort_limit_sim=50.0,
        saturation_effort=50.0,
        velocity_limit_sim=37.0,
        stiffness=0.0,
        damping=0.0,
        armature=0.01,
    ),
}

# Effort/velocity from g1_29dof.urdf; stiffness/damping applied in _compute_torques
G1_29DOF_ACTUATOR_CFG = {
    "N7520-14.3": DCMotorCfg(
        joint_names_expr=[".*_hip_pitch_.*", ".*_hip_yaw_.*", "waist_yaw_joint"],
        effort_limit_sim=88.0,
        saturation_effort=88.0,
        velocity_limit_sim=32.0,
        stiffness=0.0,
        damping=0.0,
        armature=0.01,
    ),
    "N7520-22.5": DCMotorCfg(
        joint_names_expr=[".*_hip_roll_.*", ".*_knee_.*"],
        effort_limit_sim=139.0,
        saturation_effort=139.0,
        velocity_limit_sim=20.0,
        stiffness=0.0,
        damping=0.0,
        armature=0.01,
    ),
    "N5020-16": DCMotorCfg(
        joint_names_expr=[
                ".*_shoulder_.*",
                ".*_elbow_.*",
                ".*_wrist_roll.*",
                ".*_ankle_.*",
                "waist_roll_joint",
                "waist_pitch_joint",
        ],
        effort_limit_sim=25.0,
        saturation_effort=25.0,
        velocity_limit_sim=37.0,
        stiffness=0.0,
        damping=0.0,
        armature=0.01,
    ),
    "W4010-25": DCMotorCfg(
        joint_names_expr=[".*_wrist_pitch.*", ".*_wrist_yaw.*"],
        effort_limit_sim=5.0,
        saturation_effort=5.0,
        velocity_limit_sim=22.0,
        stiffness=0.0,
        damping=0.0,
        armature=0.01,
    ),
}