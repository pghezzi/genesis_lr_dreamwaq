"""Isaac Lab actuator config for Booster K1."""
from isaaclab.actuators import DCMotorCfg

BOOSTER_ARMATURE_6416 = 0.095625
BOOSTER_ARMATURE_4310 = 0.0282528
BOOSTER_ARMATURE_6408 = 0.0478125
BOOSTER_ARMATURE_4315 = 0.0339552
BOOSTER_ARMATURE_8112 = 0.0523908
BOOSTER_ARMATURE_8116 = 0.0636012
BOOSTER_ARMATURE_ROB_14 = 0.001

# Effort/velocity from k1_22dof.urdf; stiffness/damping applied in _compute_torques
K1_ACTUATOR_CFG = {
    # Leg joints split by armature value
    "hip_pitch": DCMotorCfg(
        joint_names_expr=[".*_Hip_Pitch"],
        effort_limit=30.0,
        effort_limit_sim=30.0,
        saturation_effort=30.0,
        velocity_limit=7.1,
        velocity_limit_sim=7.1,
        stiffness=0.0,
        damping=0.0,
        armature=BOOSTER_ARMATURE_6408,
    ),
    "hip_roll": DCMotorCfg(
        joint_names_expr=[".*_Hip_Roll"],
        effort_limit=35.0,
        effort_limit_sim=35.0,
        saturation_effort=35.0,
        velocity_limit=12.9,
        velocity_limit_sim=12.9,
        stiffness=0.0,
        damping=0.0,
        armature=BOOSTER_ARMATURE_4315,
    ),
    "hip_yaw": DCMotorCfg(
        joint_names_expr=[".*_Hip_Yaw"],
        effort_limit=20.0,
        effort_limit_sim=20.0,
        saturation_effort=20.0,
        velocity_limit=18.1,
        velocity_limit_sim=18.1,
        stiffness=0.0,
        damping=0.0,
        armature=BOOSTER_ARMATURE_4310,
    ),
    "knee_pitch": DCMotorCfg(
        joint_names_expr=[".*_Knee_Pitch"],
        effort_limit=40.0,
        effort_limit_sim=40.0,
        saturation_effort=40.0,
        velocity_limit=12.5,
        velocity_limit_sim=12.5,
        stiffness=0.0,
        damping=0.0,
        armature=BOOSTER_ARMATURE_6416,
    ),
    "feet": DCMotorCfg(
        joint_names_expr=[".*_Ankle_Pitch", ".*_Ankle_Roll"],
        effort_limit=20.0,
        effort_limit_sim=20.0,
        saturation_effort=20.0,
        velocity_limit=18.1,
        velocity_limit_sim=18.1,
        stiffness=0.0,
        damping=0.0,
        armature=2.0 * BOOSTER_ARMATURE_4310,
    ),
    "arms": DCMotorCfg(
        joint_names_expr=[
            ".*_Shoulder_Pitch",
            ".*_Shoulder_Roll",
            ".*_Elbow_Pitch",
            ".*_Elbow_Yaw",
        ],
        effort_limit={
            ".*_Shoulder_Pitch": 14.0,
            ".*_Shoulder_Roll": 14.0,
            ".*_Elbow_Pitch": 14.0,
            ".*_Elbow_Yaw": 14.0,
        },
        effort_limit_sim={
            ".*_Shoulder_Pitch": 14.0,
            ".*_Shoulder_Roll": 14.0,
            ".*_Elbow_Pitch": 14.0,
            ".*_Elbow_Yaw": 14.0,
        },
        saturation_effort=14.0,
        velocity_limit={
            ".*_Shoulder_Pitch": 18.0,
            ".*_Shoulder_Roll": 18.0,
            ".*_Elbow_Pitch": 18.0,
            ".*_Elbow_Yaw": 18.0,
        },
        velocity_limit_sim={
            ".*_Shoulder_Pitch": 18.0,
            ".*_Shoulder_Roll": 18.0,
            ".*_Elbow_Pitch": 18.0,
            ".*_Elbow_Yaw": 18.0,
        },
        stiffness=0.0,
        damping=0.0,
        armature=BOOSTER_ARMATURE_ROB_14,
    ),
    "head": DCMotorCfg(
        joint_names_expr=[".*Head.*"],
        effort_limit=6.0,
        effort_limit_sim=6.0,
        saturation_effort=6.0,
        velocity_limit=20.0,
        velocity_limit_sim=20.0,
        stiffness=0.0,
        damping=0.0,
        armature=0.001,
    ),
}