"""Isaac Lab actuator config for Booster K1."""
from isaaclab.actuators import DCMotorCfg

ARMATURE_6416 = 0.095625
ARMATURE_4310 = 0.0282528
ARMATURE_6408 = 0.0478125
ARMATURE_4315 = 0.0339552
ARMATURE_8112 = 0.0523908
ARMATURE_8116 = 0.0636012
ARMATURE_ROB_14 = 0.001

# Effort/velocity from k1_22dof.urdf; stiffness/damping applied in _compute_torques
K1_ACTUATOR_CFG = {
    "legs": DCMotorCfg(
        joint_names_expr=[
                ".*_Hip_Pitch",
                ".*_Hip_Roll",
                ".*_Hip_Yaw",
                ".*_Knee_Pitch",
            ],
            effort_limit_sim={
                ".*_Hip_Pitch": 30.,
                ".*_Hip_Roll": 35.,
                ".*_Hip_Yaw": 20.,
                ".*_Knee_Pitch": 40.,
            },
            velocity_limit={
                ".*_Hip_Pitch": 7.1,
                ".*_Hip_Roll": 12.9,
                ".*_Hip_Yaw": 18.1,
                ".*_Knee_Pitch": 12.5,
            },
            stiffness=0.0,
            damping=0.0,
            armature={
                ".*_Hip_Pitch": ARMATURE_6408,
                ".*_Hip_Roll": ARMATURE_4315,
                ".*_Hip_Yaw": ARMATURE_4310,
                ".*_Knee_Pitch": ARMATURE_6416,
            },
    ),
    "feet": DCMotorCfg(
        effort_limit_sim=20.0,
        velocity_limit_sim=18.1,
        joint_names_expr=[".*_Ankle_Pitch", ".*_Ankle_Roll"],
        stiffness=0.0,
        damping=0.0,
        armature=2.0 * ARMATURE_4310,
    ),
    "arms": DCMotorCfg(
        joint_names_expr=[
                ".*_Shoulder_Pitch",
                ".*_Shoulder_Roll",
                ".*_Elbow_Pitch",
                ".*_Elbow_Yaw",
            ],
            effort_limit_sim={
                ".*_Shoulder_Pitch": 14.0,
                ".*_Shoulder_Roll": 14.0,
                ".*_Elbow_Pitch": 14.0,
                ".*_Elbow_Yaw": 14.0,
            },
            velocity_limit_sim={
                ".*_Shoulder_Pitch": 18.0,
                ".*_Shoulder_Roll": 18.0,
                ".*_Elbow_Pitch": 18.0,
                ".*_Elbow_Yaw": 18.0,
            },
            stiffness=0.0,
            damping=0.0,
            armature={
                ".*_Shoulder_Pitch": ARMATURE_ROB_14,
                ".*_Shoulder_Roll": ARMATURE_ROB_14,
                ".*_Elbow_Pitch": ARMATURE_ROB_14,
                ".*_Elbow_Yaw": ARMATURE_ROB_14,
            },
    ),
    "head": DCMotorCfg(
        joint_names_expr=[".*Head.*"],
        effort_limit_sim=6.0,
        velocity_limit_sim=20.0,
        stiffness=0.0,
        damping=0.0,
        armature=0.001,
    ),
}