# Robot Environments

**Generated:** 2025-04-03
**Path:** legged_gym/envs/

## OVERVIEW

Robot environment implementations for legged locomotion. 24+ tasks across 5 robot types. Inheritance hierarchy: BaseTask → LeggedRobot → [specialized] → [robot-specific].

## STRUCTURE

```
legged_gym/envs/
├── base/              # Base classes and specialized variants
│   ├── legged_robot.py         # Base robot env
│   ├── legged_robot_ts.py      # Teacher-Student
│   ├── legged_robot_ee.py      # Explicit Estimator
│   ├── legged_robot_cts.py     # Concurrent TS
│   ├── legged_robot_amp.py     # Adversarial Motion Priors
│   ├── legged_robot_dreamwaq.py
│   ├── legged_robot_nav.py     # Navigation
│   └── legged_robot_ts_depth.py
├── go2/               # Unitree Go2 (8 variants)
├── g1/                # Unitree G1 (3 variants)
├── k1/                # Booster K1 (4 variants)
├── tron1pf/           # TRON1 PF (2 variants)
└── tron1sf/           # TRON1 SF (1 variant)
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Base class | `base/legged_robot.py` | All robots inherit from this |
| Base config | `base/legged_robot_config.py` | LeggedRobotCfg, LeggedRobotCfgPPO |
| Add new robot | Extend LeggedRobot, create config, register | See existing robot patterns |
| Observation logic | `compute_observations()` | Override in each robot class |
| Reward functions | `_reward_*` methods | Called automatically via name |
| Task registry | `__init__.py` | Register with task_registry.register() |
| Specialized bases | `base/legged_robot_*.py` | TS, EE, CTS, AMP, etc. |

## CONVENTIONS

**Robot Class Pattern**:
```python
class GO2(LeggedRobot):  # or LeggedRobotTS, LeggedRobotEE, etc.
    def compute_observations(self):
        # Override to customize obs
        
    def _reward_tracking_lin_vel(self):
        # Reward functions auto-called
```

**Config Pattern**:
```python
class GO2Cfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env): ...
    class rewards(LeggedRobotCfg.rewards):
        class scales: ...
```

**Registration**:
```python
task_registry.register("go2", GO2, GO2Cfg(), GO2CfgPPO())
```

## ANTI-PATTERNS

1. **"[NOTE]: Must be adapted"**: Comments indicating observation-dependent code. Modifying `obs_buf` requires updating ALL `_reward_*` methods that depend on those observations.

2. **IsaacGym Reset Bug**: After `reset()`, call `simulator.forward()` before reading rigid body states (see `g1_deepmimic.py:73`).

3. **Runner Mismatch**: Each robot variant needs matching runner in `rsl_rl/`. TS → TS runner, EE → EE runner, etc.

4. **Reward Scale Serialization**: Changes to reward scales in config may break checkpoint compatibility.

## CODE MAP

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| LeggedRobot | Class | `base/legged_robot.py` | Base environment |
| LeggedRobotTS | Class | `base/legged_robot_ts.py` | Teacher-Student variant |
| LeggedRobotEE | Class | `base/legged_robot_ee.py` | Explicit Estimator variant |
| LeggedRobotCTS | Class | `base/legged_robot_cts.py` | Concurrent TS variant |
| LeggedRobotAMP | Class | `base/legged_robot_amp.py` | AMP variant |
| BaseTask | Class | `base/base_task.py` | Abstract interface |
| LeggedRobotCfg | Class | `base/legged_robot_config.py` | Base config class |

## VARIANT MATRIX

| Robot | Base | Variants |
|-------|------|----------|
| go2 | LeggedRobot | basic, wtw, ts, ee, cts, dreamwaq, cat, nav |
| g1 | LeggedRobot | basic, deepmimic, motion_vis |
| k1 | LeggedRobot | basic, amp, cts_amp, deepmimic, motion_vis |
| tron1pf | LeggedRobot | basic, ee |
| tron1sf | LeggedRobot | basic |

## METHOD-SPECIFIC PATTERNS

### Teacher-Student (go2_ts)
- **Files**: `actor_critic_ts.py`, `ppo_ts.py`, `legged_robot_ts.py`
- **Key Components**: Privilege encoder + History encoder
- **Privileged Info**: Friction, mass, CoM bias, pushes, PD scales
- **Observation History**: Stack of last N observations (default: 20)
- **Training**: Concurrent RL + supervised encoder learning
- **Command**: `python -m legged_gym.scripts.train --task=go2_ts --headless`

### Explicit Estimator (go2_ee)
- **Files**: `actor_critic_ee.py`, `ppo_ee.py`
- **Key Components**: Estimator network predicts explicit values
- **Predictions**: Base linear velocity, foot contact, foot height
- **Usage**: Real-world deployment where velocity estimation needed
- **Command**: `python -m legged_gym.scripts.train --task=go2_ee --headless`

### Walk These Ways (go2_wtw)
- **Files**: `go2_wtw.py`, `go2_wtw_config.py`
- **Behavior Params**: Gait period, base height, foot clearance, pitch, gait type
- **Observation**: Clock input (4 dims) + theta (phase offsets)
- **Reward**: Periodic gait reward using von Mises distribution
- **Curriculum**: Behavior parameter range widens with performance
- **Command**: `python -m legged_gym.scripts.train --task=go2_wtw --headless`

### DeepMimic (g1_deepmimic)
- **Files**: `g1_deepmimic.py`, `motion_loader.py`
- **Motion Data**: Process reference motions first
- **Processing**: `python -m legged_gym.scripts.process_reference_motion --task=g1_motion_vis --motion_file=motion.pkl`
- **Training**: Uses reference motion observations + tracking rewards
- **Command**: `python -m legged_gym.scripts.train --task=g1_deepmimic --headless`

### AMP (k1_amp)
- **Files**: `actor_critic_amp.py`, `ppo_amp.py`, `amp_discriminator.py`
- **Key Components**: Policy + Discriminator (adversarial training)
- **Motion Data**: Reference motions for style
- **Training**: Adversarial motion priors for stylized control
- **Command**: `python -m legged_gym.scripts.train --task=k1_amp --headless`
