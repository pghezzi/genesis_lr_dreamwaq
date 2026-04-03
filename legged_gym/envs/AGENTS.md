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
