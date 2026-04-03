# LEGGED_GYM CORE FRAMEWORK

Core legged robot RL framework with multi-simulator support (Genesis, IsaacGym, IsaacLab).

## STRUCTURE

| Directory | Purpose |
|-----------|---------|
| `envs/` | Robot environments extending LeggedRobot |
| `scripts/` | Training and inference entry points |
| `utils/` | Terrain generation, task registry, math utilities |
| `simulator/` | Simulator abstraction layer |

## KEY FILES

| Purpose | File |
|---------|------|
| Training entry | `scripts/train.py` |
| Inference | `scripts/play.py` |
| Base environment | `envs/base/legged_robot.py` |
| Config system | `envs/base/legged_robot_config.py` |
| Task registry | `utils/task_registry.py` |
| Terrain generation | `utils/terrain.py` |
| Math utilities | `utils/math_utils.py` |
| Simulator ABC | `simulator/simulator.py` |

## CONVENTIONS

**Config Pattern**: Nested classes with `class env`, `class rewards`, `class commands`, etc.

```python
class MyRobotCfg(LeggedRobotCfg):
    class env:
        num_observations = 48
    class rewards:
        class scales:
            tracking_lin_vel = 1.0
```

**Task Registration**: Add to `envs/__init__.py`:
```python
task_registry.register("robot_name", RobotClass, Cfg, CfgPPO)
```

**Simulator Selection**: `export SIMULATOR=genesis|isaacgym|isaaclab`

## ANTI-PATTERNS

1. **Observation Changes**: Modifying `obs_buf` requires updating ALL `_reward_*` methods
2. **IsaacGym Reset Bug**: After `reset()`, call `self.simulator.forward()` before reading rigid body states
3. **IsaacLab CPU Tensors**: Domain randomization tensors must be on CPU (`set_material_properties`, `set_masses`, `set_coms`)
4. **Genesis XML**: Must provide XML file path when using Genesis simulator
5. **Terrain Flags**: Cannot use `curriculum=True` with `selected=True` simultaneously
6. **IsaacLab Heightfield**: Heightfield terrain not implemented for IsaacLabSimulator

## PATTERNS

**Type Aliases**: `ObsBuf = Tensor`, `Action = Tensor`, `Reward = Tensor` in base classes

**Config Validation**: Extensive assertions in `LeggedRobot.__init__()` catch config errors early

**Debug Flags**: `cfg.env.debug*` enable visualization (height points, depth images, etc.)
