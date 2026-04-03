# PROJECT KNOWLEDGE BASE

**Generated:** 2025-04-03
**Commit:** 0073932
**Branch:** dev

## OVERVIEW
LeggedGym-Ex is a legged robot RL framework supporting Genesis, IsaacGym, and IsaacSim. Extends legged_gym with 10+ published methods (DeepMimic, AMP, Walk These Ways, etc.).

## STRUCTURE
```
LeggedGym-Ex/
├── legged_gym/          # Core framework (envs, scripts, utils, simulator)
├── rsl_rl/              # RL algorithms (PPO variants)
├── resources/           # Robot URDFs, meshes, reference motions
└── tests/               # Test scripts
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Add new robot | `legged_gym/envs/` | Extend LeggedRobot base class |
| Add RL method | `rsl_rl/` | Add PPO variant + runner + storage |
| Train policy | `legged_gym/scripts/train.py` | Entry point for training |
| Run inference | `legged_gym/scripts/play.py` | Load and run trained policy |
| Config system | `legged_gym/envs/base/legged_robot_config.py` | Nested class configs |
| Task registry | `legged_gym/utils/task_registry.py` | Factory for envs/algs |
| Terrain gen | `legged_gym/utils/terrain.py` | Heightfield/trimesh |
| Math utils | `legged_gym/utils/math_utils.py` | Quaternion ops, etc. |

## CODE MAP

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| LeggedRobot | Class | `envs/base/legged_robot.py` | Base environment |
| BaseTask | Class | `envs/base/base_task.py` | Abstract task interface |
| TaskRegistry | Class | `utils/task_registry.py` | Env/alg factory |
| OnPolicyRunner | Class | `rsl_rl/runners/` | Training orchestration |
| PPO | Class | `rsl_rl/algorithms/ppo.py` | Base RL algorithm |
| Simulator | ABC | `simulator/simulator.py` | Simulator interface |

## CONVENTIONS

**Configuration Pattern**: Nested classes inheriting from `LeggedRobotCfg`/`LeggedRobotCfgPPO`. Example: `class GO2Cfg(LeggedRobotCfg)` with nested `class env`, `class rewards`, etc.

**Task Registration**: Register in `legged_gym/envs/__init__.py`: `task_registry.register("go2", GO2, GO2Cfg, GO2CfgPPO)`

**Simulator Selection**: Set `SIMULATOR` env var: `export SIMULATOR=genesis` or `isaaclab`

**Naming**: Task names follow `<robot>_<variant>` (e.g., `go2_ts`, `k1_amp`)

## ANTI-PATTERNS (THIS PROJECT)

1. **IsaacGym Reset Bug**: After `reset()`, call `simulator.forward()` once before reading rigid body states (see `g1_deepmimic.py:73`)
2. **Observation Changes**: Modifying `obs_buf` requires updating ALL `_reward_*` methods (see "[NOTE]: Must be adapted" comments)
3. **IsaacLab Tensor Device**: Domain randomization tensors must be on CPU for IsaacLab (`set_material_properties`, `set_masses`, `set_coms`)
4. **Terrain Constraints**: Cannot use `curriculum=True` with `selected=True` simultaneously
5. **Genesis XML**: Must provide XML file path when using Genesis simulator
6. **Heightfield Limitation**: Heightfield terrain not implemented for IsaacLabSimulator

## UNIQUE STYLES

- **Type Aliases**: `ObsBuf = Tensor`, `Action = Tensor`, `Reward = Tensor` in base classes
- **Config Assertions**: Extensive validation in `LeggedRobot.__init__()` to catch config errors early
- **Debug Flags**: `cfg.env.debug*` flags for visualization (height points, depth images, etc.)
- **Paper Caveats**: Comments like "code above can't result in same reward curve as paper" indicate known deviations

## COMMANDS

```bash
# Training
python -m legged_gym.scripts.train --task go2_ts --headless
python -m legged_gym.scripts.train --task go2 --num_envs 1000

# Inference
python -m legged_gym.scripts.play --task go2_ts --resume
python -m legged_gym.scripts.play --task go2 --use_joystick --joystick_type xbox

# Motion processing (DeepMimic/AMP)
python -m legged_gym.scripts.process_reference_motion --task g1_deepmimic

# Testing
python tests/test_all_tasks.py
python tests/test_all_tasks.py --tasks go2 g1 --iterations 3

# List all tasks
python tests/test_all_tasks.py --list
```

## NOTES

- Multi-simulator support: Same code runs on Genesis/IsaacGym/IsaacLab via `SIMULATOR` env var
- 24+ registered tasks across 5 robot types (GO2, G1, K1, TRON1PF, TRON1SF)
- 8 PPO algorithm variants (TS, EE, CTS, AMP, DreamWaQ, etc.)
- Reference: External docs at https://genesis-lr-doc.readthedocs.io/en/latest/
