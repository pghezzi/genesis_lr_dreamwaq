# RSL-RL MODULE

**Generated:** 2025-04-03

## OVERVIEW

RL algorithms module with 8 PPO variants for legged robot locomotion.
Specialized implementations for sim-to-real transfer, terrain awareness, and motion priors.

## STRUCTURE

```
rsl_rl/
├── algorithms/     # PPO implementations (8 variants)
├── modules/        # Actor-critic networks per algorithm
├── runners/        # Training orchestration + registry
├── storage/        # Rollout buffers per algorithm
├── env/            # VecEnv abstract interface
└── utils/          # Runner registry, symmetry utils
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Base PPO | `algorithms/ppo.py` | Standard proximal policy optimization |
| Base runner | `runners/on_policy_runner.py` | Generic training loop |
| Actor-critic | `modules/actor_critic.py` | Base policy network |
| Base storage | `storage/rollout_storage.py` | Standard rollout buffer |
| Add new method | Create alg + module + runner + storage | Follow existing pattern |
| Runner selection | `utils/runner_registry.py` | Dynamic runner instantiation |

## ALGORITHM VARIANTS

| Variant | File | Key Feature |
|---------|------|-------------|
| PPO | `algorithms/ppo.py` | Base algorithm |
| PPO_TS | `algorithms/ppo_ts.py` | Teacher-Student for sim-to-real |
| PPO_EE | `algorithms/ppo_ee.py` | Explicit state estimator |
| PPO_CTS | `algorithms/ppo_cts.py` | Concurrent Teacher-Student |
| PPO_DreamWaQ | `algorithms/ppo_dreamwaq.py` | Implicit terrain imagination |
| PPO_TSDepth | `algorithms/ppo_ts_depth.py` | TS with depth encoder |
| PPO_AMP | `algorithms/ppo_amp.py` | Adversarial Motion Priors |
| PPO_CTS_AMP | `algorithms/ppo_cts_amp.py` | Combined CTS + AMP |

## CODE MAP

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| PPO | Class | `algorithms/ppo.py` | Base RL algorithm |
| BaseAlgorithm | Class | `algorithms/base_algorithm.py` | Algorithm interface |
| ActorCritic | Class | `modules/actor_critic.py` | Policy network |
| OnPolicyRunner | Class | `runners/on_policy_runner.py` | Training orchestration |
| RolloutStorage | Class | `storage/rollout_storage.py` | Experience buffer |
| VecEnv | ABC | `env/vec_env.py` | Environment interface |
| runner_registry | Instance | `utils/runner_registry.py` | Runner factory |

## CONVENTIONS

**Algorithm Pattern**: Extend `BaseAlgorithm`, implement `init_storage()`, `update()`, `act()`.

**Runner Registration**: Add to `runners/__init__.py`:
```python
from .my_runner import MyRunner
runner_registry.register("MyRunner", MyRunner)
```

**Component Matching**: Algorithm + Module + Storage + Runner must be compatible.
Example: `PPO_TS` uses `ActorCriticTS`, `RolloutStorageTS`, `TSRunner`.

**Config Selection**: Runner class specified in env config's `runner_class_name` field.

## TRAINING FLOW

1. **Initialization**: Runner creates actor_critic, algorithm, storage
2. **Rollout Collection**: `learn()` calls `env.step()` for N steps, stores transitions
3. **Update**: `algorithm.update()` computes PPO loss, runs backprop
4. **Logging**: TensorBoard + optional WandB, model checkpointing

## ANTI-PATTERNS

1. **Storage Mismatch**: Using base `RolloutStorage` with `PPO_TS` (needs teacher/student buffers)
2. **Missing Registration**: New runner not added to `runners/__init__.py`
3. **Device Mismatch**: Tensors must all be on same device as runner
