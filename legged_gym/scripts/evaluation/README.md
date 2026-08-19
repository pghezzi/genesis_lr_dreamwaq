# Low-level policy evaluation

`low_level_evaluation.py` evaluates one depth-WAQ low-level policy checkpoint at progressively harder terrain levels. A run uses one checkpoint and one seed; it does not aggregate checkpoints or seeds.

## Experiments

Select an experiment with `--skill`:

| Skill | Terrain | Difficulty schedule | Default forward command |
|---|---|---|---:|
| `rough` | baseline rough curriculum | existing training curriculum | 0.8 m/s |
| `leap` | gaps | gap size 0.30–1.00 m | 1.5 m/s |
| `climb` | pits | pit depth 0.25–0.60 m | 1.2 m/s |
| `stairs` | stairs branch, using its existing sign convention | step-height magnitude 0.10–0.40 m | 1.2 m/s |

Difficulty endpoints are inclusive. The evaluator compensates for terrain generation using `difficulty = row / num_rows`, so the final row reaches the requested maximum. All parallel environments share one level. Automatic terrain and command curricula are disabled.

Each environment receives a fixed straight command: zero lateral and yaw velocity, zero initial yaw randomization, and no zero-command sampling. `--forward_command` can replace the default forward speed.

At each level, completed episodes are counted across all environments until `--episodes_per_level` is reached. An episode succeeds as soon as its world-frame `+x` progress reaches `--success_distance`, which defaults to the configured length of one sub-terrain (`terrain_length`), at which point that environment is reset immediately. A natural termination or timeout before that distance is a failure. When the level quota is reached, all partial episodes are discarded and all environments restart on the next level. `--num_steps` is only a run-wide safety cap.

Terminal state is captured immediately before the environment's automatic reset, ensuring terminal progress and velocity errors are attributed to the episode that ended.

## Metrics and output

Every episode records:

- success and termination reason (`success_distance`, `termination`, or `timeout`);
- maximum forward progress, clamped to the success distance;
- episode length in simulation steps;
- mean linear tracking error, `||command_xy - base_lin_vel_xy||`;
- mean angular tracking error, `|command_yaw_rate - base_ang_vel_z|`;
- skill, terrain difficulty, seed, task, run, and checkpoint metadata.

Per-difficulty and overall summaries report episode count, success rate in percent, average distance, and the mean episode-level linear and angular tracking errors. They also indicate whether all requested episode quotas were completed before the safety cap.

A `.json` save path writes summaries and episode rows to one file. A `.csv` path writes episode rows to CSV and summaries to a sibling `_summary.json` file.

## Usage

Run from the repository root. Set the simulator before Python starts:

```bash
SIMULATOR=genesis python -m legged_gym.scripts.evaluation.low_level_evaluation \
  --skill leap \
  --task go2_depth_waq_lora \
  --load_run RUN_NAME \
  --ckpt 1000 \
  --num_envs 64 \
  --num_difficulty_levels 8 \
  --episodes_per_level 50 \
  --seed 1 \
  --headless \
  --save_path results/leap.json
```

Use the latest run and checkpoint, override forward speed, and save episode CSV data:

```bash
SIMULATOR=genesis python -m legged_gym.scripts.evaluation.low_level_evaluation \
  --skill stairs \
  --task go2_depth_waq_lora \
  --forward_command 1.0 \
  --save_path results/stairs.csv \
  --headless
```

For a small smoke evaluation:

```bash
SIMULATOR=genesis python -m legged_gym.scripts.evaluation.low_level_evaluation \
  --skill rough --num_envs 2 --num_difficulty_levels 1 \
  --episodes_per_level 2 --success_distance 0.5 --num_steps 500 --headless
```

Use `SIMULATOR=isaaclab` for an IsaacLab installation. Add `--cpu` where the selected simulator supports CPU execution. Run `--help` for all CLI options.

## Evaluation randomization settings

The script-level `EVAL_DOMAIN_RANDOMIZATION_RANGES` dictionary controls evaluation ranges independently of the training config. `EVAL_DOMAIN_RANDOMIZATION_ENABLED` controls each randomization with a boolean. The current defaults use nominal values where applicable and disable the listed randomizations. Edit these dictionaries in `low_level_evaluation.py` when evaluating a different perturbation distribution; the disabled settings are included in saved run metadata.
