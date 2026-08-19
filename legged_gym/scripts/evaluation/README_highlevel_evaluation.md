# High-level terrain-selection evaluation

`high_level_evaluation.py` evaluates three connected capabilities in closed loop:

1. instantaneous terrain classification;
2. temporally filtered classification and skill selection;
3. navigation through sequential mixed-terrain tracks using pre-swapped low-level policies.

A run evaluates one policy JIT, classifier, Bayes-filter checkpoint, seed, difficulty, and selector mode. It does not aggregate seeds or checkpoints.

## Track experiment

The evaluator creates five terrain rows by ten columns. Each column is an independent forward track containing exactly one of each terrain type:

- rough/random-uniform;
- GAP;
- PIT;
- upward stairs;
- downward stairs.

The five cells are independently permuted for each column using `--seed`. A fixed seed produces the same track sequences across difficulty values, selector modes, classifiers, and Bayes filters. The generated sequence and concrete parameters for every cell are printed and saved.

Difficulty is set with `--difficulty` in `[0,1]`:

| Terrain | Parameter |
|---|---|
| GAP | `0.30 + 0.70 × difficulty` m |
| PIT | `0.25 + 0.25 × difficulty` m |
| Stairs | `0.10 + 0.30 × difficulty` m magnitude |
| Rough | Existing depth-WAQ random-uniform mapping |

The default is ten environments, one per column. Larger values must be balanced multiples of ten. Every environment remains assigned to its column and resets at row zero. Leaving that column laterally is a failure.

## Selector modes

Choose the policy-selection source with `--selector_mode`:

| Mode | Selected skill |
|---|---|
| `oracle` | Environment ground-truth terrain label |
| `instantaneous` | Instantaneous classifier argmax |
| `bayes` | Per-environment Bayes-filter posterior argmax |
| `baseline` | Always use the rough policy |

All modes still run and score both instantaneous and Bayes-filtered predictions. Oracle labels come only from the environment terrain-label grid through the evaluator's existing look-ahead lookup. Configure the look-ahead fraction with `--look_ahead_frac`.

Raw terrain labels are canonicalized only for scoring and skill selection: rough variants map to `rough`, both stair directions to `stairs`, PIT variants to `pit`, and GAP to `gap`.

Each environment owns an independent Bayes belief. Beliefs persist between classification ticks and reset only when that environment completes an episode. Immediately after reset, non-oracle modes use the baseline skill until a valid nonzero depth frame is available.

## Classifiers and Bayes filters

`--classifier_approach` supports:

- `auto`;
- `rbf_prototype`;
- `rbf_svm`;
- `feature_nn`;
- `raw_depth_nn`.

Set `--classifier_dir` to a saved classifier run containing the applicable artifacts: `classifier.pt`, `results.json`, `extractor.pt`, `standardizer.pt`, and/or `nn_model_args.pt`. `auto` inspects `results.json` and falls back to the saved artifact structure.

Engineered-feature approaches use the saved Sobel/geometric extractor and standardizer. The raw-depth NN consumes the processed depth tensor directly.

Bayes-filter loading supports:

- `--bayes_filter_approach paired`: load `bayes_filter.pt` from `classifier_dir`;
- `--bayes_filter_approach checkpoint`: load `--bayes_filter_path`.

Classifier class ordering must exactly match the filter checkpoint's label ordering.

## Commands and policy dispatch

The policy JIT must provide `swap(lora_id)` and the depth-WAQ policy call signature. The evaluator preloads one swapped copy for every skill and dispatches each environment independently.

Default forward commands are:

| Selected skill | Speed |
|---|---:|
| Rough/baseline | 0.8 m/s |
| GAP | 1.5 m/s |
| PIT | 1.2 m/s |
| Stairs | 1.2 m/s |

Use `--fixed_forward_command` to apply one speed to every skill. Lateral velocity, yaw rate, heading, command curricula, and command resampling are disabled or continually overridden for straight traversal.

## Episodes and metrics

Each column must contribute exactly `--episodes_per_track` completed episodes. Evaluation stops when all ten quotas are complete; `--num_steps` is only a safety cap.

An episode succeeds when it either reaches the end of the final row, accounting for `--finish_margin`, or reaches the environment's normal timeout. Falls, other contact terminations, obstacle failures, and lateral track exits are failures. Course completion takes precedence when it coincides with another termination.

Terminal position, timeout state, episode length, and accumulated classifier counts are captured before the environment's automatic reset. Forward distance is maximum nonnegative `+x` progress, capped at the track finish distance.

Headline results are:

- episodic success rate;
- mean episodic forward distance;
- instantaneous classification accuracy;
- Bayes-filtered classification accuracy.

Detailed output includes per-column episode summaries, per-class support and accuracy, full confusion matrices, selected-skill accuracy, switch count, terrain-transition detection delay when available, every valid classification tick, and per-episode classifier counts.

Each run writes a complete JSON result and a per-episode CSV under `--out_dir`.

## Single-run usage

Run from the repository root after selecting an installed simulator:

```bash
SIMULATOR=genesis python -m legged_gym.scripts.evaluation.high_level_evaluation \
  --task go2_depth_waq_lora \
  --selector_mode bayes \
  --classifier_approach auto \
  --classifier_dir /path/to/classifier_run \
  --bayes_filter_approach paired \
  --policy_jit /path/to/swap_policy.jit.pt \
  --difficulty 0.5 \
  --episodes_per_track 5 \
  --seed 42 \
  --gpu cuda:0 \
  --headless \
  --out_dir results/high_level_bayes
```

Use an independent filter checkpoint and instantaneous policy selection:

```bash
SIMULATOR=genesis python -m legged_gym.scripts.evaluation.high_level_evaluation \
  --selector_mode instantaneous \
  --classifier_approach rbf_svm \
  --classifier_dir /path/to/rbf_svm_run \
  --bayes_filter_approach checkpoint \
  --bayes_filter_path /path/to/bayes_filter.pt \
  --policy_jit /path/to/swap_policy.jit.pt \
  --difficulty 0.75 --headless
```

For a bounded smoke run, use ten environments and one episode per track:

```bash
SIMULATOR=genesis python -m legged_gym.scripts.evaluation.high_level_evaluation \
  --selector_mode oracle --classifier_dir /path/to/classifier_run \
  --policy_jit /path/to/swap_policy.jit.pt \
  --num_envs 10 --episodes_per_track 1 --num_steps 500 --headless
```

## Difficulty sweep

`run_high_level_difficulty_sweep.sh` launches a separate process for every selector/difficulty pair while retaining the same seed and track permutations:

```bash
SIMULATOR=genesis \
POLICY_JIT=/path/to/swap_policy.jit.pt \
CLASSIFIER_DIR=/path/to/classifier_run \
SELECTOR_MODES="oracle instantaneous bayes baseline" \
DIFFICULTIES="0.0 0.25 0.5 0.75 1.0" \
SEED=42 \
EPISODES_PER_TRACK=5 \
GPU=cuda:0 \
OUTPUT_ROOT=results/high_level_sweep \
legged_gym/scripts/evaluation/run_high_level_difficulty_sweep.sh
```

Optional sweep variables include `CLASSIFIER_APPROACH`, `BAYES_FILTER_APPROACH`, `BAYES_FILTER_PATH`, and `TASK`.

## Evaluation randomization settings

The script-level `EVAL_DOMAIN_RANDOMIZATION_RANGES` and `EVAL_DOMAIN_RANDOMIZATION_ENABLED` dictionaries match the low-level evaluator. Edit these hardcoded dictionaries to change the evaluation perturbation distribution. Disabled randomization names are stored in the result metadata.
