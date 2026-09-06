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

- `feature_nn_deterministic`;
- `raw_depth_nn_deterministic`;
- `feature_nn_mc`;
- `raw_depth_nn_mc`.

Set `--classifier_dir` (or `--classifier_suite`) to a completed classifier-suite
root and select one of the four learned deployments. The evaluator loads the
selected model, feature preprocessing, deterministic/MC mode, MC sample count,
and uncertainty-aware temporal filter from its deployment manifest.

Engineered-feature approaches use the saved Sobel/geometric extractor and standardizer. The raw-depth NN combines the processed depth tensor with base roll, pitch, and roll/pitch/yaw angular velocities before its hidden FC layers.

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
  --classifier_approach feature_nn_mc \
  --classifier_suite /path/to/classifier_suite \
  --policy_jit /path/to/swap_policy.jit.pt \
  --difficulty 0.5 \
  --episodes_per_track 5 \
  --seed 42 \
  --gpu cuda:0 \
  --headless \
  --out_dir results/high_level_bayes
```

Use deterministic raw-depth inference and instantaneous policy selection:

```bash
SIMULATOR=genesis python -m legged_gym.scripts.evaluation.high_level_evaluation \
  --selector_mode instantaneous \
  --classifier_approach raw_depth_nn_deterministic \
  --classifier_suite /path/to/classifier_suite \
  --policy_jit /path/to/swap_policy.jit.pt \
  --difficulty 0.75 --headless
```

For a bounded smoke run, use ten environments and one episode per track:

```bash
SIMULATOR=genesis python -m legged_gym.scripts.evaluation.high_level_evaluation \
  --selector_mode oracle --classifier_approach feature_nn_deterministic \
  --classifier_suite /path/to/classifier_suite \
  --policy_jit /path/to/swap_policy.jit.pt \
  --num_envs 10 --episodes_per_track 1 --num_steps 500 --headless
```

## Difficulty sweep

`run_high_level_difficulty_sweep.sh` launches a separate process for every selector/difficulty pair while retaining the same seed and track permutations:

```bash
SIMULATOR=genesis \
POLICY_JIT=/path/to/swap_policy.jit.pt \
CLASSIFIER_DIR=/path/to/classifier_suite \
SELECTOR_MODES="oracle instantaneous bayes baseline" \
DIFFICULTIES="0.0 0.25 0.5 0.75 1.0" \
SEED=42 \
EPISODES_PER_TRACK=5 \
GPU=cuda:0 \
OUTPUT_ROOT=results/high_level_sweep \
legged_gym/scripts/evaluation/run_high_level_difficulty_sweep.sh
```

Optional sweep variables include `CLASSIFIER_APPROACH` and `TASK`.

## Evaluation randomization settings

The script-level `EVAL_DOMAIN_RANDOMIZATION_RANGES` and `EVAL_DOMAIN_RANDOMIZATION_ENABLED` dictionaries match the low-level evaluator. Edit these hardcoded dictionaries to change the evaluation perturbation distribution. Disabled randomization names are stored in the result metadata.

## Frozen paper locomotion sweep

The final no-search evaluation loads the three seeded classifiers produced by
`evaluate_paper_offline_experiments_1_2.py`, runs the six frozen routing/policy
conditions over Easy/Nominal/Hard and evaluation seeds 101/202/303/404/505, and
resumes completed conditions automatically:

```bash
SIMULATOR=genesis python -m \
  legged_gym.scripts.evaluation.run_paper_locomotion_evaluation \
  --paper_offline_dir /path/to/paper_offline_eval \
  --jit /path/to/specialist_lora_bundle.pt \
  --distilled_jit /path/to/exported_distilled_policy.pt \
  --output paper_locomotion_eval --gpu cuda:0
```

Each condition retains the existing run JSON/episode CSV. The sweep root adds
per-episode, per-run, summary, difficulty, and transition-pair CSVs; a LaTeX
main-results table; PNG/PDF figures; and a manifest containing all resolved
artifacts, terrain parameters, seeds, commands, and paired track layouts.
