# Terrain classifier training

`train_terrain_classifiers.py` is the recommended entry point for training,
hyperparameter search, structural-test evaluation, Bayesian-filter fitting, and
ordered-test evaluation. It delegates to the existing approach-specific scripts
and places each run in a separate output directory.

## Expected dataset layout

The simplest layout is:

```text
dataset_root/
├── structural/
│   ├── train.pt
│   ├── val.pt
│   ├── calibration.pt
│   └── test.pt
└── bayesian/
    ├── train.pt          # optional, preferred for transition fitting
    ├── calibration.pt    # optional fallback for transition fitting
    ├── val.pt
    └── test.pt
```

The structural files train and evaluate the instantaneous classifier. Bayesian
files contain ordered sequences. If Bayesian `train.pt` or `calibration.pt` is not
available, transition fitting reuses `val.pt` and emits an optimism warning.

Alternate subfolder names `classifier/`, `bayes/`, and `sequences/` are detected.
For other layouts, pass `--classifier-data` and `--bayesian-data` explicitly.

## Training

Run all four approaches:

```bash
python -m legged_gym.scripts.depth_data_pipeline.train_terrain_classifiers \
  --dataset /path/to/dataset_root \
  --approach all
```

Run selected approaches and choose an output directory:

```bash
python -m legged_gym.scripts.depth_data_pipeline.train_terrain_classifiers \
  --classifier-data /data/terrain/structural \
  --bayesian-data /data/terrain/bayesian \
  --output /results/terrain_suite \
  --approach rbf_svm feature_nn \
  --batch-size 128
```

Approach names are:

- `rbf_prototype`
- `rbf_svm`
- `feature_nn`
- `raw_depth_nn`
- `all`

Batch/chunk processing is enabled by default. `--batch-size` controls feature
extraction, classifier-score inference, NN batches, SVM optimization batches, and
Prototype DataLoaders. To process each split as a single batch, use:

```bash
python -m legged_gym.scripts.depth_data_pipeline.train_terrain_classifiers \
  --dataset /path/to/dataset_root \
  --approach rbf_svm \
  --no-batch-processing
```

Full-split processing can require substantially more RAM and VRAM. Bayesian search
still retains only compact CPU class scores and one temperature's probabilities.

If `--output` is omitted, a timestamped suite directory is created under
`depth_waq_selector/full_models/`. If `--dataset` is omitted, the script checks
`$TERRAIN_CLASSIFIER_DATASET`, then `depth_waq_selector/processed_data/`.

Each approach directory contains its classifier/filter artifacts, searches,
parameters, metrics, and standardized `results.json`. The suite root contains
`suite_manifest.json` and, by default, comparison CSV/JSON files. Use
`--skip-comparison` to omit automatic comparison or `--continue-on-error` to run
remaining approaches after one fails.

## Comparing saved results

Comparison never reruns training. Compare a completed suite:

```bash
python -m legged_gym.scripts.depth_data_pipeline.compare_terrain_classifier_results \
  /results/terrain_suite \
  --output /results/terrain_suite/comparison
```

Or provide individual run directories/files:

```bash
python -m legged_gym.scripts.depth_data_pipeline.compare_terrain_classifier_results \
  /results/prototype /results/svm /results/feature_nn /results/raw_nn \
  --output terrain_comparison
```

The command prints separate instantaneous and Bayesian rankings and writes:

- `<output>_instantaneous.csv`
- `<output>_bayesian.csv`
- `<output>.json`

Bayesian output includes the filtering accuracy delta on the same ordered test
sequences, selected temperature, stay probability, transition alpha/matrix,
transition-delay statistics, and false-transition rate.
