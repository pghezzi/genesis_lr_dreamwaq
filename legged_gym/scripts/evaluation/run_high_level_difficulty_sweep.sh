#!/usr/bin/env bash
set -euo pipefail

: "${POLICY_JIT:?Set POLICY_JIT to the swapped-policy JIT checkpoint}"
: "${CLASSIFIER_DIR:?Set CLASSIFIER_DIR to the saved classifier run}"

DIFFICULTIES=${DIFFICULTIES:-"0.0 0.25 0.5 0.75 1.0"}
SELECTOR_MODES=${SELECTOR_MODES:-"oracle instantaneous bayes baseline"}
CLASSIFIER_APPROACH=${CLASSIFIER_APPROACH:-auto}
BAYES_FILTER_APPROACH=${BAYES_FILTER_APPROACH:-paired}
BAYES_FILTER_PATH=${BAYES_FILTER_PATH:-}
EPISODES_PER_TRACK=${EPISODES_PER_TRACK:-1}
SEED=${SEED:-42}
GPU=${GPU:-cuda:0}
TASK=${TASK:-go2_depth_waq_lora}
OUTPUT_ROOT=${OUTPUT_ROOT:-exp_logs/high_level_sweep}

for selector in ${SELECTOR_MODES}; do
  for difficulty in ${DIFFICULTIES}; do
    output_dir="${OUTPUT_ROOT}/${selector}_${CLASSIFIER_APPROACH}_${BAYES_FILTER_APPROACH}_d${difficulty}"
    command=(python -m legged_gym.scripts.evaluation.high_level_evaluation
      --task "${TASK}" --gpu "${GPU}" --headless --seed "${SEED}"
      --selector_mode "${selector}" --difficulty "${difficulty}"
      --classifier_approach "${CLASSIFIER_APPROACH}" --classifier_dir "${CLASSIFIER_DIR}"
      --bayes_filter_approach "${BAYES_FILTER_APPROACH}"
      --policy_jit "${POLICY_JIT}" --episodes_per_track "${EPISODES_PER_TRACK}"
      --out_dir "${output_dir}")
    if [[ -n "${BAYES_FILTER_PATH}" ]]; then
      command+=(--bayes_filter_path "${BAYES_FILTER_PATH}")
    fi
    "${command[@]}"
  done
done
