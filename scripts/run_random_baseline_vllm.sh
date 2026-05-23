#!/usr/bin/env bash

set -euo pipefail

export PYTHON_BIN=${PYTHON_BIN:-.venv-vllm/bin/python}
export CONFIG_PATH=${CONFIG_PATH:-configs/random_baseline.yaml}
export CONDITION_LABEL=${CONDITION_LABEL:-"random"}
export ARMAD_EXP_TIMESTAMP=${ARMAD_EXP_TIMESTAMP:-$(TZ=Europe/London date +%Y%m%d%H%M%S)_random_baseline}
export ARMAD_SEEDS=${ARMAD_SEEDS:-"1 2 3"}
export ARMAD_PERM_SEEDS=${ARMAD_PERM_SEEDS:-"10"}

source "$(dirname "$0")/common.sh"

if [[ -n "${RANDOM_BASELINE_DATASETS:-}" ]]; then
  read -r -a DATASETS <<< "$RANDOM_BASELINE_DATASETS"
else
  DATASETS=(mmlu_pro truthful_qa gsm8k math_500)
fi

RANDOM_NUM_EXAMPLES=${RANDOM_NUM_EXAMPLES:-80}
for ds in "${DATASETS[@]}"; do
  N_EX[$ds]="${RANDOM_NUM_EXAMPLES}"
done

RUNNER_EXTRA_ARGS=(
  --parallel-examples "${PARALLEL_EXAMPLES:-1}"
  --no-progress
)

MODELS=(
  "qwen2.5-7b-vllm|qwen25_7b_vllm"
  "llama-3.1-8b-vllm|llama31_8b_vllm"
  "gemma-3-12b-vllm|gemma3_12b_vllm"
  "qwen3-30b-a3b-local-vllm|qwen3_30b_a3b_vllm"
)

echo ""
echo "========================================"
echo "RANDOM BASELINE RUN"
echo "Config: ${CONFIG_PATH}"
echo "Datasets: ${DATASETS[*]}"
echo "Examples per dataset: ${RANDOM_NUM_EXAMPLES}"
echo "Seeds: ${ARMAD_SEEDS}"
echo "Perm seeds: ${ARMAD_PERM_SEEDS}"
echo "Output dir: ${EXP_DIR}"
echo "========================================"
echo ""

for entry in "${MODELS[@]}"; do
  IFS='|' read -r model prefix <<< "$entry"
  run_loop "$model" "${RANDOM_BASELINE_GPUS:-0}" "$prefix"
done

echo ""
echo "========================================"
echo "RANDOM BASELINE RUN DONE"
echo "========================================"
echo ""
