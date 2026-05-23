#!/usr/bin/env bash

set -euo pipefail

export PYTHON_BIN=${PYTHON_BIN:-.venv-vllm/bin/python}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export CONFIG_PATH=${CONFIG_PATH:-configs/random_k_regular_baseline.yaml}
export CONDITION_LABEL=${CONDITION_LABEL:-"random_k_regular"}
export ARMAD_EXP_TIMESTAMP=${ARMAD_EXP_TIMESTAMP:-$(TZ=Europe/London date +%Y%m%d%H%M%S)_random_k_regular}
export ARMAD_SEEDS=${ARMAD_SEEDS:-"1"}
export ARMAD_PERM_SEEDS=${ARMAD_PERM_SEEDS:-"10"}

source "$(dirname "$0")/common.sh"

if [[ -n "${RANDOM_K_DATASETS:-}" ]]; then
  read -r -a DATASETS <<< "$RANDOM_K_DATASETS"
else
  DATASETS=(mmlu_pro truthful_qa gsm8k math_500)
fi

RANDOM_K_NUM_EXAMPLES=${RANDOM_K_NUM_EXAMPLES:-80}
for ds in "${DATASETS[@]}"; do
  N_EX[$ds]="${RANDOM_K_NUM_EXAMPLES}"
done

RUNNER_EXTRA_ARGS=(
  --parallel-examples "${PARALLEL_EXAMPLES:-1}"
)

echo ""
echo "========================================"
echo "RANDOM K-REGULAR BASELINE RUN"
echo "Config: ${CONFIG_PATH}"
echo "Datasets: ${DATASETS[*]}"
echo "Examples per dataset: ${RANDOM_K_NUM_EXAMPLES}"
echo "Seeds: ${ARMAD_SEEDS}"
echo "Perm seeds: ${ARMAD_PERM_SEEDS}"
echo "Output dir: ${EXP_DIR}"
echo "========================================"
echo ""

case "${1:-both}" in
  qwen)
    run_loop qwen2.5-7b-vllm "${QWEN_GPUS:-0}" qwen25_7b_vllm
    ;;
  llama)
    run_loop llama-3.1-8b-vllm "${LLAMA_GPUS:-1}" llama31_8b_vllm
    ;;
  gemma)
    run_loop gemma-3-12b-vllm "${GEMMA_GPUS:-0}" gemma3_12b_vllm
    ;;
  qwen3)
    run_loop qwen3-30b-a3b-local-vllm "${QWEN3_GPUS:-1}" qwen3_30b_a3b_vllm
    ;;
  both)
    run_loop qwen2.5-7b-vllm "${QWEN_GPUS:-0}" qwen25_7b_vllm
    run_loop llama-3.1-8b-vllm "${LLAMA_GPUS:-1}" llama31_8b_vllm
    ;;
  remaining)
    run_loop gemma-3-12b-vllm "${GEMMA_GPUS:-0}" gemma3_12b_vllm
    run_loop qwen3-30b-a3b-local-vllm "${QWEN3_GPUS:-1}" qwen3_30b_a3b_vllm
    ;;
  open4)
    run_loop qwen2.5-7b-vllm "${QWEN_GPUS:-0}" qwen25_7b_vllm
    run_loop llama-3.1-8b-vllm "${LLAMA_GPUS:-1}" llama31_8b_vllm
    run_loop gemma-3-12b-vllm "${GEMMA_GPUS:-0}" gemma3_12b_vllm
    run_loop qwen3-30b-a3b-local-vllm "${QWEN3_GPUS:-1}" qwen3_30b_a3b_vllm
    ;;
  *)
    echo "Usage: $0 [qwen|llama|gemma|qwen3|both|remaining|open4]" >&2
    exit 2
    ;;
esac

echo ""
echo "========================================"
echo "RANDOM K-REGULAR BASELINE RUN DONE"
echo "========================================"
echo ""
