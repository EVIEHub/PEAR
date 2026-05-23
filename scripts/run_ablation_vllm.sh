#!/usr/bin/env bash

set -euo pipefail

export PYTHON_BIN=${PYTHON_BIN:-.venv-vllm/bin/python}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export CONFIG_PATH=${CONFIG_PATH:-configs/ablation.yaml}
export CONDITION_LABEL=${CONDITION_LABEL:-"armad_targeted_cross, armad_influence, armad_low_confidence, armad_targeted_influence, armad_targeted_low_confidence, armad_influence_low_confidence, armad_full"}

source "$(dirname "$0")/common.sh"

# Optional quick sweeps:
#   ABLATION_DATASETS="truthful_qa math_500" ABLATION_NUM_EXAMPLES=20 scripts/run_ablation_vllm.sh qwen
if [[ -z "${ABLATION_DATASETS:-}" ]]; then
  DATASETS=(truthful_qa math_500)
fi
if [[ -n "${ABLATION_DATASETS:-}" ]]; then
  read -r -a DATASETS <<< "$ABLATION_DATASETS"
fi
if [[ -n "${ABLATION_NUM_EXAMPLES:-}" ]]; then
  for ds in "${DATASETS[@]}"; do
    N_EX["$ds"]="$ABLATION_NUM_EXAMPLES"
  done
fi

TARGET=${1:-all}

run_gemma () {
  RUNNER_EXTRA_ARGS=(
    --parallel-examples "${PARALLEL_EXAMPLES:-1}"
    --model-override tensor_parallel_size=1
    --model-override gpu_memory_utilization=0.90
    --model-override max_model_len=8192
    --model-override enforce_eager=true
  )
  run_loop gemma-3-12b-vllm "${GEMMA_GPUS:-0}" gemma3_12b_vllm_ablation
}

run_qwen () {
  RUNNER_EXTRA_ARGS=(
    --parallel-examples "${PARALLEL_EXAMPLES:-1}"
  )
  run_loop qwen2.5-14b-vllm "${QWEN_GPUS:-1}" qwen25_14b_vllm_ablation
}

run_llama () {
  RUNNER_EXTRA_ARGS=(
    --parallel-examples "${PARALLEL_EXAMPLES:-1}"
  )
  run_loop llama-3.1-8b-vllm "${LLAMA_GPUS:-2}" llama31_8b_vllm_ablation
}

echo ""
echo "========================================"
echo "AR-MAD ABLATION vLLM RUN"
echo "Config            : ${CONFIG_PATH}"
echo "Target            : ${TARGET}"
echo "Models            : gemma-3-12b-vllm, llama-3.1-8b-vllm, qwen2.5-14b-vllm"
echo "Datasets          : ${DATASETS[*]}"
echo "Parallel examples : ${PARALLEL_EXAMPLES:-1}"
echo "Output dir        : ${EXP_DIR}"
echo "========================================"
echo ""

case "$TARGET" in
  gemma)
    run_gemma
    ;;
  qwen)
    run_qwen
    ;;
  llama)
    run_llama
    ;;
  all)
    run_gemma
    run_llama
    run_qwen
    ;;
  *)
    echo "Unknown target: ${TARGET}" >&2
    echo "Usage: $0 [gemma|llama|qwen|all]" >&2
    exit 2
    ;;
esac

echo ""
echo "========================================"
echo "AR-MAD ABLATION vLLM DONE"
echo "========================================"
echo ""
