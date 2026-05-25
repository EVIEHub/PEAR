#!/usr/bin/env bash

set -euo pipefail

export PYTHON_BIN=${PYTHON_BIN:-.venv-vllm/bin/python}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export CONFIG_PATH=${CONFIG_PATH:-configs/main_large.yaml}
export CONDITION_LABEL=${CONDITION_LABEL:-"cot, cot_sc, fixed_clique, fixed_star, fixed_chain, fixed_ring, random_k_regular, pear_full"}
export PEAR_EXP_TIMESTAMP=${PEAR_EXP_TIMESTAMP:-$(TZ=Europe/London date +%Y%m%d%H%M%S)_main}

source "$(dirname "$0")/common.sh"

if [[ -n "${MAIN_DATASETS:-}" ]]; then
  read -r -a DATASETS <<< "$MAIN_DATASETS"
fi

set_model_examples () {
  local default_n=$1
  N_EX[mmlu_pro]=${MAIN_N_MMLU_PRO:-$default_n}
  N_EX[truthful_qa]=${MAIN_N_TRUTHFUL_QA:-$default_n}
  N_EX[gsm8k]=${MAIN_N_GSM8K:-$default_n}
  N_EX[math_500]=${MAIN_N_MATH_500:-$default_n}
}

GEMMA_NUM_EXAMPLES=${GEMMA_NUM_EXAMPLES:-40}
QWEN_NUM_EXAMPLES=${QWEN_NUM_EXAMPLES:-50}
LLAMA_NUM_EXAMPLES=${LLAMA_NUM_EXAMPLES:-50}

TARGET=${1:-all}

run_gemma () {
  set_model_examples "$GEMMA_NUM_EXAMPLES"
  RUNNER_EXTRA_ARGS=(
    --parallel-examples "${PARALLEL_EXAMPLES:-1}"
    --model-override tensor_parallel_size=1
    --model-override gpu_memory_utilization=0.90
    --model-override max_model_len=8192
    --model-override enforce_eager=true
  )
  run_loop gemma-3-12b-vllm "${GEMMA_GPUS:-0}" gemma3_12b_vllm
}

run_qwen () {
  set_model_examples "$QWEN_NUM_EXAMPLES"
  RUNNER_EXTRA_ARGS=(
    --parallel-examples "${PARALLEL_EXAMPLES:-1}"
  )
  run_loop qwen2.5-14b-vllm "${QWEN_GPUS:-1}" qwen25_14b_vllm
}

run_llama () {
  set_model_examples "$LLAMA_NUM_EXAMPLES"
  RUNNER_EXTRA_ARGS=(
    --parallel-examples "${PARALLEL_EXAMPLES:-1}"
  )
  run_loop llama-3.1-8b-vllm "${LLAMA_GPUS:-2}" llama31_8b_vllm
}

echo ""
echo "========================================"
echo "PEAR MAIN LARGE vLLM RUN"
echo "Config            : ${CONFIG_PATH}"
echo "Target            : ${TARGET}"
echo "Datasets          : ${DATASETS[*]}"
echo "Examples          : Gemma=${GEMMA_NUM_EXAMPLES}, Qwen=${QWEN_NUM_EXAMPLES}, Llama=${LLAMA_NUM_EXAMPLES}"
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
    run_qwen
    run_llama
    ;;
  *)
    echo "Unknown target: ${TARGET}" >&2
    echo "Usage: $0 [gemma|qwen|llama|all]" >&2
    exit 2
    ;;
esac

echo ""
echo "========================================"
echo "PEAR MAIN LARGE vLLM DONE"
echo "========================================"
echo ""
