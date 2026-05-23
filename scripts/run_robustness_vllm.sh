#!/usr/bin/env bash

set -euo pipefail

export PYTHON_BIN=${PYTHON_BIN:-.venv-vllm/bin/python}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export ARMAD_EXP_TIMESTAMP=${ARMAD_EXP_TIMESTAMP:-$(TZ=Europe/London date +%Y%m%d%H%M%S)_robustness}
export CONDITION_LABEL=${ROBUSTNESS_CONDITION_LABEL:-"fixed_clique, fixed_star, fixed_chain, fixed_ring, armad_full"}

source "$(dirname "$0")/common.sh"

if [[ -n "${ROBUSTNESS_DATASETS:-}" ]]; then
  read -r -a DATASETS <<< "$ROBUSTNESS_DATASETS"
fi

set_model_examples () {
  local default_n=$1
  N_EX[mmlu_pro]=${ROBUSTNESS_N_MMLU_PRO:-$default_n}
  N_EX[truthful_qa]=${ROBUSTNESS_N_TRUTHFUL_QA:-$default_n}
  N_EX[gsm8k]=${ROBUSTNESS_N_GSM8K:-$default_n}
  N_EX[math_500]=${ROBUSTNESS_N_MATH_500:-$default_n}
}

scenario_config () {
  case "$1" in
    malicious) echo "configs/robustness_malicious.yaml" ;;
    confidence) echo "configs/robustness_confidence.yaml" ;;
    critique|critique_noise) echo "configs/robustness_critique_noise.yaml" ;;
    *)
      echo "Unknown robustness scenario: $1" >&2
      echo "Use malicious | confidence | critique | all" >&2
      exit 2
      ;;
  esac
}

GEMMA_NUM_EXAMPLES=${GEMMA_NUM_EXAMPLES:-40}
QWEN_NUM_EXAMPLES=${QWEN_NUM_EXAMPLES:-50}
LLAMA_NUM_EXAMPLES=${LLAMA_NUM_EXAMPLES:-50}

TARGET=${1:-all}
SCENARIO_ARG=${2:-${ROBUSTNESS_SCENARIO:-all}}

run_gemma () {
  local scenario=$1
  set_model_examples "$GEMMA_NUM_EXAMPLES"
  RUNNER_EXTRA_ARGS=(
    --parallel-examples "${PARALLEL_EXAMPLES:-1}"
    --model-override tensor_parallel_size=1
    --model-override gpu_memory_utilization=0.90
    --model-override max_model_len=8192
    --model-override enforce_eager=true
  )
  run_loop gemma-3-12b-vllm "${GEMMA_GPUS:-0}" "gemma3_12b_vllm_${scenario}"
}

run_qwen () {
  local scenario=$1
  set_model_examples "$QWEN_NUM_EXAMPLES"
  RUNNER_EXTRA_ARGS=(
    --parallel-examples "${PARALLEL_EXAMPLES:-1}"
  )
  run_loop qwen2.5-14b-vllm "${QWEN_GPUS:-1}" "qwen25_14b_vllm_${scenario}"
}

run_llama () {
  local scenario=$1
  set_model_examples "$LLAMA_NUM_EXAMPLES"
  RUNNER_EXTRA_ARGS=(
    --parallel-examples "${PARALLEL_EXAMPLES:-1}"
  )
  run_loop llama-3.1-8b-vllm "${LLAMA_GPUS:-2}" "llama31_8b_vllm_${scenario}"
}

run_target_for_scenario () {
  local scenario=$1
  export CONFIG_PATH=$(scenario_config "$scenario")
  echo ""
  echo "========================================"
  echo "AR-MAD ROBUSTNESS vLLM RUN"
  echo "Scenario          : ${scenario}"
  echo "Config            : ${CONFIG_PATH}"
  echo "Target            : ${TARGET}"
  echo "Datasets          : ${DATASETS[*]}"
  echo "Examples          : Gemma=${GEMMA_NUM_EXAMPLES}, Qwen=${QWEN_NUM_EXAMPLES}, Llama=${LLAMA_NUM_EXAMPLES}"
  echo "Parallel examples : ${PARALLEL_EXAMPLES:-1}"
  echo "Output dir        : ${EXP_DIR}"
  echo "========================================"
  echo ""

  case "$TARGET" in
    gemma) run_gemma "$scenario" ;;
    qwen) run_qwen "$scenario" ;;
    llama) run_llama "$scenario" ;;
    all)
      run_gemma "$scenario"
      run_qwen "$scenario"
      run_llama "$scenario"
      ;;
    *)
      echo "Unknown target: ${TARGET}" >&2
      echo "Usage: $0 [gemma|qwen|llama|all] [malicious|confidence|critique|all]" >&2
      exit 2
      ;;
  esac
}

if [[ "$SCENARIO_ARG" == "all" ]]; then
  for scenario in malicious confidence critique; do
    run_target_for_scenario "$scenario"
  done
else
  run_target_for_scenario "$SCENARIO_ARG"
fi

echo ""
echo "========================================"
echo "AR-MAD ROBUSTNESS vLLM DONE"
echo "========================================"
echo ""
