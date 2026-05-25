#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

usage () {
  cat <<'USAGE'
Usage:
  scripts/run_vllm.sh [target ...]

Targets:
  qwen7b | qwen        Qwen2.5-7B-Instruct
  qwen14b | qwen14     Qwen2.5-14B-Instruct
  llama                Llama-3.1-8B-Instruct
  gemma                Gemma-3-12B-IT
  qwen3                Qwen3-30B-A3B-Instruct
  deepseek             DeepSeek-R1-Distill-Qwen-14B
  open4                qwen7b + llama + gemma + qwen3

Common environment overrides:
  VLLM_DATASETS="gsm8k mmlu_pro"
  VLLM_NUM_EXAMPLES=200
  VLLM_TARGET_GPUS=0              # useful for a single target
  PARALLEL_EXAMPLES=1
  CONFIG_PATH=configs/main_large.yaml
  PEAR_SEEDS="1 2 3"
  PEAR_PERM_SEEDS="10"

Model-specific overrides still work, e.g.:
  QWEN_GPUS=0 QWEN_NUM_EXAMPLES=200 scripts/run_vllm.sh qwen7b
  GEMMA_GPUS=2 GEMMA_NUM_EXAMPLES=40 scripts/run_vllm.sh gemma
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

TARGETS=("${@:-qwen7b}")
if [[ ${#TARGETS[@]} -eq 1 ]]; then
  case "${TARGETS[0]}" in
    open4)
      TARGETS=(qwen7b llama gemma qwen3)
      ;;
  esac
fi

_default_suffix () {
  case "$1" in
    qwen|qwen7b) echo "qwen25_7b_multiseed" ;;
    qwen14|qwen14b) echo "qwen25_14b_multiseed" ;;
    llama) echo "llama31_8b_multiseed" ;;
    gemma) echo "gemma3_12b_multiseed" ;;
    qwen3) echo "qwen3_30b_multiseed" ;;
    deepseek) echo "deepseek14b" ;;
    *) echo "vllm" ;;
  esac
}

if [[ ${#TARGETS[@]} -eq 1 ]]; then
  DEFAULT_SUFFIX="$(_default_suffix "${TARGETS[0]}")"
else
  DEFAULT_SUFFIX="vllm_multi"
fi

export PYTHON_BIN=${PYTHON_BIN:-.venv-vllm/bin/python}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export CONFIG_PATH=${CONFIG_PATH:-configs/main_large.yaml}
export CONDITION_LABEL=${CONDITION_LABEL:-"cot, cot_sc, fixed_clique, fixed_star, fixed_chain, fixed_ring, random_k_regular, pear_full"}
export PEAR_EXP_TIMESTAMP=${PEAR_EXP_TIMESTAMP:-$(TZ=Europe/London date +%Y%m%d%H%M%S)_${VLLM_EXP_SUFFIX:-$DEFAULT_SUFFIX}}
export PEAR_SEEDS=${PEAR_SEEDS:-"1 2 3"}
export PEAR_PERM_SEEDS=${PEAR_PERM_SEEDS:-"10"}

source "${SCRIPT_DIR}/common.sh"

if [[ -n "${VLLM_DATASETS:-}" ]]; then
  read -r -a DATASETS <<< "$VLLM_DATASETS"
else
  DATASETS=(gsm8k mmlu_pro)
fi

_set_dataset_caps () {
  local default_n=$1
  for ds in "${DATASETS[@]}"; do
    case "$ds" in
      mmlu_pro) N_EX[$ds]="${VLLM_N_MMLU_PRO:-$default_n}" ;;
      truthful_qa) N_EX[$ds]="${VLLM_N_TRUTHFUL_QA:-$default_n}" ;;
      gsm8k) N_EX[$ds]="${VLLM_N_GSM8K:-$default_n}" ;;
      math_500) N_EX[$ds]="${VLLM_N_MATH_500:-$default_n}" ;;
      *) N_EX[$ds]="$default_n" ;;
    esac
  done
}

_run_target () {
  local target=$1
  local model=""
  local gpus=""
  local tag=""
  local display=""
  local examples=""

  RUNNER_EXTRA_ARGS=(--parallel-examples "${PARALLEL_EXAMPLES:-1}")

  case "$target" in
    qwen|qwen7b)
      model="qwen2.5-7b-vllm"
      gpus="${VLLM_TARGET_GPUS:-${QWEN_GPUS:-0}}"
      tag="qwen25_7b_vllm"
      display="Qwen2.5-7B"
      examples="${QWEN_NUM_EXAMPLES:-${VLLM_NUM_EXAMPLES:-200}}"
      ;;
    qwen14|qwen14b)
      model="qwen2.5-14b-vllm"
      gpus="${VLLM_TARGET_GPUS:-${QWEN_GPUS:-1}}"
      tag="qwen25_14b_vllm"
      display="Qwen2.5-14B"
      examples="${QWEN_NUM_EXAMPLES:-${VLLM_NUM_EXAMPLES:-200}}"
      ;;
    llama)
      model="llama-3.1-8b-vllm"
      gpus="${VLLM_TARGET_GPUS:-${LLAMA_GPUS:-1}}"
      tag="llama31_8b_vllm"
      display="Llama-3.1-8B"
      examples="${LLAMA_NUM_EXAMPLES:-${VLLM_NUM_EXAMPLES:-200}}"
      ;;
    gemma)
      model="gemma-3-12b-vllm"
      gpus="${VLLM_TARGET_GPUS:-${GEMMA_GPUS:-2}}"
      tag="gemma3_12b_vllm"
      display="Gemma-3-12B"
      examples="${GEMMA_NUM_EXAMPLES:-${VLLM_NUM_EXAMPLES:-200}}"
      RUNNER_EXTRA_ARGS+=(
        --model-override tensor_parallel_size=1
        --model-override gpu_memory_utilization=0.90
        --model-override max_model_len=8192
        --model-override enforce_eager=true
      )
      ;;
    qwen3)
      model="qwen3-30b-a3b-local-vllm"
      gpus="${VLLM_TARGET_GPUS:-${QWEN3_GPUS:-3}}"
      tag="qwen3_30b_a3b_vllm"
      display="Qwen3-30B-A3B"
      examples="${QWEN3_NUM_EXAMPLES:-${VLLM_NUM_EXAMPLES:-200}}"
      ;;
    deepseek)
      model="deepseek-r1-distill-qwen-14b-vllm"
      gpus="${VLLM_TARGET_GPUS:-${DEEPSEEK_GPUS:-2}}"
      tag="deepseek_r1_distill_qwen_14b_vllm"
      display="DeepSeek-R1-Distill-Qwen-14B"
      examples="${DEEPSEEK_NUM_EXAMPLES:-${VLLM_NUM_EXAMPLES:-30}}"
      ;;
    *)
      echo "Unknown vLLM target: ${target}" >&2
      usage >&2
      exit 2
      ;;
  esac

  _set_dataset_caps "$examples"

  echo ""
  echo "========================================"
  echo "${display} vLLM RUN"
  echo "Target            : ${target}"
  echo "Model             : ${model}"
  echo "GPU(s)            : ${gpus}"
  echo "Config            : ${CONFIG_PATH}"
  echo "Datasets          : ${DATASETS[*]}"
  echo "Examples/dataset  : ${examples}"
  echo "Seeds             : ${PEAR_SEEDS}"
  echo "Perm seeds        : ${PEAR_PERM_SEEDS}"
  echo "Parallel examples : ${PARALLEL_EXAMPLES:-1}"
  echo "Output dir        : ${EXP_DIR}"
  echo "========================================"
  echo ""

  run_loop "$model" "$gpus" "$tag"
}

echo ""
echo "========================================"
echo "PARAMETERIZED vLLM RUN"
echo "Targets: ${TARGETS[*]}"
echo "========================================"
echo ""

for target in "${TARGETS[@]}"; do
  _run_target "$target"
done

echo ""
echo "========================================"
echo "PARAMETERIZED vLLM RUN DONE"
echo "========================================"
echo ""
