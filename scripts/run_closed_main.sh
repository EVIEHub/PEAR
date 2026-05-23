#!/usr/bin/env bash

set -euo pipefail

export PYTHON_BIN=${PYTHON_BIN:-.venv-vllm/bin/python}
export CONFIG_PATH=${CONFIG_PATH:-configs/main_large.yaml}
export CONDITION_LABEL=${CONDITION_LABEL:-"cot, cot_sc, fixed_clique, fixed_star, fixed_chain, fixed_ring, random_k_regular, armad_full"}
export ARMAD_EXP_TIMESTAMP=${ARMAD_EXP_TIMESTAMP:-$(TZ=Europe/London date +%Y%m%d%H%M%S)_closed_main}

source "$(dirname "$0")/common.sh"

# Keep the closed-source sweep bounded by default. Override with
# CLOSED_NUM_EXAMPLES=0 to use the full local dataset splits.
CLOSED_NUM_EXAMPLES=${CLOSED_NUM_EXAMPLES:-20}

if [[ -n "${CLOSED_DATASETS:-}" ]]; then
  read -r -a DATASETS <<< "$CLOSED_DATASETS"
fi

N_EX[mmlu_pro]=${CLOSED_N_MMLU_PRO:-$CLOSED_NUM_EXAMPLES}
N_EX[truthful_qa]=${CLOSED_N_TRUTHFUL_QA:-$CLOSED_NUM_EXAMPLES}
N_EX[gsm8k]=${CLOSED_N_GSM8K:-$CLOSED_NUM_EXAMPLES}
N_EX[math_500]=${CLOSED_N_MATH_500:-$CLOSED_NUM_EXAMPLES}

TARGET=${1:-all}

run_gpt_nano () {
  RUNNER_EXTRA_ARGS=(
    --parallel-examples "${PARALLEL_EXAMPLES:-1}"
  )
  run_loop gpt-5.4-nano-2026-03-17 "" gpt54_nano
}

run_claude_haiku () {
  RUNNER_EXTRA_ARGS=(
    --parallel-examples "${PARALLEL_EXAMPLES:-1}"
  )
  run_loop claude-haiku-4-5 "" claude_haiku_45
}

echo ""
echo "========================================"
echo "AR-MAD CLOSED-SOURCE MAIN RUN"
echo "Config            : ${CONFIG_PATH}"
echo "Target            : ${TARGET}"
echo "Datasets          : ${DATASETS[*]}"
echo "Examples          : ${CLOSED_NUM_EXAMPLES}"
echo "Parallel examples : ${PARALLEL_EXAMPLES:-1}"
echo "Output dir        : ${EXP_DIR}"
echo "API key env       : OPENAI_API_KEY"
echo "Base URL env      : OPENAI_BASE_URL"
echo "========================================"
echo ""

case "$TARGET" in
  gpt|gpt_nano|gpt-5.4-nano)
    run_gpt_nano
    ;;
  claude|haiku|claude-haiku)
    run_claude_haiku
    ;;
  all)
    run_gpt_nano
    run_claude_haiku
    ;;
  *)
    echo "Unknown target: ${TARGET}" >&2
    echo "Usage: $0 [gpt|claude|all]" >&2
    exit 2
    ;;
esac

echo ""
echo "========================================"
echo "AR-MAD CLOSED-SOURCE MAIN DONE"
echo "========================================"
echo ""
