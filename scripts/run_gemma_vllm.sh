#!/usr/bin/env bash

set -euo pipefail

# Backward-compatible wrapper for the parameterized vLLM runner.
export VLLM_DATASETS=${GEMMA_DATASETS:-${VLLM_DATASETS:-"gsm8k mmlu_pro"}}
export VLLM_NUM_EXAMPLES=${GEMMA_NUM_EXAMPLES:-${VLLM_NUM_EXAMPLES:-200}}
export VLLM_TARGET_GPUS=${GEMMA_GPUS:-${VLLM_TARGET_GPUS:-2}}
export VLLM_EXP_SUFFIX=${VLLM_EXP_SUFFIX:-gemma3_12b_multiseed}

exec "$(dirname "$0")/run_vllm.sh" gemma
