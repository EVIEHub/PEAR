#!/usr/bin/env bash

set -euo pipefail

# Backward-compatible wrapper for the parameterized vLLM runner.
export VLLM_DATASETS=${QWEN_DATASETS:-${VLLM_DATASETS:-"gsm8k mmlu_pro"}}
export VLLM_NUM_EXAMPLES=${QWEN_NUM_EXAMPLES:-${VLLM_NUM_EXAMPLES:-200}}
export VLLM_TARGET_GPUS=${QWEN_GPUS:-${VLLM_TARGET_GPUS:-0}}
export VLLM_EXP_SUFFIX=${VLLM_EXP_SUFFIX:-qwen25_7b_multiseed}

exec "$(dirname "$0")/run_vllm.sh" qwen7b
