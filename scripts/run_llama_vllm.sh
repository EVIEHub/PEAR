#!/usr/bin/env bash

set -euo pipefail

# Backward-compatible wrapper for the parameterized vLLM runner.
export VLLM_DATASETS=${LLAMA_DATASETS:-${VLLM_DATASETS:-"gsm8k mmlu_pro"}}
export VLLM_NUM_EXAMPLES=${LLAMA_NUM_EXAMPLES:-${VLLM_NUM_EXAMPLES:-200}}
export VLLM_TARGET_GPUS=${LLAMA_GPUS:-${VLLM_TARGET_GPUS:-1}}
export VLLM_EXP_SUFFIX=${VLLM_EXP_SUFFIX:-llama31_8b_multiseed}

exec "$(dirname "$0")/run_vllm.sh" llama
