#!/usr/bin/env bash

set -euo pipefail

# Backward-compatible wrapper for the parameterized vLLM runner.
export CONFIG_PATH=${CONFIG_PATH:-configs/main_large.yaml}
export CONDITION_LABEL=${CONDITION_LABEL:-"cot, cot_sc, fixed_clique, fixed_star, fixed_chain, fixed_ring, random_k_regular, armad_full"}
export VLLM_NUM_EXAMPLES=${DEEPSEEK_NUM_EXAMPLES:-${VLLM_NUM_EXAMPLES:-30}}
export VLLM_TARGET_GPUS=${DEEPSEEK_GPUS:-${VLLM_TARGET_GPUS:-2}}
export VLLM_EXP_SUFFIX=${VLLM_EXP_SUFFIX:-tuned}
export ARMAD_SEEDS=${ARMAD_SEEDS:-"0"}
export ARMAD_PERM_SEEDS=${ARMAD_PERM_SEEDS:-"10"}

exec "$(dirname "$0")/run_vllm.sh" deepseek
