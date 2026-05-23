#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/.."

# =========================
# Environment
# =========================

export HF_HUB_CACHE=/home/dev/hf_cache/imad_models
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-$TRANSFORMERS_OFFLINE}
export TZ=${TZ:-Europe/London}
PYTHON_BIN=${PYTHON_BIN:-python}

# =========================
# Global experiment settings
# =========================

DATASETS=(
  mmlu_pro
  truthful_qa
  gsm8k
  math_500
)

# Per-dataset example caps
declare -A N_EX=(
  [mmlu_pro]=30
  [truthful_qa]=30
  [gsm8k]=30
  [math_500]=30
)

# ExpPlan_v3 first-pass sweep conditions are configured in configs/default.yaml.
# The shell loop sweeps only model x dataset; condition names are stored inside
# each run's results and transcripts, not in the folder name.
CONDITION_LABEL=${CONDITION_LABEL:-"cot, cot_sc, fixed_clique, armad_full"}
CONFIG_PATH=${CONFIG_PATH:-configs/default.yaml}
export ARMAD_EXP_TIMESTAMP=${ARMAD_EXP_TIMESTAMP:-$(date +%Y%m%d%H%M%S)}
EXP_DIR=${EXP_DIR:-outputs/exp_${ARMAD_EXP_TIMESTAMP}}
LOG_DIR=${LOG_DIR:-logs/exp_${ARMAD_EXP_TIMESTAMP}}

mkdir -p "$EXP_DIR"
mkdir -p "$LOG_DIR"

# Replication
# ARMAD_SEEDS and ARMAD_PERM_SEEDS are space-separated lists.
# Total runs per condition = examples x len(SEEDS) x len(PERMS).
if [[ -n "${ARMAD_SEEDS:-}" ]]; then
  read -r -a SEEDS <<< "$ARMAD_SEEDS"
else
  SEEDS=(0)
fi
if [[ -n "${ARMAD_PERM_SEEDS:-}" ]]; then
  read -r -a PERMS <<< "$ARMAD_PERM_SEEDS"
else
  PERMS=(10)
fi

# =========================
# Core runner
# =========================

run_loop () {

  local model=$1
  local gpus=$2
  local prefix=$3
  local -a extra_args=()

  if declare -p RUNNER_EXTRA_ARGS >/dev/null 2>&1; then
    extra_args=("${RUNNER_EXTRA_ARGS[@]}")
  fi

  for ds in "${DATASETS[@]}"; do

        local n=${N_EX[$ds]:-0}

        local tag="${prefix}_${ds}"

        echo ""
        echo "=================================================="
        echo "[$(date -Iseconds)] START ${tag}"
        echo "Model      : ${model}"
        echo "GPUs       : ${gpus}"
        echo "Output dir : ${EXP_DIR}"
        echo "Conditions : ${CONDITION_LABEL}"
        echo "Dataset    : ${ds}"
        echo "Examples   : ${n}"
        echo "Seeds      : ${SEEDS[*]}"
        echo "Perm seeds : ${PERMS[*]}"
        echo "=================================================="

        # Optional resume / skip
        if compgen -G "${EXP_DIR}/[0-9]*_${tag}/summary.json" > /dev/null; then
          echo "Skipping existing run: ${tag}"
          continue
        fi

        CUDA_VISIBLE_DEVICES="$gpus" \
        "$PYTHON_BIN" main.py \
          --config "$CONFIG_PATH" \
          --model "$model" \
          --dataset "$ds" \
          --num-examples "$n" \
          --output-dir "$EXP_DIR" \
          "${extra_args[@]}" \
          --tag "$tag" \
          --seeds "${SEEDS[@]}" \
          --perm-seeds "${PERMS[@]}" \
          2>&1 | tee "${LOG_DIR}/${tag}.log"

        echo "[$(date -Iseconds)] END ${tag}"
        echo ""
  done
}
