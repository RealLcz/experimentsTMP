#!/usr/bin/env bash
# Qwen3-30B-A3B RSPO 1-step diagnostic: verify weight sync + router capture + logprob alignment.
# Uses conservative smoke config (small batch, short context) per the modification guide.
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export SAVE_FREQ=-1
export TEST_FREQ=-1
export RESUME_MODE=disable
export VAL_BEFORE_TRAIN=False
export TOTAL_TRAINING_STEPS=1
export TOTAL_EPOCHS=1

export N_GPUS=${N_GPUS:-8}
export NNODES=${NNODES:-1}
# Smoke config per guide section 22: small batch, short context for first validation
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-2}
export MAX_CONTEXT_LENGTH=${MAX_CONTEXT_LENGTH:-8192}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-$((MAX_CONTEXT_LENGTH - MAX_PROMPT_LENGTH))}
export PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}
export ROLLOUT_N=${ROLLOUT_N:-12}
export ROLLOUT_TP=${ROLLOUT_TP:-8}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.40}

export PROJECT_NAME=${PROJECT_NAME:-qwen3_rspo_debug}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_rspo_diagnostic_1step}
export OUTPUT_DIR=${OUTPUT_DIR:-/mnt/vast/home/ym56kacy/jinhe/MendelGM/wom/outputs/qwen3_diagnostic_1step}
export ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-${OUTPUT_DIR}/rollout_dumps}
export PARAM_MODE=${PARAM_MODE:-attention_only}

mkdir -p "$OUTPUT_DIR" "$ROLLOUT_DATA_DIR"

echo "[diagnostic] mode=${PARAM_MODE} nnodes=${NNODES} n_gpus=${N_GPUS}"
echo "[diagnostic] context=${MAX_CONTEXT_LENGTH} prompt=${MAX_PROMPT_LENGTH} response=${MAX_RESPONSE_LENGTH}"
echo "[diagnostic] train_batch=${TRAIN_BATCH_SIZE} mini_batch=${PPO_MINI_BATCH_SIZE} rollout_n=${ROLLOUT_N}"
echo "[diagnostic] use_remove_padding=True (Qwen3 text-only MoE, flash_attention_2)"

bash "$SCRIPT_DIR/run_pilot_qwen3_full.sh" "$@"
