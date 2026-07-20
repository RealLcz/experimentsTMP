#!/usr/bin/env bash
# Gemma4 RSPO 1-step diagnostic: verify padded SDPA + logprob alignment before 100-step pilots.
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
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-4}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-2}
export MAX_CONTEXT_LENGTH=${MAX_CONTEXT_LENGTH:-8192}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-$((MAX_CONTEXT_LENGTH - MAX_PROMPT_LENGTH))}
export PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}
export ROLLOUT_N=${ROLLOUT_N:-12}
export ROLLOUT_TP=${ROLLOUT_TP:-8}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.35}

export PROJECT_NAME=${PROJECT_NAME:-gemma4_rspo_debug}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-gemma4_rspo_diagnostic_1step}
export OUTPUT_DIR=${OUTPUT_DIR:-/mnt/vast/home/ym56kacy/jinhe/MendelGM/wom/outputs/gemma4_diagnostic_1step}
export ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-${OUTPUT_DIR}/rollout_dumps}
export PARAM_MODE=${PARAM_MODE:-attention_only}

mkdir -p "$OUTPUT_DIR" "$ROLLOUT_DATA_DIR"

echo "[diagnostic] mode=${PARAM_MODE} nnodes=${NNODES} n_gpus=${N_GPUS}"
echo "[diagnostic] context=${MAX_CONTEXT_LENGTH} prompt=${MAX_PROMPT_LENGTH} response=${MAX_RESPONSE_LENGTH}"
echo "[diagnostic] train_batch=${TRAIN_BATCH_SIZE} mini_batch=${PPO_MINI_BATCH_SIZE} rollout_n=${ROLLOUT_N}"
echo "[diagnostic] use_remove_padding=False (required for Gemma4 SDPA)"
echo "[diagnostic] rollout_data_dir=${ROLLOUT_DATA_DIR}"

bash "$SCRIPT_DIR/run_pilot_gemma4_full.sh" "$@"
