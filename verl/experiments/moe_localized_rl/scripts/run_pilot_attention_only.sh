#!/usr/bin/env bash
# MoE-Localized RL Pilot v2: Attention-only GSPO on OLMoE + MATH.
# Checkpoint: save_freq=10, keep latest only, auto-resume on restart.
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/pilot_v2_checkpoint_policy.sh"

########################### user-adjustable ###########################
MODEL_PATH=${MODEL_PATH:-/mnt/vast/home/ym56kacy/jinhe/MendelGM/wom/models/OLMoE-1B-7B-0125-Instruct}
TRAIN_FILE=${TRAIN_FILE:-/mnt/vast/home/ym56kacy/jinhe/MendelGM/wom/data/math/train.parquet}
TEST_FILE=${TEST_FILE:-/mnt/vast/home/ym56kacy/jinhe/MendelGM/wom/data/math/test.parquet}

N_GPUS=${N_GPUS:-8}
NNODES=${NNODES:-1}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}
ROLLOUT_N=${ROLLOUT_N:-16}
ROLLOUT_TP=${ROLLOUT_TP:-2}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.45}

ACTOR_LR=${ACTOR_LR:-1e-6}
CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-3e-4}
CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-4e-4}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-15}

PROJECT_NAME=${PROJECT_NAME:-moe_localized_rl_pilot_v2}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-olmoe_gspo_attention_only_math_seed0}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/vast/home/ym56kacy/jinhe/MendelGM/wom/outputs/pilot_v2_attention_only}
PARAM_MODE=${PARAM_MODE:-attention_only}
SEED=${SEED:-42}
########################### end user-adjustable ###########################

mkdir -p "$OUTPUT_DIR"

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="['$TRAIN_FILE']"
    data.val_files="['$TEST_FILE']"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.shuffle=True
    data.seed=${SEED}
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean
    actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW}
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH}
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.data_loader_seed=${SEED}
    actor_rollout_ref.actor.checkpoint.save_contents="${CKPT_SAVE_CONTENTS}"
    actor_rollout_ref.actor.parameter_update.mode=${PARAM_MODE}
    actor_rollout_ref.actor.parameter_update.strict=true
    actor_rollout_ref.actor.parameter_update.log_parameter_stats=true
    actor_rollout_ref.actor.parameter_update.output_dir=${OUTPUT_DIR}
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
)

REF=(
    actor_rollout_ref.ref.strategy=fsdp2
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

TRAINER=(
    trainer.balance_batch=True
    trainer.critic_warmup=0
    trainer.logger='["console","tensorboard"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.default_local_dir=${OUTPUT_DIR}
    trainer.n_gpus_per_node=${N_GPUS}
    trainer.nnodes=${NNODES}
    trainer.save_freq=${SAVE_FREQ}
    trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP}
    trainer.resume_mode=${RESUME_MODE}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
    trainer.val_before_train=${VAL_BEFORE_TRAIN}
)

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "$@"
