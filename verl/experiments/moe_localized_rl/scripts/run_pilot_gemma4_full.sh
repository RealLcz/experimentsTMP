#!/usr/bin/env bash
# Gemma 4 26B-A4B Pilot: Full GSPO + Router Shift (RSPO) on MATH.
# Checkpoint: save_freq=10, keep latest only, auto-resume on restart.
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/pilot_v2_checkpoint_policy.sh"

########################### user-adjustable ###########################
MODEL_PATH=${MODEL_PATH:-/mnt/vast/home/ym56kacy/jinhe/MendelGM/wom/models/gemma-4-26B-A4B-it}
TRAIN_FILE=${TRAIN_FILE:-/mnt/vast/home/ym56kacy/jinhe/MendelGM/wom/data/math/train.parquet}
TEST_FILE=${TEST_FILE:-/mnt/vast/home/ym56kacy/jinhe/MendelGM/wom/data/math/test.parquet}

N_GPUS=${N_GPUS:-8}
NNODES=${NNODES:-1}
# Context length is fixed at 8k for MATH CoT + reward parsing. GRPO needs
# rollout.n>=12 for stable group-relative rewards; batch sizes stay small for memory.
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-4}
# 8k total context for MATH CoT: ~2k prompt + ~6k generation budget.
MAX_CONTEXT_LENGTH=${MAX_CONTEXT_LENGTH:-8192}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-$((MAX_CONTEXT_LENGTH - MAX_PROMPT_LENGTH))}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}
ROLLOUT_N=${ROLLOUT_N:-12}
# One TP=8 rollout replica avoids four independent server processes racing to
# initialise the same JIT/binary extension caches on an 8-GPU H200 node.
ROLLOUT_TP=${ROLLOUT_TP:-8}
# 8k max_model_len needs a larger KV reservation; keep util moderate on H100.
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.38}
# In the colocated V1 trainer, vLLM weights are restored after each actor
# update.  Keeping the FSDP shard and Adam states resident at that point can
# exceed H100 memory (and is substantially worse for the full-parameter run).
# Offload between engine calls; the train context stages them back before the
# next forward/update, so this changes placement/performance, not semantics.
ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-false}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-false}
# FSDP2 CPUOffloadPolicy keeps shards off GPU between train steps; required on
# single-node colocated runs where vLLM TP shares the same 8 GPUs as FSDP.
ACTOR_OFFLOAD_POLICY=${ACTOR_OFFLOAD_POLICY:-false}

ACTOR_LR=${ACTOR_LR:-1e-6}
CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-3e-4}
CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-4e-4}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-15}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}

PROJECT_NAME=${PROJECT_NAME:-gemma4_localized_rl}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-gemma4_26b_a4b_rspo_full_math_seed0}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/vast/home/ym56kacy/jinhe/MendelGM/wom/outputs/workingF}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-}
PARAM_MODE=${PARAM_MODE:-full}
SEED=${SEED:-42}
########################### end user-adjustable ###########################

mkdir -p "$OUTPUT_DIR"

echo "[gemma4] context=${MAX_CONTEXT_LENGTH} prompt=${MAX_PROMPT_LENGTH} response=${MAX_RESPONSE_LENGTH}"
echo "[gemma4] train_batch=${TRAIN_BATCH_SIZE} mini_batch=${PPO_MINI_BATCH_SIZE} rollout_n=${ROLLOUT_N}"

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
    data.continuous_token.model_family=gemma4
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    # Gemma4 is run with SDPA below. The generic remove-padding path flattens
    # independent samples and relies on FlashAttention varlen boundaries;
    # with SDPA that would incorrectly allow cross-sample attention and make
    # recomputed actor log-probabilities diverge from rollout probabilities.
    actor_rollout_ref.model.use_remove_padding=False
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.trust_remote_code=True
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa
)

ACTOR=(
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.fsdp_config.strategy=fsdp2
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo
    actor_rollout_ref.actor.policy_loss.router_shift.enabled=true
    actor_rollout_ref.actor.policy_loss.router_shift.gamma_min=0.8
    actor_rollout_ref.actor.policy_loss.router_shift.stop_gradient=true
    actor_rollout_ref.actor.policy_loss.router_shift.log_diagnostics=true
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
    actor_rollout_ref.actor.fsdp_config.param_offload=${ACTOR_PARAM_OFFLOAD}
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD}
    actor_rollout_ref.actor.fsdp_config.offload_policy=${ACTOR_OFFLOAD_POLICY}
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
    # Avoid concurrent FlashInfer JIT/cache races across the four TP=2 rollout
    # replicas on this cluster. Native NCCL + eager mode is slower but robust
    # and does not change sampling or training semantics.
    actor_rollout_ref.rollout.enforce_eager=true
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce=true
    +actor_rollout_ref.rollout.engine_kwargs.vllm.moe_backend=triton
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.max_model_len=${MAX_CONTEXT_LENGTH}
)

REF=(
    actor_rollout_ref.ref.strategy=fsdp2
    actor_rollout_ref.ref.fsdp_config.strategy=fsdp2
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
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
    trainer.val_before_train=${VAL_BEFORE_TRAIN}
)

if [[ -n "$ROLLOUT_DATA_DIR" ]]; then
    TRAINER+=(trainer.rollout_data_dir=${ROLLOUT_DATA_DIR})
fi

RAY_ARGS=()
if [[ -n "${RAY_ADDRESS:-}" ]]; then
    RAY_ARGS+=(+ray_kwargs.ray_init.address="${RAY_ADDRESS}")
fi

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "${RAY_ARGS[@]}" \
    "$@"
