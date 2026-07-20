#!/usr/bin/env bash
# Evaluate pilot v2 checkpoints on MATH test set.
set -euo pipefail

WOM_ROOT=/mnt/vast/home/ym56kacy/jinhe/MendelGM/wom
VERL_ROOT=$WOM_ROOT/verl
TOOL=$VERL_ROOT/experiments/moe_localized_rl/tools/eval_math_checkpoint.py
BASE_MODEL=$WOM_ROOT/models/OLMoE-1B-7B-0125-Instruct
TEST_FILE=$WOM_ROOT/data/math/test.parquet
OUT_DIR=$WOM_ROOT/outputs/pilot_v2_eval

export PATH=/mnt/vast/home/ym56kacy/jinhe/conda_env/verl/bin:$PATH
export PYTHONPATH=$VERL_ROOT:${PYTHONPATH:-}
export HF_HOME=/mnt/vast/home/ym56kacy/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# Avoid flashinfer JIT link failure (cannot find -lcudart) on compute nodes
export VLLM_USE_FLASHINFER_SAMPLER=0
export LD_LIBRARY_PATH=/mnt/vast/home/ym56kacy/jinhe/conda_env/verl/lib:${LD_LIBRARY_PATH:-}
export LIBRARY_PATH=/mnt/vast/home/ym56kacy/jinhe/conda_env/verl/lib:${LIBRARY_PATH:-}

cd "$VERL_ROOT"

run_eval() {
    local name=$1
    local actor_dir=$2
    local step=$3
    local notes=$4
    echo "========== Evaluating: $name (step=$step) =========="
    python3 "$TOOL" \
        --name "$name" \
        --actor_dir "$actor_dir" \
        --base_model "$BASE_MODEL" \
        --test_file "$TEST_FILE" \
        --output_dir "$OUT_DIR" \
        --training_step "$step" \
        --notes "$notes" \
        --tp 2 \
        --gpu_mem_util 0.45
}

# 1. Attention-only — healthy checkpoint (stopped at step ~232)
run_eval \
    "attention_only_step230" \
    "$WOM_ROOT/outputs/pilot_v2_attention_only/global_step_230/actor" \
    230 \
    "Healthy checkpoint; training stopped manually at ~232/435."

# 2. Full — only collapsed checkpoint remains on disk (step 340)
#    Pre-collapse step-60 ckpt was deleted by cleanup policy.
if [[ -d "$WOM_ROOT/outputs/pilot_v2_full/global_step_340/actor" ]]; then
    run_eval \
        "full_step340_collapsed" \
        "$WOM_ROOT/outputs/pilot_v2_full/global_step_340/actor" \
        340 \
        "COLLAPSED checkpoint only; pre-collapse step-60 ckpt no longer on disk."
fi

# 3. MoE — only collapsed checkpoint remains on disk (step 360)
if [[ -d "$WOM_ROOT/outputs/pilot_v2_moe/global_step_360/actor" ]]; then
    run_eval \
        "moe_step360_collapsed" \
        "$WOM_ROOT/outputs/pilot_v2_moe/global_step_360/actor" \
        360 \
        "COLLAPSED checkpoint only; pre-collapse step-70 ckpt no longer on disk."
fi

# 4. Base model baseline
echo "========== Evaluating: base_model =========="
python3 "$TOOL" \
    --name "base_model" \
    --hf_model "$BASE_MODEL" \
    --test_file "$TEST_FILE" \
    --output_dir "$OUT_DIR" \
    --notes "Untrained OLMoE-1B-7B-0125-Instruct baseline." \
    --tp 2 \
    --gpu_mem_util 0.45

echo ""
echo "All evaluations complete. Results in $OUT_DIR"
ls -la "$OUT_DIR"/*.json
