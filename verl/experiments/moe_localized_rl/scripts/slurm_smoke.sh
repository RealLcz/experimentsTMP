#!/usr/bin/env bash
#SBATCH --job-name=olmoe_gspo_smoke
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --gres=gpu:h100:8
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/vast/home/ym56kacy/jinhe/MendelGM/wom/outputs/slurm/%x-%j.out
#
# SLURM launcher for MoE-Localized RL smoke tests.
# Usage:
#   sbatch slurm_smoke.sh full    # Full GSPO smoke test
#   sbatch slurm_smoke.sh moe     # MoE-only GSPO smoke test
#
# This script sets up the environment and calls the appropriate smoke script.

set -euo pipefail

MODE=${1:-full}
echo "[slurm_smoke] mode=$MODE"
echo "[slurm_smoke] node: $(hostname)"
echo "[slurm_smoke] SLURM_JOB_ID: $SLURM_JOB_ID"
echo "[slurm_smoke] gres: $(scontrol show job $SLURM_JOB_ID | grep GRES)"

# Environment setup
export VERL_LOGGING_LEVEL=INFO
export HF_HOME=/mnt/vast/home/ym56kacy/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_DEBUG=WARN
# Avoid issues with torch_distributed init
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

# Conda env
export PATH=/mnt/vast/home/ym56kacy/jinhe/conda_env/verl/bin:$PATH
source /mnt/vast/home/ym56kacy/jinhe/conda_env/verl/bin/activate 2>/dev/null || true

# verl repo root
VERL_ROOT=/mnt/vast/home/ym56kacy/jinhe/MendelGM/wom/verl
export PYTHONPATH=$VERL_ROOT:${PYTHONPATH:-}

cd "$VERL_ROOT"

# Create output dirs
mkdir -p /mnt/vast/home/ym56kacy/jinhe/MendelGM/wom/outputs/slurm
mkdir -p /mnt/vast/home/ym56kacy/jinhe/MendelGM/wom/outputs/smoke_full
mkdir -p /mnt/vast/home/ym56kacy/jinhe/MendelGM/wom/outputs/smoke_moe

# Record environment for reproducibility
{
  echo "=== Environment ==="
  echo "date: $(date -Iseconds)"
  echo "hostname: $(hostname)"
  echo "slurm_job_id: $SLURM_JOB_ID"
  echo "verl_commit: $(cd $VERL_ROOT && git rev-parse HEAD)"
  echo "verl_branch: $(cd $VERL_ROOT && git rev-parse --abbrev-ref HEAD)"
  echo "python: $(which python3)"
  echo "python_version: $(python3 --version)"
  echo "torch_version: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null)"
  echo "torch_cuda: $(python3 -c 'import torch; print(torch.version.cuda)' 2>/dev/null)"
  echo "transformers_version: $(python3 -c 'import transformers; print(transformers.__version__)' 2>/dev/null)"
  echo "vllm_version: $(python3 -c 'import vllm; print(vllm.__version__)' 2>/dev/null)"
  echo "ray_version: $(python3 -c 'import ray; print(ray.__version__)' 2>/dev/null)"
  echo "nvidia_smi:"
  nvidia-smi 2>&1 | head -20 || echo "nvidia-smi not available"
  echo "=== End Environment ==="
} > "/mnt/vast/home/ym56kacy/jinhe/MendelGM/wom/outputs/slurm/env_${SLURM_JOB_ID}.txt"

# Run the appropriate smoke script
case "$MODE" in
    full)
        bash experiments/moe_localized_rl/scripts/run_smoke_full.sh
        ;;
    moe)
        bash experiments/moe_localized_rl/scripts/run_smoke_moe.sh
        ;;
    *)
        echo "Unknown mode: $MODE. Use 'full' or 'moe'."
        exit 1
        ;;
esac
