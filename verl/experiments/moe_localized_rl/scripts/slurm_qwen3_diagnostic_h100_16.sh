#!/usr/bin/env bash
#SBATCH --job-name=qwen3_diag
#SBATCH --partition=all
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --gres=gpu:h100:8
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/vast/home/ym56kacy/jinhe/MendelGM/wom/outputs/slurm/%x-%j.out
#
# 1-step diagnostic on 2 full H100 nodes (16 GPU, exclusive).

set -euo pipefail

export NNODES=2
export N_GPUS=8
export ROLLOUT_TP=8
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.35}

SCRIPT_ROOT=/mnt/vast/home/ym56kacy/jinhe/MendelGM/wom/verl/experiments/moe_localized_rl/scripts
exec bash "$SCRIPT_ROOT/slurm_qwen3_pilot.sh" diagnostic
