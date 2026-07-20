#!/usr/bin/env bash
#SBATCH --job-name=workingA
#SBATCH --partition=all
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --gres=gpu:h100:8
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/vast/home/ym56kacy/jinhe/MendelGM/wom/outputs/slurm/%x-%j.out
#
# Gemma 4 26B-A4B RSPO pilot on 16x H100 (2 nodes x 8 GPUs).
# Usage:
#   sbatch --job-name=workingA slurm_gemma4_pilot_h100_16.sh attention_only
#   sbatch --job-name=workingF slurm_gemma4_pilot_h100_16.sh full

set -euo pipefail

export NNODES=2
export N_GPUS=8
export ROLLOUT_TP=8
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.40}

SCRIPT_ROOT=/mnt/vast/home/ym56kacy/jinhe/MendelGM/wom/verl/experiments/moe_localized_rl/scripts
exec bash "$SCRIPT_ROOT/slurm_gemma4_pilot.sh" "${1:-attention_only}"
