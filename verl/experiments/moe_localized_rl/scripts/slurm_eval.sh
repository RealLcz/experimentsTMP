#!/usr/bin/env bash
#SBATCH --job-name=pilot_v2_eval
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100:8
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/vast/home/ym56kacy/jinhe/MendelGM/wom/outputs/slurm/pilot_v2_eval-%j.out

set -euo pipefail
export VERL_LOGGING_LEVEL=INFO
bash /mnt/vast/home/ym56kacy/jinhe/MendelGM/wom/verl/experiments/moe_localized_rl/scripts/run_eval_pilot_v2.sh
