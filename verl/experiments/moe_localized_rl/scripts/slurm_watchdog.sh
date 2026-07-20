#!/usr/bin/env bash
#SBATCH --job-name=pilot_watchdog
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/vast/home/ym56kacy/jinhe/MendelGM/wom/outputs/slurm/pilot_watchdog-%j.out

set -euo pipefail
export INTERVAL=120
export MAX_RESUBMIT=9999
bash /mnt/vast/home/ym56kacy/jinhe/MendelGM/wom/verl/experiments/moe_localized_rl/tools/watchdog_pilot_v2.sh
