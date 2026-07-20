#!/usr/bin/env bash
# Gemma 4 26B-A4B Pilot: Attention-only GSPO + Router Shift (RSPO) on MATH.
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-gemma4_26b_a4b_rspo_attention_only_math_seed0}
export OUTPUT_DIR=${OUTPUT_DIR:-/mnt/vast/home/ym56kacy/jinhe/MendelGM/wom/outputs/workingA}
export PARAM_MODE=${PARAM_MODE:-attention_only}

bash "$SCRIPT_DIR/run_pilot_gemma4_full.sh" "$@"
