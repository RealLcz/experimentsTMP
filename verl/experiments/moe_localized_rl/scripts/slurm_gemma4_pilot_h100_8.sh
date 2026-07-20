#!/usr/bin/env bash
# DEPRECATED for Gemma4 8k-context runs — use H200 or 2x H100 full nodes.
echo "[DEPRECATED] single H100 node — use slurm_gemma4_pilot.sh (H200) or slurm_gemma4_pilot_h100_16.sh" >&2
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_ROOT/slurm_gemma4_pilot_h100_16.sh" "$@"
