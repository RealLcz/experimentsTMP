#!/usr/bin/env bash
# DEPRECATED — fragmented 8x2 GPU topology causes GPU contention/OOM.
# Redirect to full 2-node H100 launcher.
echo "[DEPRECATED] slurm_gemma4_pilot_h100_16_fragmented.sh — use slurm_gemma4_pilot_h100_16.sh (2 full H100 nodes)" >&2
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_ROOT/slurm_gemma4_pilot_h100_16.sh" "${1:-full}"
