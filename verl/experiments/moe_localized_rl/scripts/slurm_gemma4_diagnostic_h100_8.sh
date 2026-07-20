#!/usr/bin/env bash
# DEPRECATED for Gemma4 8k-context runs — single H100 80GB colocated OOMs.
# Use slurm_gemma4_diagnostic_h200_8.sh or slurm_gemma4_diagnostic_h100_16.sh.
echo "[DEPRECATED] single H100 node — use H200 (1 node) or H100x2 (2 full nodes)" >&2
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_ROOT/slurm_gemma4_diagnostic.sh" auto
