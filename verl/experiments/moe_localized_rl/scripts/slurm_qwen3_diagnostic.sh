#!/usr/bin/env bash
# Submit Qwen3 diagnostic on an allowed full-node topology only:
#   1x H200 node (8 GPU)  - preferred
#   2x H100 nodes (16 GPU) - fallback if H200 queue is empty
#
# Usage:
#   sbatch slurm_qwen3_diagnostic.sh          # auto
#   sbatch slurm_qwen3_diagnostic.sh h200
#   sbatch slurm_qwen3_diagnostic.sh h100_16

set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS="$SCRIPT_ROOT/../tools/pick_gemma4_topology.sh"
MODE=${1:-auto}

submit_h200() {
    sbatch "$SCRIPT_ROOT/slurm_qwen3_diagnostic_h200_8.sh"
}

submit_h100_16() {
    sbatch "$SCRIPT_ROOT/slurm_qwen3_diagnostic_h100_16.sh"
}

case "$MODE" in
    h200)
        submit_h200
        ;;
    h100_16|h100)
        submit_h100_16
        ;;
    auto)
        topo=$("$TOOLS")
        "$TOOLS" --explain
        if [[ "$topo" == h200 ]]; then
            echo "[slurm_qwen3_diagnostic] auto -> 1x H200 exclusive"
            submit_h200
        else
            echo "[slurm_qwen3_diagnostic] auto -> 2x H100 exclusive"
            submit_h100_16
        fi
        ;;
    *)
        echo "Usage: $0 [auto|h200|h100_16]"
        exit 1
        ;;
esac
