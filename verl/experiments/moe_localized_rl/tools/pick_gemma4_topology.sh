#!/usr/bin/env bash
# Pick Gemma4 SLURM topology from actually schedulable full-node resources:
#   h200     — 1 exclusive H200 node (8 GPU), when an idle non-drained node exists
#   h100_16  — 2 exclusive H100 nodes (16 GPU), otherwise if >=2 idle H100 nodes
#   h100_16  — fallback when H200 is busy/drained but Slurm can place 2-node jobs
#
# Usage: pick_gemma4_topology.sh [--explain]
set -euo pipefail

EXPLAIN=${1:-}

h200_idle_nodes() {
    sinfo -p h200 -N -h -o "%N %t" 2>/dev/null | awk '$2 == "idle" { print $1 }'
}

h100_idle_nodes() {
    sinfo -p all -N -h -o "%N %t" 2>/dev/null | awk '$2 == "idle" { print $1 }'
}

count_lines() {
    [[ -z "${1// }" ]] && echo 0 && return
    wc -l <<< "$1" | tr -d ' '
}

h200_idle=$(h200_idle_nodes || true)
h100_idle=$(h100_idle_nodes || true)
n_h200_idle=$(count_lines "$h200_idle")
n_h100_idle=$(count_lines "$h100_idle")

pick=""
reason=""

if [[ "$n_h200_idle" -ge 1 ]]; then
    pick=h200
    reason="H200 idle node(s): $(echo "$h200_idle" | tr '\n' ' ' | sed 's/ $//')"
elif [[ "$n_h100_idle" -ge 2 ]]; then
    pick=h100_16
    reason=">=2 idle H100 nodes: $(echo "$h100_idle" | head -2 | tr '\n' ' ' | sed 's/ $//')"
else
    # H200 partition up but no idle node (mixed/drain) — use 2xH100; Slurm queues until 2 nodes free.
    h200_state=$(sinfo -p h200 -N -h -o "%N %t %E" 2>/dev/null | tr '\n' '; ')
    pick=h100_16
    reason="no idle H200 (state: ${h200_state:-unknown}); submit 2xH100 exclusive"
fi

if [[ "$EXPLAIN" == "--explain" ]]; then
    echo "topology=$pick"
    echo "reason=$reason"
    echo "h200_idle_count=$n_h200_idle"
    echo "h100_idle_count=$n_h100_idle"
else
    echo "$pick"
fi
