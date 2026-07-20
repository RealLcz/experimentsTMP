#!/usr/bin/env bash
# Keep only the latest global_step_* checkpoint in an output directory.
set -euo pipefail

OUT_DIR=${1:?usage: cleanup_old_ckpts.sh <output_dir>}
KEEP=${2:-1}

mapfile -t STEPS < <(ls -d "$OUT_DIR"/global_step_* 2>/dev/null | while read -r p; do
    step=$(basename "$p" | sed 's/global_step_//')
    echo "$step $p"
done | sort -n -k1 | awk '{print $2}')

TOTAL=${#STEPS[@]}
if [[ "$TOTAL" -le "$KEEP" ]]; then
    echo "[cleanup] $OUT_DIR: $TOTAL ckpt(s), nothing to remove"
    exit 0
fi

REMOVE_COUNT=$((TOTAL - KEEP))
for ((i=0; i<REMOVE_COUNT; i++)); do
    rm -rf "${STEPS[$i]}"
    echo "[cleanup] removed ${STEPS[$i]}"
done
LATEST="${STEPS[$((TOTAL-1))]}"
LATEST_STEP=$(basename "$LATEST" | sed 's/global_step_//')
echo "$LATEST_STEP" > "$OUT_DIR/latest_checkpointed_iteration.txt"
echo "[cleanup] $OUT_DIR: kept $LATEST (tracker=$LATEST_STEP, removed $REMOVE_COUNT)"
