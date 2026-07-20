#!/usr/bin/env bash
# Monitor workingF / workingA Gemma4 pilot jobs until both complete or fail.
set -uo pipefail

SLURM_DIR=/mnt/vast/home/ym56kacy/jinhe/MendelGM/wom/outputs/slurm
LOG=/mnt/vast/home/ym56kacy/jinhe/MendelGM/wom/verl/experiments/moe_localized_rl/gemma4_pilot_monitor.log
POLL_SEC=${POLL_SEC:-90}
MAX_POLLS=${MAX_POLLS:-480}

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

latest_log() {
    local name=$1
    ls -t "$SLURM_DIR/${name}-"*.out 2>/dev/null | head -1
}

job_status() {
    local name=$1
    squeue -u "$USER" -n "$name" -h -o "%T %M %D %R" 2>/dev/null | head -1
}

scan_log() {
    local f=$1
    [[ -f "$f" ]] || { echo "no_log"; return; }
    if rg -q "RayTaskError|ActorDiedError|OutOfMemoryError|CUDA out of memory|Traceback \(most recent" "$f" 2>/dev/null \
        && ! rg -q "Training Progress:.*100/100" "$f" 2>/dev/null; then
        echo "error"
        return
    fi
    if rg -q "Training Progress:" "$f" 2>/dev/null; then
        rg -o "Training Progress: [^$]+" "$f" 2>/dev/null | tail -1
        return
    fi
    if rg -q "Total training steps: 100" "$f" 2>/dev/null; then
        echo "training_init"
        return
    fi
    echo "starting"
}

poll_one() {
    local name=$1
    local st
    st=$(job_status "$name")
    local f
    f=$(latest_log "$name")
    local prog
    prog=$(scan_log "$f")
    if [[ -n "$st" ]]; then
        log "$name RUNNING queue=[$st] log=$(basename "${f:-none}) prog=$prog"
    else
        local sacct_st
        sacct_st=$(sacct -u "$USER" -n --name="$name" --format=State -P 2>/dev/null | head -1)
        log "$name NOT_IN_QUEUE sacct=${sacct_st:-unknown} log=$(basename "${f:-none}) prog=$prog"
    fi
}

log "=== gemma4 pilot monitor start (poll=${POLL_SEC}s) ==="
for i in $(seq 1 "$MAX_POLLS"); do
    log "--- poll $i ---"
    poll_one workingF
    poll_one workingA

    f_done=0; a_done=0
    squeue -u "$USER" -n workingF -h >/dev/null 2>&1 || f_done=1
    squeue -u "$USER" -n workingA -h >/dev/null 2>&1 || a_done=1

    if [[ $f_done -eq 1 && $a_done -eq 1 ]]; then
        for name in workingF workingA; do
            f=$(latest_log "$name")
            if rg -q "val/.*accuracy|validation|test_score" "$f" 2>/dev/null; then
                log "$name finished with validation metrics in log"
            elif rg -q "Training Progress:.*100/100" "$f" 2>/dev/null; then
                log "$name reached step 100"
            elif rg -q "RayTaskError|ActorDiedError|OOM" "$f" 2>/dev/null; then
                log "WARNING $name may have failed — check $f"
            else
                log "$name exited — check $f"
            fi
        done
        log "=== both jobs left queue ==="
        break
    fi
    sleep "$POLL_SEC"
done
