#!/usr/bin/env bash
# Watchdog: monitor pilot v2 jobs, auto-resubmit on failure.
# veRL resume_mode=auto picks up the latest global_step_* in OUTPUT_DIR.
set -euo pipefail

WOM_ROOT=/mnt/vast/home/ym56kacy/jinhe/MendelGM/wom
SCRIPTS=$WOM_ROOT/verl/experiments/moe_localized_rl/scripts
MANIFEST=$WOM_ROOT/verl/experiments/moe_localized_rl/pilot_v2_jobs.json
LOG=$WOM_ROOT/verl/experiments/moe_localized_rl/pilot_v2_watchdog.log
CLEANUP=$WOM_ROOT/verl/experiments/moe_localized_rl/tools/cleanup_old_ckpts.sh
INTERVAL=${INTERVAL:-120}
MAX_RESUBMIT=${MAX_RESUBMIT:-9999}

declare -A SLURM_MODE=( [full]=full [moe]=moe [attention_only]=attention_only )
declare -A SLURM_EXCLUDE=( [moe]=cn19 [full]=cn19 )

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

get_job_id() {
    python3 -c "import json; print(json.load(open('$MANIFEST')).get('$1',{}).get('job_id',''))" 2>/dev/null
}

set_job_id() {
    python3 -c "
import json
d=json.load(open('$MANIFEST'))
d['$1']['job_id']='$2'
d['$1']['resubmits']=d['$1'].get('resubmits',0)+($3)
json.dump(d, open('$MANIFEST','w'), indent=2)
"
}

find_latest_ckpt() {
    local out_dir=$1
    ls -d "$out_dir"/global_step_* 2>/dev/null | sort -t_ -k3 -n | tail -1 || true
}

cleanup_output_dir() {
    local mode=$1
    local out_sub
    out_sub=$(python3 -c "import json; print(json.load(open('$MANIFEST'))['$mode']['output_dir'])")
    bash "$CLEANUP" "$WOM_ROOT/outputs/$out_sub" 1
}

resubmit() {
    local mode=$1
    local slurm_mode=${SLURM_MODE[$mode]}
    local exclude=${SLURM_EXCLUDE[$mode]:-}
    local out_dir
    out_dir=$(python3 -c "import json; print(json.load(open('$MANIFEST'))['$mode']['output_dir'])")
    out_dir=$WOM_ROOT/outputs/$out_dir
    local ckpt
    ckpt=$(find_latest_ckpt "$out_dir")
    local extra=""
    [[ -n "$exclude" ]] && extra="--exclude=$exclude"
    local new_job
    new_job=$(cd "$SCRIPTS" && sbatch --parsable $extra slurm_pilot.sh "$slurm_mode")
    local count
    count=$(python3 -c "import json; print(json.load(open('$MANIFEST'))['$mode'].get('resubmits',0)+1)")
    set_job_id "$mode" "$new_job" 1
    if [[ -n "$ckpt" ]]; then
        log "RESUBMIT $mode -> job $new_job (resume from $(basename "$ckpt"), attempt $count)"
    else
        log "RESUBMIT $mode -> job $new_job (from scratch, attempt $count)"
    fi
    cleanup_output_dir "$mode"
}

job_state() {
    local job_id=$1
    if squeue -j "$job_id" -h 2>/dev/null | grep -q .; then
        squeue -j "$job_id" -h -o "%T" 2>/dev/null | head -1
        return
    fi
    sacct -j "$job_id" --format=State -P -n 2>/dev/null | head -1 || echo "UNKNOWN"
}

check_mode() {
    local mode=$1
    local job_id
    job_id=$(get_job_id "$mode")
    [[ -z "$job_id" ]] && { resubmit "$mode"; return; }

    local state
    state=$(job_state "$job_id")

    case "$state" in
        RUNNING|PENDING|CONFIGURING|COMPLETING|REQUEUED|SUSPENDED)
            return 0
            ;;
        COMPLETED)
            log "DONE $mode job $job_id COMPLETED"
            return 0
            ;;
        FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED|UNKNOWN)
            local count
            count=$(python3 -c "import json; print(json.load(open('$MANIFEST'))['$mode'].get('resubmits',0))")
            if [[ "$count" -ge "$MAX_RESUBMIT" ]]; then
                log "ALERT $mode job $job_id state=$state — max resubmits ($MAX_RESUBMIT) reached"
                return 1
            fi
            log "FAIL $mode job $job_id state=$state"
            resubmit "$mode"
            ;;
    esac
}

log "Watchdog started (interval=${INTERVAL}s, max_resubmit=$MAX_RESUBMIT)"
log "Checkpoint policy: save_freq=10, max_actor_ckpt_to_keep=1, resume_mode=auto"

while true; do
    any_running=0
    all_done=1
    for mode in full moe attention_only; do
        job_id=$(get_job_id "$mode")
        state=$(job_state "$job_id")
        out_sub=$(python3 -c "import json; print(json.load(open('$MANIFEST'))['$mode']['output_dir'])")
        ckpt=$(find_latest_ckpt "$WOM_ROOT/outputs/$out_sub")
        prog=""
        if [[ -f "$WOM_ROOT/outputs/slurm/olmoe_gspo_pilot-${job_id}.out" ]]; then
            prog=$(grep -oP 'Training Progress: \K[^$]+' "$WOM_ROOT/outputs/slurm/olmoe_gspo_pilot-${job_id}.out" 2>/dev/null | tail -1 || true)
        fi
        log "[$mode] job=$job_id state=$state ckpt=${ckpt:-none} progress=${prog:-n/a}"
        [[ "$state" == "RUNNING" || "$state" == "PENDING" ]] && any_running=1
        [[ "$state" != "COMPLETED" ]] && all_done=0
        check_mode "$mode" || true
        cleanup_output_dir "$mode" 2>/dev/null || true
    done

    if [[ "$all_done" -eq 1 ]]; then
        log "ALL THREE EXPERIMENTS COMPLETED. Watchdog exiting."
        exit 0
    fi
    sleep "$INTERVAL"
done
