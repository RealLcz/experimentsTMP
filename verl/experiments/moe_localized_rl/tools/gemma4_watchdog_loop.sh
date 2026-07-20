#!/usr/bin/env bash
# Continuous Gemma4 pipeline watchdog: monitor, auto-submit next stage, resubmit on failure.
set -uo pipefail

ROOT=/mnt/vast/home/ym56kacy/jinhe/MendelGM/wom
SLURM_DIR=$ROOT/outputs/slurm
SCRIPTS=$ROOT/verl/experiments/moe_localized_rl/scripts
CHECK=$ROOT/verl/experiments/moe_localized_rl/tools/check_gemma4_step1.sh
LOG=$ROOT/verl/experiments/moe_localized_rl/gemma4_watchdog.log
POLL_SEC=${POLL_SEC:-90}

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

latest_log() { ls -t "$SLURM_DIR/${1}-"*.out 2>/dev/null | head -1; }

job_running_name() { squeue -u "$USER" -n "$1" -h -o "%i" 2>/dev/null | head -1; }

job_failed() {
    local jid=$1
    [[ -z "$jid" ]] && return 1
    local st ec
    read -r st ec _ <<< "$(sacct -j "$jid" -n -X --format=State,ExitCode --noheader 2>/dev/null | head -1 | tr '|' ' ')"
    [[ "$st" == "FAILED" || "$st" == "TIMEOUT" || "$st" == "CANCELLED" ]] && return 0
    [[ "${ec%%:*}" != "0" ]] && return 0
    return 1
}

log_fatal() {
    local f=$1
    [[ ! -f "$f" ]] && return 1
    rg -q "Error executing job|CUDA error: out of memory|Engine core initialization failed|cuda:0 and cpu|RayTaskError|ActorDiedError" "$f" 2>/dev/null \
        && ! rg -q "step:1 |Training Progress:.*100/100" "$f" 2>/dev/null
}

diag_passed() {
    local f=$1
    [[ -f "$f" ]] && bash "$CHECK" "$f" >/dev/null 2>&1
}

pilot_done() {
    local f=$1
    [[ -f "$f" ]] && rg -q "Training Progress:.*100/100" "$f" 2>/dev/null && ! log_fatal "$f"
}

pick_topology() {
    bash "$ROOT/verl/experiments/moe_localized_rl/tools/pick_gemma4_topology.sh"
}

submit_diag() {
    local topo
    topo=$(pick_topology)
    log "Submitting gemma4_diag topology=$topo (H200 1-node or H100 2-node exclusive only)..."
    if [[ "$topo" == h200 ]]; then
        (cd "$SCRIPTS" && sbatch slurm_gemma4_diagnostic_h200_8.sh) 2>&1 | tee -a "$LOG"
    else
        (cd "$SCRIPTS" && sbatch slurm_gemma4_diagnostic_h100_16.sh) 2>&1 | tee -a "$LOG"
    fi
}

submit_workingA() {
    local topo
    topo=$(pick_topology)
    log "Submitting workingA topology=$topo..."
    if [[ "$topo" == h200 ]]; then
        (cd "$SCRIPTS" && sbatch --job-name=workingA slurm_gemma4_pilot.sh attention_only) 2>&1 | tee -a "$LOG"
    else
        (cd "$SCRIPTS" && sbatch --job-name=workingA slurm_gemma4_pilot_h100_16.sh attention_only) 2>&1 | tee -a "$LOG"
    fi
}

submit_workingF() {
    local topo
    topo=$(pick_topology)
    log "Submitting workingF topology=$topo..."
    if [[ "$topo" == h200 ]]; then
        (cd "$SCRIPTS" && sbatch --job-name=workingF slurm_gemma4_pilot.sh full) 2>&1 | tee -a "$LOG"
    else
        (cd "$SCRIPTS" && sbatch --job-name=workingF slurm_gemma4_pilot_h100_16.sh full) 2>&1 | tee -a "$LOG"
    fi
}

log "=== gemma4 watchdog start poll=${POLL_SEC}s ==="

while true; do
    diag_jid=$(job_running_name gemma4_diag || true)
    wa_jid=$(job_running_name workingA || true)
    wf_jid=$(job_running_name workingF || true)

    diag_log=$(latest_log gemma4_diag)
    wa_log=$(latest_log workingA)
    wf_log=$(latest_log workingF)

    if [[ -n "$wf_jid" ]]; then
        prog=$(rg -o "Training Progress: [^$]+" "$wf_log" 2>/dev/null | tail -1 || true)
        log "workingF RUNNING jid=$wf_jid ${prog:-starting}"
    elif pilot_done "$wa_log" && [[ -z "$wf_jid" ]]; then
        log "workingA complete — submit workingF"
        submit_workingF
    elif [[ -n "$wa_jid" ]]; then
        prog=$(rg -o "Training Progress: [^$]+" "$wa_log" 2>/dev/null | tail -1 || true)
        log "workingA RUNNING jid=$wa_jid ${prog:-starting}"
    elif diag_passed "$diag_log" && [[ -z "$wa_jid" ]]; then
        log "Diagnostic PASSED — submit workingA"
        submit_workingA
    elif [[ -n "$diag_jid" ]]; then
        prog=$(rg -o "Training Progress: [^$]+" "$diag_log" 2>/dev/null | tail -1 || true)
        step=$(rg "step:1 " "$diag_log" 2>/dev/null | tail -1 | rg -o "rollout_corr/kl:[^ ]+" || true)
        log "gemma4_diag RUNNING jid=$diag_jid ${prog:-init} ${step:-}"
    elif pilot_done "$wf_log"; then
        log "=== PIPELINE COMPLETE: diagnostic + workingA + workingF ==="
        exit 0
    elif [[ -f "$diag_log" ]] && diag_passed "$diag_log"; then
        log "Diagnostic passed (log only) — submit workingA"
        submit_workingA
    elif [[ -f "$diag_log" ]] && log_fatal "$diag_log"; then
        log "WARN: last diagnostic failed — will not auto-resubmit (watchdog submits on idle only)"
    elif [[ -z "$diag_jid" && -z "$wa_jid" && -z "$wf_jid" ]]; then
        if [[ ! -f "$diag_log" ]] || log_fatal "$diag_log"; then
            log "No active jobs — submit diagnostic"
            submit_diag
        fi
    fi

    sleep "$POLL_SEC"
done
