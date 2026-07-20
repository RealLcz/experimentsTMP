#!/usr/bin/env bash
# Monitor Gemma4 diagnostic -> workingA (100-step attention-only) -> workingF (100-step full RSPO).
# Auto-submits the next stage when the previous one passes health checks.
set -uo pipefail

ROOT=/mnt/vast/home/ym56kacy/jinhe/MendelGM/wom
SLURM_DIR=$ROOT/outputs/slurm
SCRIPTS=$ROOT/verl/experiments/moe_localized_rl/scripts
CHECK=$ROOT/verl/experiments/moe_localized_rl/tools/check_gemma4_step1.sh
LOG=$ROOT/verl/experiments/moe_localized_rl/gemma4_pipeline_orchestrator.log
POLL_SEC=${POLL_SEC:-60}

DIAG_JOB=${DIAG_JOB:-}
WORKINGA_JOB=""
WORKINGF_JOB=""

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

job_running() {
    local jid=$1
    squeue -j "$jid" -h 2>/dev/null | grep -q .
}

latest_slurm_log() {
    local prefix=$1
    ls -t "$SLURM_DIR/${prefix}-"*.out 2>/dev/null | head -1
}

wait_for_job() {
    local jid=$1 label=$2
    while job_running "$jid"; do
        local f
        f=$(latest_slurm_log "$label")
        if [[ -f "$f" ]]; then
            local prog
            prog=$(rg -o "Training Progress: [^$]+" "$f" 2>/dev/null | tail -1 || true)
            [[ -n "$prog" ]] && log "$label job=$jid $prog"
        fi
        sleep "$POLL_SEC"
    done
    sacct -j "$jid" -n -X --format=State,ExitCode,Elapsed 2>/dev/null | head -1 | while read -r st ec el; do
        log "$label job=$jid finished state=$st exit=$ec elapsed=$el"
    done
}

job_failed() {
    local jid=$1
    local st ec
    read -r st ec _ <<< "$(sacct -j "$jid" -n -X --format=State,ExitCode --noheader 2>/dev/null | head -1 | tr '|' ' ')"
    [[ "$st" == "FAILED" || "$st" == "TIMEOUT" || "$st" == "CANCELLED" ]] && return 0
    [[ "${ec%%:*}" != "0" ]] && return 0
    return 1
}

log_has_fatal() {
    local f=$1
    rg -q "Error executing job|CUDA error: out of memory|Engine core initialization failed|RayTaskError|ActorDiedError" "$f" 2>/dev/null \
        && ! rg -q "step:1 |Training Progress:.*100/100" "$f" 2>/dev/null
}

pick_topology() {
    bash "$ROOT/verl/experiments/moe_localized_rl/tools/pick_gemma4_topology.sh"
}

submit_workingA() {
    if squeue -u "$USER" -n workingA -h 2>/dev/null | grep -q .; then
        log "workingA already in queue; skip submit"
        return 0
    fi
    local topo out
    topo=$(pick_topology)
    log "Submitting workingA (100-step attention-only, topology=$topo)..."
    if [[ "$topo" == h200 ]]; then
        out=$(cd "$SCRIPTS" && sbatch --job-name=workingA slurm_gemma4_pilot.sh attention_only 2>&1)
    else
        out=$(cd "$SCRIPTS" && sbatch --job-name=workingA slurm_gemma4_pilot_h100_16.sh attention_only 2>&1)
    fi
    log "$out"
    WORKINGA_JOB=$(awk '{print $NF}' <<< "$out")
}

submit_workingF() {
    if squeue -u "$USER" -n workingF -h 2>/dev/null | grep -q .; then
        log "workingF already in queue; skip submit"
        return 0
    fi
    local topo out
    topo=$(pick_topology)
    log "Submitting workingF (100-step full RSPO, topology=$topo)..."
    if [[ "$topo" == h200 ]]; then
        out=$(cd "$SCRIPTS" && sbatch --job-name=workingF slurm_gemma4_pilot.sh full 2>&1)
    else
        out=$(cd "$SCRIPTS" && sbatch --job-name=workingF slurm_gemma4_pilot_h100_16.sh full 2>&1)
    fi
    log "$out"
    WORKINGF_JOB=$(awk '{print $NF}' <<< "$out")
}

check_step1_health() {
    local f=$1
    if [[ ! -f "$CHECK" ]]; then
        log "WARN: missing $CHECK"
        return 1
    fi
    if bash "$CHECK" "$f"; then
        return 0
    fi
    return 3
}

check_pilot_complete() {
    local f=$1
    rg -q "Training Progress:.*100/100" "$f" 2>/dev/null
}

check_pilot_healthy() {
    local f=$1
    # Last logged step should not show broken-training signatures.
    local last_step
    last_step=$(rg "step:[0-9]+ " "$f" 2>/dev/null | tail -1 || true)
    [[ -z "$last_step" ]] && return 1
    python3 - "$last_step" <<'PY'
import re, sys
line = sys.argv[1]
m = dict(re.findall(r"([\w/]+):([-+0-9.eE]+)", line))
kl = float(m.get("rollout_corr/kl", 99))
grad = float(m.get("actor/grad_norm", 0))
reward = float(m.get("critic/rewards/mean", 0))
clip = float(m.get("response_length/clip_ratio", 1))
ok = kl < 1.0 and grad > 0 and clip < 0.98
print(f"last_step kl={kl} reward={reward} grad={grad} clip={clip} ok={ok}")
raise SystemExit(0 if ok else 1)
PY
}

log "=== gemma4 pipeline orchestrator start ==="

if [[ -z "$DIAG_JOB" ]]; then
    DIAG_JOB=$(squeue -u "$USER" -n gemma4_diag -h -o "%i" 2>/dev/null | head -1 || true)
fi
if [[ -z "$DIAG_JOB" ]]; then
    DIAG_JOB=$(ls -t "$SLURM_DIR"/gemma4_diag-*.out 2>/dev/null | head -1 | rg -o '[0-9]+(?=\.out$)' || true)
fi
log "Diagnostic job/log id: ${DIAG_JOB:-unknown}"

if [[ -n "$DIAG_JOB" ]] && job_running "$DIAG_JOB"; then
    wait_for_job "$DIAG_JOB" gemma4_diag
fi

DIAG_LOG=$(latest_slurm_log gemma4_diag)
if [[ -z "$DIAG_LOG" || ! -f "$DIAG_LOG" ]]; then
    log "ERROR: no diagnostic log found"
    exit 1
fi
log "Diagnostic log: $DIAG_LOG"

if job_failed "$DIAG_JOB" || log_has_fatal "$DIAG_LOG"; then
    log "ERROR: diagnostic failed — manual fix required. See $DIAG_LOG"
    exit 2
fi

if ! check_step1_health "$DIAG_LOG"; then
    log "ERROR: diagnostic step-1 health checks failed. See $DIAG_LOG"
    exit 3
fi
log "Diagnostic PASSED — launching workingA"

submit_workingA
if [[ -z "$WORKINGA_JOB" ]]; then
    WORKINGA_JOB=$(squeue -u "$USER" -n workingA -h -o "%i" 2>/dev/null | head -1 || true)
fi
[[ -n "$WORKINGA_JOB" ]] && wait_for_job "$WORKINGA_JOB" workingA

WA_LOG=$(latest_slurm_log workingA)
if [[ -z "$WA_LOG" || ! -f "$WA_LOG" ]]; then
    log "ERROR: workingA log missing"
    exit 4
fi

if job_failed "$WORKINGA_JOB" || log_has_fatal "$WA_LOG" || ! check_pilot_complete "$WA_LOG"; then
    log "ERROR: workingA did not complete 100 steps cleanly. See $WA_LOG"
    exit 5
fi
if ! check_pilot_healthy "$WA_LOG"; then
    log "WARN: workingA finished but last-step metrics look unhealthy — review $WA_LOG before full run"
fi
log "workingA COMPLETE — launching workingF"

submit_workingF
if [[ -z "$WORKINGF_JOB" ]]; then
    WORKINGF_JOB=$(squeue -u "$USER" -n workingF -h -o "%i" 2>/dev/null | head -1 || true)
fi
[[ -n "$WORKINGF_JOB" ]] && wait_for_job "$WORKINGF_JOB" workingF

WF_LOG=$(latest_slurm_log workingF)
if [[ -z "$WF_LOG" || ! -f "$WF_LOG" ]]; then
    log "ERROR: workingF log missing"
    exit 6
fi

if job_failed "$WORKINGF_JOB" || log_has_fatal "$WF_LOG" || ! check_pilot_complete "$WF_LOG"; then
    log "ERROR: workingF did not complete 100 steps cleanly. See $WF_LOG"
    exit 7
fi

log "=== PIPELINE COMPLETE: diagnostic + workingA + workingF all finished ==="
exit 0
