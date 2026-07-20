#!/usr/bin/env bash
# Monitor pilot v2 experiments: job status, training progress, disk usage.
set -euo pipefail

WOM_ROOT=/mnt/vast/home/ym56kacy/jinhe/MendelGM/wom
SLURM_DIR=$WOM_ROOT/outputs/slurm
REPORT=$WOM_ROOT/verl/experiments/moe_localized_rl/pilot_v2_monitor.json
JOBS_MANIFEST=$WOM_ROOT/verl/experiments/moe_localized_rl/pilot_v2_jobs.json

declare -A MODES=(
    [full]="olmoe_gspo_full_math_seed0|pilot_v2_full"
    [moe]="olmoe_gspo_moe_math_seed0|pilot_v2_moe"
    [attention_only]="olmoe_gspo_attention_only_math_seed0|pilot_v2_attention_only"
)

get_job_id() {
    local mode=$1
    if [[ -f "$JOBS_MANIFEST" ]]; then
        python3 -c "import json; d=json.load(open('$JOBS_MANIFEST')); print(d.get('$mode',{}).get('job_id',''))" 2>/dev/null || true
    fi
}

timestamp=$(date -Iseconds)
disk_use=$(df -h /mnt/vast | awk 'NR==2 {print $5}')
disk_avail=$(df -h /mnt/vast | awk 'NR==2 {print $4}')

echo "============================================================"
echo "Pilot v2 Monitor — $timestamp"
echo "Disk /mnt/vast: used=$disk_use avail=$disk_avail"
echo "============================================================"

json_modes="{"
first=true

for mode in full moe attention_only; do
    IFS='|' read -r exp_name out_dir <<< "${MODES[$mode]}"
    out_path=$WOM_ROOT/outputs/$out_dir

    # Job ID: prefer manifest, fallback to log search
    latest_job=$(get_job_id "$mode")
    latest_log=""
    if [[ -n "$latest_job" ]]; then
        latest_log="$SLURM_DIR/olmoe_gspo_pilot-${latest_job}.out"
        [[ -f "$latest_log" ]] || latest_log=""
    fi
    if [[ -z "$latest_job" ]]; then
        for f in $(ls -t "$SLURM_DIR"/olmoe_gspo_pilot-*.out 2>/dev/null); do
            if grep -q "$exp_name" "$f" 2>/dev/null; then
                latest_log=$f
                latest_job=$(basename "$f" | sed 's/olmoe_gspo_pilot-//;s/.out//')
                break
            fi
        done
    fi

    # Job state from sacct
    state="NOT_SUBMITTED"
    elapsed="-"
    if [[ -n "$latest_job" ]]; then
        sacct_line=$(sacct -j "$latest_job" --format=State,Elapsed -P -n 2>/dev/null | head -1 || true)
        if [[ -n "$sacct_line" ]]; then
            state=$(echo "$sacct_line" | cut -d'|' -f1)
            elapsed=$(echo "$sacct_line" | cut -d'|' -f2)
        fi
    fi

    # Training progress from log
    progress="N/A"
    if [[ -n "$latest_log" && -f "$latest_log" ]]; then
        progress=$(grep -oP 'Training Progress: \K[^$]+' "$latest_log" 2>/dev/null | tail -1 || echo "init/unknown")
    fi

    # Checkpoint disk usage
    ckpt_size="0"
    ckpt_count=0
    if [[ -d "$out_path" ]]; then
        ckpt_count=$(find "$out_path" -maxdepth 2 -name "global_step_*" -type d 2>/dev/null | wc -l)
        ckpt_size=$(du -sh "$out_path" 2>/dev/null | awk '{print $1}')
    fi

    # Error check
    error=""
    if [[ -n "$latest_log" && -f "$latest_log" ]]; then
        if grep -qE "Error executing job|RayTaskError|RuntimeError: Engine core" "$latest_log" 2>/dev/null; then
            error=$(grep -E "Error executing job|RuntimeError:" "$latest_log" 2>/dev/null | tail -1 | cut -c1-120)
        fi
    fi

    echo ""
    echo "[$mode] experiment=$exp_name"
    echo "  job_id:    ${latest_job:-none}"
    echo "  state:     $state (elapsed=$elapsed)"
    echo "  progress:  $progress"
    echo "  output:    $out_path ($ckpt_size, $ckpt_count checkpoints)"
    if [[ -n "$error" ]]; then
        echo "  ERROR:     $error"
    fi

    if $first; then first=false; else json_modes+=","; fi
    json_modes+="\"$mode\":{\"job_id\":\"${latest_job:-}\",\"state\":\"$state\",\"elapsed\":\"$elapsed\",\"progress\":\"$progress\",\"output_dir\":\"$out_path\",\"disk\":\"$ckpt_size\",\"checkpoints\":$ckpt_count}"
done

json_modes+="}"
mkdir -p "$(dirname "$REPORT")"
cat > "$REPORT" <<EOF
{"timestamp":"$timestamp","disk_used":"$disk_use","disk_avail":"$disk_avail","experiments":$json_modes}
EOF

echo ""
echo "Report saved: $REPORT"

# Disk budget warning (3 experiments × 2 checkpoints × ~14GB model-only ≈ 84GB)
total_pilot=$(du -sh "$WOM_ROOT/outputs/pilot_v2_"* 2>/dev/null | awk '{s+=$1} END {print s}' || echo "0")
echo ""
echo "Checkpoint policy: save_freq=10, max_actor_ckpt_to_keep=1, resume_mode=auto"
