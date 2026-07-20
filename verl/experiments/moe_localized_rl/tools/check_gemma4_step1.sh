#!/usr/bin/env bash
# Parse a Gemma4 pilot/diagnostic slurm log and report step-1 health signals.
set -euo pipefail

LOG=${1:-}
if [[ -z "$LOG" || ! -f "$LOG" ]]; then
    echo "Usage: $0 /path/to/slurm.log"
    exit 1
fi

echo "=== config ==="
rg -n "use_remove_padding" "$LOG" | head -5 || true

echo "=== step 1 metrics ==="
STEP_LINE=$(rg "step:1 " "$LOG" | tail -1 || true)
if [[ -z "$STEP_LINE" ]]; then
    echo "FAIL: no step:1 line found"
    exit 2
fi
echo "$STEP_LINE"

python3 - "$LOG" "$STEP_LINE" <<'PY'
import re
import sys

log_path, line = sys.argv[1], sys.argv[2]
metrics = dict(re.findall(r"([\w/]+):([-+0-9.eE]+)", line))

def f(key, default=0.0):
    try:
        return float(metrics.get(key, default))
    except ValueError:
        return default

kl = f("rollout_corr/kl")
reward = f("critic/rewards/mean")
grad = f("actor/grad_norm")
clip = f("response_length/clip_ratio")

with open(log_path, "r", encoding="utf-8", errors="ignore") as fh:
    log_text = fh.read()
bad_padding = "use_remove_padding=True" in log_text and "use_remove_padding=False" not in log_text

checks = [
    ("rollout_corr/kl < 0.5", kl < 0.5, kl),
    ("critic/rewards/mean > 0", reward > 0, reward),
    ("actor/grad_norm > 0", grad > 0, grad),
    ("response_length/clip_ratio < 0.95", clip < 0.95, clip),
    ("log shows use_remove_padding=False", not bad_padding, int(not bad_padding)),
]

print("=== health checks ===")
ok = True
for name, passed, value in checks:
    status = "PASS" if passed else "FAIL"
    print(f"{status}: {name} (value={value})")
    ok = ok and passed

raise SystemExit(0 if ok else 3)
PY
