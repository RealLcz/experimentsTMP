#!/usr/bin/env bash
# Download google/gemma-4-26B-A4B-it into the local models/ directory.
set -euo pipefail

SOURCE_MODEL_ID=${SOURCE_MODEL_ID:-google/gemma-4-26B-A4B-it}
MODEL_DIR=${MODEL_DIR:-/mnt/vast/home/ym56kacy/jinhe/MendelGM/wom/models/gemma-4-26B-A4B-it}

if [[ -f "${MODEL_DIR}/config.json" ]]; then
    echo "[download_gemma4] Model already present at ${MODEL_DIR}"
    exit 0
fi

export HF_HOME=${HF_HOME:-/mnt/vast/home/ym56kacy/.cache/huggingface}
mkdir -p "$(dirname "$MODEL_DIR")"

ensure_transformers_for_gemma4() {
    python3 <<'PY'
import importlib.metadata
from packaging import version

required = version.parse("5.5.0")
blocked = version.parse("5.6.0")
try:
    current = version.parse(importlib.metadata.version("transformers"))
except importlib.metadata.PackageNotFoundError:
    current = version.parse("0.0.0")

if current >= required and current != blocked:
    print(f"[download_gemma4] transformers {current} already OK")
    raise SystemExit(0)
raise SystemExit(1)
PY
    if [[ $? -eq 0 ]]; then
        return 0
    fi
    echo "[download_gemma4] Installing transformers>=5.5.0 (required for Gemma 4)..."
    python3 -m pip install -q -U 'transformers>=5.5.0,!=5.6.0'
}

ensure_transformers_for_gemma4

echo "[download_gemma4] Downloading ${SOURCE_MODEL_ID} -> ${MODEL_DIR}"
python3 <<PY
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="${SOURCE_MODEL_ID}",
    local_dir="${MODEL_DIR}",
    local_dir_use_symlinks=False,
)
print("Download complete:", "${MODEL_DIR}")
PY
