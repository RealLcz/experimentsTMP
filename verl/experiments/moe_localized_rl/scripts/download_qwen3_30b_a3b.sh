#!/usr/bin/env bash
# Download Qwen3-30B-A3B model from HuggingFace if not already present.
set -euo pipefail

MODEL_DIR=${MODEL_DIR:-/mnt/vast/home/ym56kacy/jinhe/MendelGM/wom/models/Qwen3-30B-A3B}
MODEL_ID="Qwen/Qwen3-30B-A3B"

if [[ -d "$MODEL_DIR" ]] && [[ -f "$MODEL_DIR/config.json" ]]; then
    # Verify at least one safetensors file exists
    if ls "$MODEL_DIR"/*.safetensors >/dev/null 2>&1; then
        echo "[download_qwen3] Model already present at $MODEL_DIR"
        exit 0
    fi
fi

echo "[download_qwen3] Downloading $MODEL_ID to $MODEL_DIR..."
mkdir -p "$MODEL_DIR"

export HF_HOME=/mnt/vast/home/ym56kacy/.cache/huggingface
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='$MODEL_ID',
    local_dir='$MODEL_DIR',
    local_dir_use_symlinks=False,
)
print('[download_qwen3] Download complete')
" || {
    echo "[download_qwen3] huggingface_hub download failed, trying huggingface-cli..."
    python3 -m huggingface_hub.commands.huggingface_cli download "$MODEL_ID" --local-dir "$MODEL_DIR"
}

echo "[download_qwen3] Model files:"
ls -la "$MODEL_DIR"/*.safetensors 2>/dev/null | head -5
echo "[download_qwen3] Done"
