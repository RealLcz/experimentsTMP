#!/usr/bin/env bash
#SBATCH --job-name=gemma4_pilot
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --gres=gpu:h200:8
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/vast/home/ym56kacy/jinhe/MendelGM/wom/outputs/slurm/%x-%j.out
#
# SLURM launcher for Gemma 4 26B-A4B GSPO pilot runs (8x H200, max 100 steps + val).
# Alternative 16x H100: sbatch --partition=all --nodes=2 --gres=gpu:h100:8 \
#   --export=ALL,NNODES=2,N_GPUS=8 --job-name=workingF slurm_gemma4_pilot.sh full
# Usage:
#   sbatch --job-name=workingF slurm_gemma4_pilot.sh full
#   sbatch --job-name=workingA slurm_gemma4_pilot.sh attention_only

set -euo pipefail

MODE=${1:-full}
echo "[slurm_gemma4_pilot] mode=$MODE"
echo "[slurm_gemma4_pilot] node: $(hostname)"
echo "[slurm_gemma4_pilot] SLURM_JOB_ID: $SLURM_JOB_ID"
echo "[slurm_gemma4_pilot] SLURM_JOB_NAME: ${SLURM_JOB_NAME:-unset}"

export VERL_LOGGING_LEVEL=INFO
export HF_HOME=/mnt/vast/home/ym56kacy/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0
export HYDRA_FULL_ERROR=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
# Keep vLLM's optional telemetry thread from writing into the quota-limited
# user cache.  A failed telemetry write is non-fatal, but otherwise emits a
# misleading background traceback during healthy training.
export VLLM_NO_USAGE_STATS=1
# A stale, manually installed deep_gemm_cpp extension in this shared Conda
# environment is ABI-incompatible with Torch 2.10. These BF16 Gemma4 runs use
# the explicitly selected Triton MoE backend and do not need DeepGEMM.
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
export VLLM_USE_DEEP_GEMM_E8M0=0
# This environment also serves SGLang and contains kernels for a different
# Triton ABI.  The repo-local triton_kernels shim selects vLLM's bundled copy.
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_USE_FLASHINFER_MOE_FP16=0
export VLLM_ALLREDUCE_USE_FLASHINFER=0
export NCCL_DEBUG=WARN
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

export PATH=/mnt/vast/home/ym56kacy/jinhe/conda_env/verl/bin:$PATH
export VERL_PYTHON=/mnt/vast/home/ym56kacy/jinhe/conda_env/verl/bin/python3
export RAY_START_SCRIPT=/mnt/vast/home/ym56kacy/jinhe/conda_env/verl/lib/python3.12/site-packages/ray/scripts/scripts.py
export CUDA_HOME=/mnt/vast/home/ym56kacy/jinhe/conda_env/verl
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}
export LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/targets/x86_64-linux/lib:${LIBRARY_PATH:-}
source /mnt/vast/home/ym56kacy/jinhe/conda_env/verl/bin/activate 2>/dev/null || true

# GPU cleanup: check for stale processes and report GPU state before training
echo "[slurm_gemma4_pilot] Checking GPU state on all allocated nodes..."
srun --overlap --ntasks-per-node=1 bash -c '
    # Report any stale GPU processes (do not kill - let SLURM handle allocation)
    echo "[$(hostname)] GPU state before training:"
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader 2>/dev/null
    echo "[$(hostname)] Stale GPU processes (if any):"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || echo "none"
' 2>&1
echo "[slurm_gemma4_pilot] GPU state check complete"

VERL_ROOT=/mnt/vast/home/ym56kacy/jinhe/MendelGM/wom/verl
export PYTHONPATH=$VERL_ROOT:${PYTHONPATH:-}
cd "$VERL_ROOT"

python3 - <<'PY'
import triton
import triton_kernels

print(f"[gemma4] triton={triton.__version__} triton_kernels={triton_kernels.__file__}")
if "vllm/third_party/triton_kernels" not in next(iter(triton_kernels.__path__)):
    raise RuntimeError("vLLM did not select its bundled Triton kernels")
from triton_kernels.matmul_ogs import PrecisionConfig  # noqa: F401
PY

mkdir -p /mnt/vast/home/ym56kacy/jinhe/MendelGM/wom/outputs/slurm
mkdir -p /mnt/vast/home/ym56kacy/jinhe/MendelGM/wom/outputs/workingF
mkdir -p /mnt/vast/home/ym56kacy/jinhe/MendelGM/wom/outputs/workingA

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
    print(f"[gemma4] transformers {current} already satisfies Gemma 4 requirement")
    raise SystemExit(0)

raise SystemExit(1)
PY
    if [[ $? -eq 0 ]]; then
        return 0
    fi
    echo "[gemma4] Upgrading transformers for Gemma 4 support..."
    python3 -m pip install -q -U 'transformers>=5.5.0,!=5.6.0'
}

ensure_transformers_for_gemma4

ensure_vllm_for_gemma4() {
    python3 <<'PY'
import importlib.metadata
from packaging import version

required = version.parse("0.19.0")
try:
    current = version.parse(importlib.metadata.version("vllm"))
except importlib.metadata.PackageNotFoundError:
    current = version.parse("0.0.0")

if current >= required:
    print(f"[gemma4] vllm {current} already satisfies Gemma 4 requirement")
    raise SystemExit(0)
raise SystemExit(1)
PY
    if [[ $? -eq 0 ]]; then
        return 0
    fi
    echo "[gemma4] Upgrading vLLM for Gemma 4 rollout support..."
    python3 -m pip install -q 'vllm==0.19.1'
}

ensure_vllm_for_gemma4

ensure_flash_attn_compatible() {
    if python3 -c "from flash_attn.ops.triton.rotary import apply_rotary" 2>/dev/null; then
        echo "[gemma4] flash_attn rotary import OK"
        return 0
    fi
    echo "[gemma4] Removing broken/incompatible flash_attn..."
    python3 -m pip uninstall -y flash-attn 2>/dev/null || true
    local site_pkgs
    site_pkgs=$(python3 -c "import site; print(site.getsitepackages()[0])")
    rm -rf "${site_pkgs}"/flash_attn* 2>/dev/null || true
    if python3 -c "import importlib.util; import sys; sys.exit(0 if importlib.util.find_spec('flash_attn') is None else 1)"; then
        echo "[gemma4] flash_attn removed; vLLM will use torch/triton backends"
    else
        echo "[gemma4] WARNING: flash_attn still present after cleanup"
    fi
}

ensure_flash_attn_compatible

python3 -m pip install -q colorama 2>/dev/null || true

# Short pilot: train at most 100 steps, then run validation on MATH test set.
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-100}
export TEST_FREQ=${TEST_FREQ:-100}
export RESUME_MODE=${RESUME_MODE:-disable}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
export SAVE_FREQ=${SAVE_FREQ:-50}
export N_GPUS=${N_GPUS:-${SLURM_GPUS_ON_NODE:-8}}
export NNODES=${NNODES:-${SLURM_NNODES:-1}}

echo "[slurm_gemma4_pilot] nnodes=$NNODES n_gpus_per_node=$N_GPUS"

setup_ray_slurm_cluster() {
    if [[ "${NNODES:-1}" -le 1 ]]; then
        return 0
    fi

    mapfile -t _ray_nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
    local head_node=${_ray_nodes[0]}
    local head_node_ip
    head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" bash -lc 'hostname --ip-address')
    # Prefer IPv4 when hostname returns multiple addresses (see verl ray_on_slurm.slurm).
    if [[ "$head_node_ip" == *" "* ]]; then
        IFS=' ' read -ra _ray_addrs <<<"$head_node_ip"
        if [[ ${#_ray_addrs[0]} -gt 16 ]]; then
            head_node_ip=${_ray_addrs[1]}
        else
            head_node_ip=${_ray_addrs[0]}
        fi
        echo "[slurm_gemma4_pilot] Resolved Ray head IP: $head_node_ip"
    fi

    local port=6379
    export RAY_ADDRESS="${head_node_ip}:${port}"
    export GEMMA4_RAY_HEAD_NODE="$head_node"
    local num_gpus=${SLURM_GPUS_ON_NODE:-${N_GPUS:-8}}
    local num_cpus=${SLURM_CPUS_PER_TASK:-64}

    echo "[slurm_gemma4_pilot] Ray head=$head_node address=$RAY_ADDRESS gpus_per_node=$num_gpus"
    srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
        "$VERL_PYTHON" "$RAY_START_SCRIPT" start --head --node-ip-address="$head_node_ip" --port="$port" \
        --num-cpus "$num_cpus" --num-gpus "$num_gpus" --disable-usage-stats --block &
    sleep 15

    for ((i = 1; i < ${#_ray_nodes[@]}; i++)); do
        local worker_node=${_ray_nodes[$i]}
        local worker_node_ip
        worker_node_ip=$(srun --overlap --nodes=1 --ntasks=1 -w "$worker_node" bash -lc 'hostname --ip-address')
        if [[ "$worker_node_ip" == *" "* ]]; then
            IFS=' ' read -ra _worker_addrs <<<"$worker_node_ip"
            if [[ ${#_worker_addrs[0]} -gt 16 ]]; then
                worker_node_ip=${_worker_addrs[1]}
            else
                worker_node_ip=${_worker_addrs[0]}
            fi
        fi
        echo "[slurm_gemma4_pilot] Ray worker on $worker_node ip=$worker_node_ip"
        srun --overlap --nodes=1 --ntasks=1 -w "$worker_node" \
            "$VERL_PYTHON" "$RAY_START_SCRIPT" start --address="$RAY_ADDRESS" \
            --node-ip-address="$worker_node_ip" \
            --num-cpus "$num_cpus" --num-gpus "$num_gpus" --disable-usage-stats --block &
        sleep 10
    done
    sleep 10
    srun --overlap --nodes=1 --ntasks=1 -w "$head_node" "$VERL_PYTHON" "$RAY_START_SCRIPT" status || true
}

run_pilot_mode() {
    local script=""
    case "$MODE" in
        full)
            script=run_pilot_gemma4_full.sh
            ;;
        attention_only|attention)
            script=run_pilot_gemma4_attention_only.sh
            ;;
        diagnostic|diag)
            script=run_diagnostic_gemma4_1step.sh
            ;;
        *)
            echo "Unknown mode: $MODE. Use 'full', 'attention_only', or 'diagnostic'."
            exit 1
            ;;
    esac

    if [[ "${NNODES:-1}" -gt 1 ]]; then
        srun --overlap --nodes=1 --ntasks=1 -w "$GEMMA4_RAY_HEAD_NODE" \
            bash "experiments/moe_localized_rl/scripts/${script}"
    else
        bash "experiments/moe_localized_rl/scripts/${script}"
    fi
}

{
  echo "=== Environment ==="
  echo "date: $(date -Iseconds)"
  echo "hostname: $(hostname)"
  echo "slurm_job_id: $SLURM_JOB_ID"
  echo "slurm_job_name: ${SLURM_JOB_NAME:-unset}"
  echo "mode: $MODE"
  echo "verl_commit: $(cd $VERL_ROOT && git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "python: $(which python3)"
  echo "python_version: $(python3 --version)"
  echo "torch_version: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null)"
  echo "transformers_version: $(python3 -c 'import transformers; print(transformers.__version__)' 2>/dev/null)"
  echo "vllm_version: $(python3 -c 'import vllm; print(vllm.__version__)' 2>/dev/null)"
  nvidia-smi 2>&1 | head -20 || true
  echo "=== End Environment ==="
} > "/mnt/vast/home/ym56kacy/jinhe/MendelGM/wom/outputs/slurm/env_gemma4_${SLURM_JOB_ID}.txt"

bash experiments/moe_localized_rl/scripts/download_gemma4_26b_a4b.sh

setup_ray_slurm_cluster
run_pilot_mode
