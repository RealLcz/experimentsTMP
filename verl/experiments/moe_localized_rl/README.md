# MoE-Localized RL Pilot: OLMoE GSPO Experiment

## Overview

This experiment investigates whether **only updating MoE experts + router** (while freezing attention, norms, embedding, and LM head) can retain most of the RL improvement achieved by full-parameter GSPO training.

```
OLMoE-1B-7B-0125-Instruct
        +
       GSPO
        |
        +-------------------+
        |                   |
   Full-GSPO          MoE-only-GSPO
        |                   |
 Update all params    Update experts + router
```

## Setup

- **Model**: `allenai/OLMoE-1B-7B-0125-Instruct` (6.92B total params, ~1B active)
- **Algorithm**: GSPO (sequence-level importance ratio) + GRPO advantage
- **Data**: GSM8K (7473 train, 1319 test)
- **Infrastructure**: veRL 0.9.0.dev (commit `1bfd02c4`), FSDP2, vLLM rollout
- **Hardware**: 8x H100 per run

## Parameter Categories (OLMoE)

| Category    | Parameters    | % of total | Full | MoE-only | Experts-only |
|-------------|---------------|-----------|------|----------|-------------|
| Attention   | 268M          | 3.9%      | Train | Frozen  | Frozen      |
| Experts     | 6.44B         | 93.1%     | Train | Train   | Train       |
| Router      | 2.1M          | 0.03%     | Train | Train   | Frozen      |
| Norm        | 67K           | 0.001%    | Train | Frozen  | Frozen      |
| Embedding   | 103M          | 1.5%      | Train | Frozen  | Frozen      |
| LM Head     | 103M          | 1.5%      | Train | Frozen  | Frozen      |

## Files

```
experiments/moe_localized_rl/
├── README.md                          # this file
├── configs/
│   ├── olmoe_full_gspo.yaml           # Full GSPO config
│   ├── olmoe_moe_gspo.yaml            # MoE-only GSPO config (core experiment)
│   └── olmoe_experts_only_gspo.yaml   # Experts-only config (Run C, optional)
├── scripts/
│   ├── run_smoke_full.sh              # Smoke test (Full)
│   ├── run_smoke_moe.sh               # Smoke test (MoE-only)
│   ├── run_pilot_full.sh              # Pilot run (Full)
│   ├── run_pilot_moe.sh               # Pilot run (MoE-only)
│   ├── slurm_smoke.sh                 # SLURM launcher for smoke tests
│   └── slurm_pilot.sh                 # SLURM launcher for pilot runs
└── tools/
    ├── inspect_model_modules.py       # Inspect OLMoE module structure
    ├── verify_trainable_params.py     # Verify trainable parameter distribution
    └── compare_checkpoints.py         # Checkpoint delta comparison
```

## Core veRL Modifications

The modifications are minimal and config-driven (see `IMPLEMENTATION_MAP.md` for details):

1. **`verl/utils/parameter_update_policy.py`** (NEW): `apply_parameter_update_policy(model, mode, strict)` utility.
2. **`verl/workers/config/actor.py`** (MODIFIED): Added `ParameterUpdateConfig` dataclass + field on `ActorConfig`.
3. **`verl/trainer/config/actor/actor.yaml`** (MODIFIED): Added `parameter_update:` YAML block (default `mode: full`).
4. **`verl/workers/engine/fsdp/transformer_impl.py`** (MODIFIED): 
   - Insert `_apply_parameter_update_policy(module)` in `_build_model_optimizer` after `_build_module`, before `_build_fsdp_module`.
   - Filter `p for p in module.parameters() if p.requires_grad` in `_build_optimizer`.
5. **`verl/workers/engine_workers.py`** (MODIFIED): Pass `parameter_update_config` from `ActorConfig` to `TrainingWorkerConfig` to `FSDPEngine`.
6. **`verl/workers/config/engine.py`** (MODIFIED): Added `parameter_update_config` field to `TrainingWorkerConfig`.

**Backward compatibility**: Default `mode: full` is a no-op - vanilla veRL behavior is preserved.

## Running

### Smoke test (2-5 optimizer steps, ~50 min)

```bash
sbatch scripts/slurm_smoke.sh full   # Full GSPO
sbatch scripts/slurm_smoke.sh moe    # MoE-only GSPO
```

### Pilot run (full experiment)

```bash
sbatch scripts/slurm_pilot.sh full   # Full GSPO
sbatch scripts/slurm_pilot.sh moe    # MoE-only GSPO
```

## Correctness Validation

Three layers of validation (per the programming guide):

1. **`requires_grad` check**: Unit tests verify the correct params are trainable/frozen.
2. **Gradient check**: Backward pass populates gradients only on trainable params.
3. **Checkpoint delta check**: After one optimizer step, frozen params are bitwise identical.

Run the tests:
```bash
python -m pytest tests/models/test_parameter_update_policy.py -v
python -m pytest tests/models/test_checkpoint_delta.py -v
```

## Research Question (RQ1)

> When Full GSPO can learn, does only updating pretrained native MoE experts + router also produce significant RL improvement?

Metric: **MoE RL Retention Ratio**
```
          MoE final - Base
rho = -------------------------
          Full final - Base
```

- `rho ≈ 1` → Positive signal → scale to Qwen3-30B-A3B
- `rho ≈ 0` → Hypothesis fails → study Freeze-Attention / MoE+Attention-LoRA
