# IMPLEMENTATION_MAP.md

## veRL Checkout

- **Path**: `/mnt/vast/home/ym56kacy/jinhe/MendelGM/wom/verl`
- **Repo**: volcengine/verl (fresh clone)
- **Version**: 0.9.0.dev
- **Commit**: `1bfd02c4359a44310759aab7c0382e2bf5836756`

## Model Architecture (OLMoE)

`allenai/OLMoE-1B-7B-0125-Instruct` loads via HF `AutoModelForCausalLM` (architecture `OlmoeForCausalLM`). Per-layer `OlmoeDecoderLayer` contains:
- `self_attn`: `OlmoeAttention`
- `mlp`: `OlmoeSparseMoeBlock` containing `gate` (`OlmoeTopKRouter`) and `experts` (`OlmoeExperts`)
- `input_layernorm`, `post_attention_layernorm`: `OlmoeRMSNorm`
- `model.embed_tokens`: embedding
- `lm_head`: LM head

OLMoE is NOT registered in veRL's Megatron model registry (`verl/models/registry.py`), but it loads through the generic FSDP + HF auto-load path. No veRL-side registration is required for FSDP training + vLLM rollout.

## Integration Points

### 1. Actor Model Initialization -> FSDP Wrap -> Optimizer Creation

**File**: `verl/workers/engine/fsdp/transformer_impl.py`
**Method**: `FSDPEngine._build_model_optimizer` (lines 564-599)

```python
def _build_model_optimizer(self):
    module = self._build_module()               # HF model loaded (line 568)
    if self._is_lora:
        module = self._build_lora_module(module)
    if self._qat_enabled and not self.engine_config.forward_only:
        module = self._apply_qat(module)
    torch.distributed.barrier()
    if self.rank == 0:
        print_model_size(module)
    log_gpu_memory_usage("After init model from HF AutoModel", logger=logger)
    # >>> INSERTION POINT: apply_parameter_update_policy(module, config) <<<
    module = self._build_fsdp_module(module)    # FSDP wrap (line 585)
    if not self.engine_config.forward_only:
        optimizer = self._build_optimizer(module)   # line 588
        lr_scheduler = self._build_lr_scheduler(optimizer)
    ...
```

**Insertion point**: between `print_model_size` / `log_gpu_memory_usage("After init model...")` and `_build_fsdp_module(module)`. At this point `module` is a plain `nn.Module` with original parameter names (`model.layers.0.mlp.gate.weight`, etc.), pre-FSDP, pre-optimizer.

### 2. FSDP / FSDP2 Wrapping

**File**: `verl/workers/engine/fsdp/transformer_impl.py`
**Method**: `FSDPEngine._build_fsdp_module` (lines 361-470)
- FSDP1 (lines 399-425): uses `FSDP(..., use_orig_params=self.engine_config.use_orig_params, ...)`. **Default `use_orig_params=False` flattens params**, so per-parameter `requires_grad` must be set BEFORE wrapping (which our insertion point satisfies).
- FSDP2 (lines 426-449): uses `fully_shard` per-module. Preserves original params.

**Config**: `verl/workers/config/engine.py:265` — `FSDPEngineConfig.use_orig_params: bool = False`. Default `actor.strategy=fsdp` (FSDP1). To use FSDP2 set `actor.strategy=fsdp2`.

### 3. Optimizer Creation (NO requires_grad filter — needs fix)

**File**: `verl/workers/engine/fsdp/transformer_impl.py`
**Method**: `FSDPEngine._build_optimizer` (lines 472-477)

```python
def _build_optimizer(self, module):
    from verl.workers.config.optimizer import build_optimizer
    optimizer = build_optimizer(module.parameters(), self.optimizer_config)
    return optimizer
```

**File**: `verl/workers/config/optimizer.py`
**Function**: `build_optimizer` (lines 242-295) — passes `parameters` straight to `optimizer_cls(parameters, **args)`.

**Required change**: pass `[p for p in module.parameters() if p.requires_grad]` so frozen params don't get optimizer state allocated. This is needed especially for FSDP1 (where frozen params still appear in `module.parameters()` even if `requires_grad=False`).

### 4. Actor Config (where to add `parameter_update`)

**File**: `verl/workers/config/actor.py`
- `ActorConfig` dataclass (lines 103-194) — add field `parameter_update: ParameterUpdateConfig = field(default_factory=ParameterUpdateConfig)`
- `FSDPActorConfig` (lines 288-338) — inherits the field.

**YAML**: `verl/trainer/config/actor/actor.yaml` — add a `parameter_update:` block with `_target_: verl.workers.config.ParameterUpdateConfig`, `mode: full`, `strict: true`, `log_parameter_stats: true`. (Must match the `policy_loss`, `qat`, `router_replay` pattern.)

**Export**: `verl/workers/config/__init__.py` — add `ParameterUpdateConfig` to exports.

### 5. GSPO Policy Loss (DO NOT MODIFY)

**File**: `verl/trainer/ppo/core_algos.py`
**Function**: `compute_policy_loss_gspo` (lines 1538-1611), registered via `@register_policy_loss("gspo")`.
- Computes sequence-level importance ratio, applies `clip_ratio_low`/`clip_ratio_high`, aggregates with `seq-mean-token-mean`.
- Reads `config.clip_ratio_low`, `config.clip_ratio_high`, `config.clip_ratio_c` from `ActorConfig`.

**Enabled via**: `actor_rollout_ref.actor.policy_loss.loss_mode=gspo`.
**Advantage**: `algorithm.adv_estimator=grpo` (`compute_grpo_outcome_advantage`, lines 267-331).

**Canonical GSPO flags** (from `verl/examples/gspo_trainer/run_qwen3_8b_fsdp.sh`):
```
actor.policy_loss.loss_mode=gspo
actor.loss_agg_mode=seq-mean-token-mean
actor.clip_ratio_low=3e-4
actor.clip_ratio_high=4e-4
actor.clip_ratio_c=10.0
algorithm.adv_estimator=grpo
```

### 6. Actor -> Rollout Weight Sync (DO NOT MODIFY)

**Trainer**: `verl/checkpoint_engine/base.py:CheckpointEngineManager.update_weights` (lines 485-518).
**Actor worker**: `verl/workers/engine_workers.py:ActorRolloutRefWorker.update_weights` (lines 670-755) — calls `self.actor.engine.get_per_tensor_param()`.
**FSDP export**: `verl/workers/engine/fsdp/transformer_impl.py:get_per_tensor_param` (lines 855-916) — yields ALL params from `state_dict()`. This is **freeze-friendly**: frozen params keep their (unchanged) values and are still synced, so the rollout engine always receives a complete, consistent model.
**vLLM receive**: `verl/workers/rollout/vllm_rollout/vllm_rollout.py:ServerAdapter.update_weights` (lines 263-305).

No modification needed. The whole-model sync automatically includes frozen (unchanged) params.

### 7. Main Training Loop

**File**: `verl/trainer/ppo/ray_trainer.py`
**Method**: `RayPPOTrainer.fit` (lines 1368-1683) — the legacy trainer. Default `trainer.use_v1=true` (V1 trainer in `verl/trainer/ppo/v1/`), but the V1 trainer follows the same sequence.

Per-step sequence: rollout -> reward -> advantage (GRPO) -> actor update (GSPO) -> checkpoint -> weight sync.

## Files To Create

| Path | Purpose |
|------|---------|
| `verl/verl/utils/parameter_update_policy.py` | `apply_parameter_update_policy(model, mode, strict)` + per-model-type dispatch + audit/strict validation |
| `verl/tests/models/test_parameter_update_policy.py` | Unit tests (full/moe/experts_only, requires_grad, backward gradient) |
| `verl/tests/models/test_checkpoint_delta.py` | Checkpoint delta test (one optimizer step) |
| `verl/experiments/moe_localized_rl/configs/olmoe_full_gspo.yaml` | Full GSPO config |
| `verl/experiments/moe_localized_rl/configs/olmoe_moe_gspo.yaml` | MoE-only GSPO config |
| `verl/experiments/moe_localized_rl/configs/olmoe_experts_only_gspo.yaml` | Experts-only config |
| `verl/experiments/moe_localized_rl/scripts/run_smoke_full.sh` | Smoke test script (Full) |
| `verl/experiments/moe_localized_rl/scripts/run_smoke_moe.sh` | Smoke test script (MoE-only) |
| `verl/experiments/moe_localized_rl/scripts/run_pilot_full.sh` | Pilot script (Full) |
| `verl/experiments/moe_localized_rl/scripts/run_pilot_moe.sh` | Pilot script (MoE-only) |
| `verl/experiments/moe_localized_rl/tools/inspect_model_modules.py` | Inspect model modules |
| `verl/experiments/moe_localized_rl/tools/verify_trainable_params.py` | Verify trainable params |
| `verl/experiments/moe_localized_rl/tools/compare_checkpoints.py` | Checkpoint delta comparison |
| `verl/experiments/moe_localized_rl/README.md` | Experiment README |

## Files To Modify (Minimal)

| File | Change |
|------|--------|
| `verl/verl/workers/config/actor.py` | Add `ParameterUpdateConfig` dataclass + field on `ActorConfig` |
| `verl/verl/workers/config/__init__.py` | Export `ParameterUpdateConfig` |
| `verl/verl/trainer/config/actor/actor.yaml` | Add `parameter_update:` YAML block (default `mode: full`) |
| `verl/verl/workers/engine/fsdp/transformer_impl.py` | (1) Insert `apply_parameter_update_policy()` call in `_build_model_optimizer` after `_build_module`, before `_build_fsdp_module`. (2) Modify `_build_optimizer` to filter `p for p in module.parameters() if p.requires_grad`. |

## Design Decisions

1. **Default `mode: full`** → vanilla veRL behavior, full backward compatibility.
2. **Freezing happens pre-FSDP** so it works with both FSDP1 (`use_orig_params=False`) and FSDP2.
3. **Optimizer param filter** added so frozen params don't waste optimizer state. For `mode=full` this is a no-op (all params have `requires_grad=True`).
4. **Class-based dispatch** (`isinstance(module, OlmoeExperts)`) rather than string matching, with a `MODEL_PARAMETER_POLICIES` dict keyed on model type for future extension.
5. **Strict validation** asserts expected trainable/frozen counts per category — fails the run loudly, no silent fallback.
6. **Weight sync untouched** — full model is synced each step (frozen params are unchanged but still transmitted).
7. **FSDP2 preferred** for the MoE-only run (`actor.strategy=fsdp2`) per guidance, with FSDP1 + `use_orig_params=true` as fallback.
