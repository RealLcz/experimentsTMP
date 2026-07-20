# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Parameter update policy for localized RL on Mixture-of-Experts models.

This module provides ``apply_parameter_update_policy`` which selectively freezes
or unfreezes parameter groups of a HuggingFace model so that only a target
subspace (e.g. MoE experts + router) is trainable. The policy is applied after
model construction and *before* FSDP wrapping / optimizer creation, which keeps
it compatible with both FSDP1 (``use_orig_params=False`` flattens params) and
FSDP2.

Supported modes:
    - ``full``: all parameters trainable (vanilla veRL behavior).
    - ``moe``: only MoE experts + router trainable (core experiment).
    - ``experts_only``: only MoE experts trainable (router frozen).
    - ``attention_only``: only attention layers trainable (control for MoE-only).

The implementation dispatches on the HF model type via
``MODEL_PARAMETER_POLICIES`` so new architectures (qwen3_moe, deepseek, ...)
can be registered without modifying the core function.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Category helpers
# ---------------------------------------------------------------------------

def _numel_of(param: Optional[torch.nn.Parameter]) -> int:
    return int(param.numel()) if param is not None else 0


def _set_requires_grad(module: nn.Module, value: bool) -> None:
    for p in module.parameters(recurse=True):
        p.requires_grad = value


def _module_total_numel(module: Optional[nn.Module]) -> int:
    if module is None:
        return 0
    return sum(_numel_of(p) for p in module.parameters(recurse=True))


def _module_trainable_numel(module: Optional[nn.Module]) -> int:
    if module is None:
        return 0
    return sum(_numel_of(p) for p in module.parameters(recurse=True) if p.requires_grad)


# ---------------------------------------------------------------------------
# OLMoE policy
# ---------------------------------------------------------------------------

def _get_olmoe_block_class() -> Optional[type]:
    """Return the OlmoeSparseMoeBlock class if transformers ships OLMoE."""
    try:
        from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock
    except Exception:  # pragma: no cover - transformers without olmoe
        return None
    return OlmoeSparseMoeBlock


def _is_olmoe_model(model: nn.Module) -> bool:
    try:
        from transformers.models.olmoe.modeling_olmoe import OlmoeForCausalLM
    except Exception:
        return False
    # Walk up to the root module to handle wrapped models.
    return isinstance(model, OlmoeForCausalLM) or any(
        isinstance(m, OlmoeForCausalLM) for m in model.modules()
    )


def apply_olmoe_policy(model: nn.Module, mode: str) -> dict[str, Any]:
    """Apply the parameter update policy to an OLMoE model in-place.

    OLMoE structure (transformers >= 4.46):
        OlmoeForCausalLM
          .model.embed_tokens           -> embedding
          .model.layers.{i}.self_attn   -> attention (OlmoeAttention / Sdpa/Flash2)
          .model.layers.{i}.mlp         -> OlmoeSparseMoeBlock
              .gate                     -> nn.Linear  (router)
              .experts                  -> nn.ModuleList[OlmoeMLP]  (experts)
          .model.layers.{i}.input_layernorm         -> OlmoeRMSNorm
          .model.layers.{i}.post_attention_layernorm-> OlmoeRMSNorm
          .model.norm                    -> OlmoeRMSNorm
          .lm_head                       -> nn.Linear

    We use ``isinstance`` on ``OlmoeSparseMoeBlock`` to robustly identify the
    MoE block (router = ``.gate``, experts = ``.experts``), and classify the
    remaining modules by name prefix so we never accidentally match a dense
    FFN labelled "mlp".
    """
    OlmoeSparseMoeBlock = _get_olmoe_block_class()
    if OlmoeSparseMoeBlock is None:
        raise ImportError(
            "OLMoE policy requested but transformers does not expose "
            "OlmoeSparseMoeBlock. Upgrade transformers or pick mode='full'."
        )

    # 1. Start from a known state: full sets all True, the others freeze first.
    if mode == "full":
        for p in model.parameters():
            p.requires_grad = True
    else:
        for p in model.parameters():
            p.requires_grad = False

    # 2. Re-enable target parameters for localized modes.
    if mode in ("moe", "experts_only"):
        for module in model.modules():
            if isinstance(module, OlmoeSparseMoeBlock):
                # Router (gate) is trainable only in moe mode.
                if mode == "moe" and hasattr(module, "gate"):
                    _set_requires_grad(module.gate, True)
                # Experts are trainable in both moe and experts_only modes.
                if hasattr(module, "experts"):
                    _set_requires_grad(module.experts, True)
    elif mode == "attention_only":
        from transformers.models.olmoe.modeling_olmoe import OlmoeAttention

        for module in model.modules():
            if isinstance(module, OlmoeAttention):
                _set_requires_grad(module, True)

    # 3. Build a per-category audit.
    return _audit_olmoe(model, mode, OlmoeSparseMoeBlock)


def _audit_olmoe(model: nn.Module, mode: str, OlmoeSparseMoeBlock: type) -> dict[str, Any]:
    """Compute per-category parameter counts for an OLMoE model."""

    # Identify the root OlmoeForCausalLM (handles FSDP pre-wrap plain module).
    from transformers.models.olmoe.modeling_olmoe import (
        OlmoeAttention,
        OlmoeRMSNorm,
    )

    categories: dict[str, dict[str, int]] = {
        "attention": {"trainable": 0, "total": 0},
        "experts": {"trainable": 0, "total": 0},
        "router": {"trainable": 0, "total": 0},
        "norm": {"trainable": 0, "total": 0},
        "embedding": {"trainable": 0, "total": 0},
        "lm_head": {"trainable": 0, "total": 0},
    }

    # Walk named modules so we can attribute each module to a category.
    seen_params: set[int] = set()

    def add(module: nn.Module, category: str) -> None:
        for p in module.parameters(recurse=True):
            pid = id(p)
            if pid in seen_params:
                continue
            seen_params.add(pid)
            n = _numel_of(p)
            categories[category]["total"] += n
            if p.requires_grad:
                categories[category]["trainable"] += n

    def is_attention_internal_norm(name: str) -> bool:
        # OLMoE places q_norm / k_norm inside self_attn; treat as attention.
        return ".self_attn." in f".{name}."

    # embed_tokens: search for it under model.embed_tokens (HF naming).
    for name, module in model.named_modules():
        if name.endswith("embed_tokens") and isinstance(module, nn.Embedding):
            add(module, "embedding")
        elif name == "lm_head" or name.endswith("lm_head"):
            # nn.Linear LM head (OLMoE does not tie embeddings by default).
            if isinstance(module, nn.Linear):
                add(module, "lm_head")
        elif isinstance(module, OlmoeSparseMoeBlock):
            if hasattr(module, "gate"):
                add(module.gate, "router")
            if hasattr(module, "experts"):
                add(module.experts, "experts")
        elif isinstance(module, OlmoeAttention):
            add(module, "attention")
        elif isinstance(module, OlmoeRMSNorm):
            if is_attention_internal_norm(name):
                add(module, "attention")
            else:
                add(module, "norm")

    return _build_audit_summary(model, mode, categories)


# ---------------------------------------------------------------------------
# Gemma 4 policy (26B-A4B MoE text stack inside multimodal wrapper)
# ---------------------------------------------------------------------------

def _get_gemma4_policy_classes():
    try:
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4ForConditionalGeneration,
            Gemma4RMSNorm,
            Gemma4TextAttention,
            Gemma4TextExperts,
            Gemma4TextRouter,
        )
    except Exception:
        return None
    return {
        "root": Gemma4ForConditionalGeneration,
        "attention": Gemma4TextAttention,
        "experts": Gemma4TextExperts,
        "router": Gemma4TextRouter,
        "norm": Gemma4RMSNorm,
    }


def _is_gemma4_model(model: nn.Module) -> bool:
    classes = _get_gemma4_policy_classes()
    if classes is not None:
        root = classes["root"]
        if isinstance(model, root) or any(isinstance(m, root) for m in model.modules()):
            return True
        # Text-only Gemma4 blocks may appear without the multimodal wrapper class.
        attn = classes["attention"]
        if any(isinstance(m, attn) for m in model.modules()):
            return True
    config = getattr(model, "config", None)
    if config is not None:
        if getattr(config, "model_type", None) == "gemma4":
            return True
        text_config = getattr(config, "text_config", None)
        if text_config is not None and getattr(text_config, "model_type", None) in (
            "gemma4_text",
            "gemma4",
        ):
            return True
    return False


def apply_gemma4_policy(model: nn.Module, mode: str) -> dict[str, Any]:
    classes = _get_gemma4_policy_classes()
    if classes is None:
        raise ImportError(
            "Gemma4 policy requested but transformers does not expose gemma4 models. "
            "Install transformers>=5.5.0 or use mode='full'."
        )

    Gemma4TextAttention = classes["attention"]
    Gemma4TextExperts = classes["experts"]
    Gemma4TextRouter = classes["router"]
    Gemma4RMSNorm = classes["norm"]

    if mode == "full":
        for p in model.parameters():
            p.requires_grad = True
    else:
        for p in model.parameters():
            p.requires_grad = False

    if mode in ("moe", "experts_only"):
        for module in model.modules():
            if isinstance(module, Gemma4TextRouter):
                if mode == "moe":
                    _set_requires_grad(module, True)
            elif isinstance(module, Gemma4TextExperts):
                _set_requires_grad(module, True)
    elif mode == "attention_only":
        for module in model.modules():
            if isinstance(module, Gemma4TextAttention):
                _set_requires_grad(module, True)

    categories: dict[str, dict[str, int]] = {
        "attention": {"trainable": 0, "total": 0},
        "experts": {"trainable": 0, "total": 0},
        "router": {"trainable": 0, "total": 0},
        "norm": {"trainable": 0, "total": 0},
        "embedding": {"trainable": 0, "total": 0},
        "lm_head": {"trainable": 0, "total": 0},
    }
    seen_params: set[int] = set()

    def add(module: nn.Module, category: str) -> None:
        for p in module.parameters(recurse=True):
            pid = id(p)
            if pid in seen_params:
                continue
            seen_params.add(pid)
            n = _numel_of(p)
            categories[category]["total"] += n
            if p.requires_grad:
                categories[category]["trainable"] += n

    for name, module in model.named_modules():
        if name.endswith("embed_tokens") and isinstance(module, nn.Embedding):
            add(module, "embedding")
        elif name == "lm_head" or name.endswith("lm_head"):
            if isinstance(module, nn.Linear):
                add(module, "lm_head")
        elif isinstance(module, Gemma4TextExperts):
            add(module, "experts")
        elif isinstance(module, Gemma4TextRouter):
            add(module, "router")
        elif isinstance(module, Gemma4TextAttention):
            add(module, "attention")
        elif isinstance(module, Gemma4RMSNorm):
            add(module, "norm")

    return _build_audit_summary(model, mode, categories)


# ---------------------------------------------------------------------------
# GPT-OSS policy
# ---------------------------------------------------------------------------

def _is_gpt_oss_model(model: nn.Module) -> bool:
    try:
        from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM
    except Exception:
        return False
    return isinstance(model, GptOssForCausalLM) or any(
        isinstance(m, GptOssForCausalLM) for m in model.modules()
    )


def apply_gpt_oss_policy(model: nn.Module, mode: str) -> dict[str, Any]:
    """Apply parameter update policy to GPT-OSS models."""
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssAttention, GptOssMLP

    if mode == "full":
        for p in model.parameters():
            p.requires_grad = True
    else:
        for p in model.parameters():
            p.requires_grad = False

    if mode in ("moe", "experts_only"):
        for module in model.modules():
            if isinstance(module, GptOssMLP):
                if mode == "moe" and hasattr(module, "router"):
                    _set_requires_grad(module.router, True)
                if hasattr(module, "experts"):
                    _set_requires_grad(module.experts, True)
    elif mode == "attention_only":
        for module in model.modules():
            if isinstance(module, GptOssAttention):
                _set_requires_grad(module, True)

    return _audit_gpt_oss(model, mode)


def _audit_gpt_oss(model: nn.Module, mode: str) -> dict[str, Any]:
    from transformers.models.gpt_oss.modeling_gpt_oss import (
        GptOssAttention,
        GptOssMLP,
        GptOssRMSNorm,
    )

    categories: dict[str, dict[str, int]] = {
        "attention": {"trainable": 0, "total": 0},
        "experts": {"trainable": 0, "total": 0},
        "router": {"trainable": 0, "total": 0},
        "norm": {"trainable": 0, "total": 0},
        "embedding": {"trainable": 0, "total": 0},
        "lm_head": {"trainable": 0, "total": 0},
    }
    seen_params: set[int] = set()

    def add(module: nn.Module, category: str) -> None:
        for p in module.parameters(recurse=True):
            pid = id(p)
            if pid in seen_params:
                continue
            seen_params.add(pid)
            n = _numel_of(p)
            categories[category]["total"] += n
            if p.requires_grad:
                categories[category]["trainable"] += n

    for name, module in model.named_modules():
        if name.endswith("embed_tokens") and isinstance(module, nn.Embedding):
            add(module, "embedding")
        elif name == "lm_head" or name.endswith("lm_head"):
            if isinstance(module, nn.Linear):
                add(module, "lm_head")
        elif isinstance(module, GptOssMLP):
            if hasattr(module, "router"):
                add(module.router, "router")
            if hasattr(module, "experts"):
                add(module.experts, "experts")
        elif isinstance(module, GptOssAttention):
            add(module, "attention")
        elif isinstance(module, GptOssRMSNorm):
            add(module, "norm")

    return _build_audit_summary(model, mode, categories)


def _build_audit_summary(
    model: nn.Module, mode: str, categories: dict[str, dict[str, int]]
) -> dict[str, Any]:
    total = sum(c["total"] for c in categories.values())
    trainable = sum(c["trainable"] for c in categories.values())
    summary = {
        "mode": mode,
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
        "trainable_ratio": (trainable / total) if total > 0 else 0.0,
        "categories": categories,
    }
    return summary


# ---------------------------------------------------------------------------
# Registry / dispatch
# ---------------------------------------------------------------------------

MODEL_PARAMETER_POLICIES: dict[str, Any] = {
    "olmoe": {
        "is_model": _is_olmoe_model,
        "apply": apply_olmoe_policy,
    },
    "gemma4": {
        "is_model": _is_gemma4_model,
        "apply": apply_gemma4_policy,
    },
    "gpt_oss": {
        "is_model": _is_gpt_oss_model,
        "apply": apply_gpt_oss_policy,
    },
}


def _detect_model_type(model: nn.Module) -> Optional[str]:
    for model_type, policy in MODEL_PARAMETER_POLICIES.items():
        try:
            if policy["is_model"](model):
                return model_type
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

VALID_MODES = ("full", "moe", "experts_only", "attention_only")


def apply_parameter_update_policy(
    model: nn.Module,
    mode: str,
    strict: bool = True,
    log_parameter_stats: bool = True,
    output_dir: Optional[str] = None,
) -> dict[str, Any]:
    """Apply a parameter update policy to ``model`` in-place.

    Args:
        model: A HuggingFace model (pre-FSDP wrapping).
        mode: One of ``full``, ``moe``, ``experts_only``, ``attention_only``.
        strict: When True, validate that the expected categories are
            trainable / frozen and raise on any mismatch. Failures abort
            training (no silent fallback).
        log_parameter_stats: Print the human-readable parameter audit.
        output_dir: If provided, write ``parameter_policy.json`` here.

    Returns:
        Audit summary dict with per-category trainable / total counts.
    """
    if mode not in VALID_MODES:
        raise ValueError(
            f"Invalid parameter_update.mode '{mode}'. Must be one of {VALID_MODES}."
        )

    model_type = _detect_model_type(model)
    if model_type is None:
        # Unknown model: only 'full' is safe (no-op). Non-full is rejected so we
        # never silently train parameters that should be frozen.
        if mode != "full":
            raise ValueError(
                "parameter_update.mode='%s' requires a registered model policy, "
                "but no model type matched the given model. Register a policy in "
                "MODEL_PARAMETER_POLICIES or use mode='full'." % mode
            )
        # 'full' on unknown model: ensure all params trainable and return a
        # minimal audit.
        for p in model.parameters():
            p.requires_grad = True
        summary = _build_audit_summary(
            model,
            mode,
            {
                "attention": {"trainable": 0, "total": 0},
                "experts": {"trainable": 0, "total": 0},
                "router": {"trainable": 0, "total": 0},
                "norm": {"trainable": 0, "total": 0},
                "embedding": {"trainable": 0, "total": 0},
                "lm_head": {"trainable": 0, "total": 0},
            },
        )
        if log_parameter_stats:
            _log_audit(summary, model_type="unknown")
        if output_dir is not None:
            _write_audit(summary, output_dir, model_type="unknown")
        return summary

    policy = MODEL_PARAMETER_POLICIES[model_type]
    summary = policy["apply"](model, mode)
    summary["model_type"] = model_type
    summary["mode"] = mode

    if strict:
        _strict_validate(summary, mode)

    if log_parameter_stats:
        _log_audit(summary, model_type=model_type)
    if output_dir is not None:
        _write_audit(summary, output_dir, model_type=model_type)

    return summary


def _strict_validate(summary: dict[str, Any], mode: str) -> None:
    """Assert the expected trainable/frozen counts per category.

    Any assertion failure raises and aborts training (no silent fallback).
    """
    cat = summary["categories"]

    def trainable_of(name: str) -> int:
        return int(cat.get(name, {}).get("trainable", 0))

    if mode == "full":
        # All categories with non-zero params must be trainable.
        for name, c in cat.items():
            if c["total"] > 0 and c["trainable"] != c["total"]:
                raise AssertionError(
                    f"[parameter_update][strict] full mode requires all '{name}' "
                    f"params trainable, got {c['trainable']}/{c['total']}."
                )
        return

    if mode == "attention_only":
        if trainable_of("attention") <= 0:
            raise AssertionError(
                "[parameter_update][strict] mode='attention_only' requires trainable "
                "attention layers, but attention trainable params == 0."
            )
        for name in ("experts", "router", "norm", "embedding", "lm_head"):
            if trainable_of(name) != 0:
                raise AssertionError(
                    f"[parameter_update][strict] mode='attention_only' requires "
                    f"'{name}' to be fully frozen, but {trainable_of(name)} params "
                    "are trainable."
                )
        return

    # For moe / experts_only: shared params must be frozen.
    for name in ("attention", "norm", "embedding", "lm_head"):
        if trainable_of(name) != 0:
            raise AssertionError(
                f"[parameter_update][strict] mode='{mode}' requires '{name}' to be "
                f"fully frozen, but {trainable_of(name)} params are trainable."
            )

    # Experts must be trainable in both moe and experts_only.
    if trainable_of("experts") <= 0:
        raise AssertionError(
            f"[parameter_update][strict] mode='{mode}' requires trainable experts, "
            f"but experts trainable params == 0."
        )

    if mode == "moe":
        if trainable_of("router") <= 0:
            raise AssertionError(
                "[parameter_update][strict] mode='moe' requires a trainable router, "
                "but router trainable params == 0."
            )
    elif mode == "experts_only":
        if trainable_of("router") != 0:
            raise AssertionError(
                "[parameter_update][strict] mode='experts_only' requires the router "
                f"to be frozen, but {trainable_of('router')} params are trainable."
            )


def _log_audit(summary: dict[str, Any], model_type: str) -> None:
    cat = summary["categories"]
    sep = "=" * 60
    lines = [
        "",
        sep,
        "Parameter Update Policy",
        sep,
        f"Mode: {summary['mode']}",
        f"Model type: {model_type}",
        "",
        f"Total parameters:     {summary['total_parameters']:>15,d}",
        f"Trainable parameters: {summary['trainable_parameters']:>15,d}",
        f"Frozen parameters:    {summary['frozen_parameters']:>15,d}",
        f"Trainable ratio:      {summary['trainable_ratio'] * 100:>14.2f} %",
        "",
        "By category:",
    ]
    for name in ("attention", "experts", "router", "norm", "embedding", "lm_head"):
        c = cat.get(name, {"trainable": 0, "total": 0})
        lines.append(
            f"  {name:<12s} {c['trainable']:>12,d} trainable / {c['total']:>12,d} total"
        )
    lines.append(sep)
    logger.info("\n".join(lines))
    # Also print to stdout so it is visible in SLURM logs regardless of log level.
    print("\n".join(lines), flush=True)


def _write_audit(summary: dict[str, Any], output_dir: str, model_type: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "parameter_policy.json")
    payload = {
        "mode": summary["mode"],
        "model_type": model_type,
        "total_parameters": summary["total_parameters"],
        "trainable_parameters": summary["trainable_parameters"],
        "frozen_parameters": summary["frozen_parameters"],
        "trainable_ratio": summary["trainable_ratio"],
        "categories": summary["categories"],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"[parameter_update] Wrote parameter policy audit to {path}")


# ---------------------------------------------------------------------------
# Convenience: trainable parameter iterator for optimizer construction
# ---------------------------------------------------------------------------

def trainable_parameters(model: nn.Module):
    """Yield parameters with requires_grad=True (for optimizer construction)."""
    for p in model.parameters():
        if p.requires_grad:
            yield p
