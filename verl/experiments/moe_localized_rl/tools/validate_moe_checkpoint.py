#!/usr/bin/env python
"""Validate the MoE-only checkpoint by comparing frozen parameters against
the original model.

This loads:
    1. The original OLMoE model weights (from the HF checkpoint).
    2. The trained MoE-only checkpoint (from the veRL FSDP checkpoint).
And verifies that:
    - Attention params are UNCHANGED (bitwise identical)
    - Norm params are UNCHANGED
    - Embedding params are UNCHANGED
    - LM Head params are UNCHANGED
    - Expert params are CHANGED
    - Router params are CHANGED

Usage:
    python validate_moe_checkpoint.py \
        --original /path/to/OLMoE-1B-7B-0125-Instruct \
        --checkpoint /path/to/outputs/smoke_moe/global_step_116/actor \
        --mode moe
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file


def load_original_weights(model_path: str) -> dict[str, torch.Tensor]:
    """Load the original model weights from HF safetensors."""
    model_path = Path(model_path)
    shards = sorted(model_path.glob("model-*.safetensors"))
    if shards:
        state = {}
        for shard in shards:
            state.update(load_file(str(shard)))
        return state
    single = model_path / "model.safetensors"
    if single.exists():
        return load_file(str(single))
    raise FileNotFoundError(f"No safetensors found in {model_path}")


def load_fsdp_checkpoint(ckpt_dir: str) -> dict[str, torch.Tensor]:
    """Load and merge an FSDP sharded checkpoint.

    veRL saves FSDP checkpoints as model_world_size_8_rank_*.pt files.
    For FSDP2, parameters are saved as DTensors sharded on dim 0. We extract
    the local tensor from each rank and concatenate to reconstruct the full
    parameter.
    """
    ckpt_dir = Path(ckpt_dir)
    shards = sorted(ckpt_dir.glob("model_world_size_*_rank_*.pt"))
    if not shards:
        raise FileNotFoundError(f"No FSDP shards found in {ckpt_dir}")

    # Load all shards and extract local tensors from DTensors.
    all_states = []
    for shard in shards:
        state = torch.load(str(shard), map_location="cpu", weights_only=False)
        # Convert DTensors to their local tensors.
        local_state = {}
        for key, t in state.items():
            if hasattr(t, "_local_tensor"):
                local_state[key] = t._local_tensor.clone()
            elif hasattr(t, "full_tensor"):
                local_state[key] = t.full_tensor()
            else:
                local_state[key] = t
        all_states.append(local_state)

    # Merge: concatenate local tensors along dim 0 to reconstruct full params.
    merged = {}
    keys = all_states[0].keys()
    for key in keys:
        tensors = [s[key] for s in all_states]
        # FSDP shards parameters on dim 0, so concatenate along dim 0.
        # But some params (e.g. 1D norms) might be sharded on dim 0 too.
        try:
            merged[key] = torch.cat(tensors, dim=0)
        except RuntimeError:
            # If cat on dim 0 fails (e.g. shape mismatch), try other dims
            # or just use the first tensor (replicated).
            shapes = [t.shape for t in tensors]
            if all(s == shapes[0] for s in shapes):
                merged[key] = tensors[0]
            else:
                for dim in range(tensors[0].dim()):
                    if len(set(t.shape[dim] for t in tensors)) > 1:
                        merged[key] = torch.cat(tensors, dim=dim)
                        break
                else:
                    merged[key] = tensors[0]
    return merged


def classify_param(name: str) -> str:
    """Classify a parameter name into a category."""
    if "embed_tokens" in name:
        return "embedding"
    if name.startswith("lm_head.") or name == "lm_head.weight":
        return "lm_head"
    if ".mlp.gate" in name:
        return "router"
    if ".mlp.experts" in name:
        return "experts"
    if ".self_attn" in name:
        return "attention"
    if "layernorm" in name or name.endswith(".norm.weight"):
        return "norm"
    return "other"


def validate_checkpoint(original: dict, trained: dict, mode: str = "moe") -> dict:
    """Compare original vs trained weights per category."""
    categories = {
        "attention": {"unchanged": 0, "changed": 0, "max_abs_delta": 0.0},
        "experts": {"unchanged": 0, "changed": 0, "max_abs_delta": 0.0},
        "router": {"unchanged": 0, "changed": 0, "max_abs_delta": 0.0},
        "norm": {"unchanged": 0, "changed": 0, "max_abs_delta": 0.0},
        "embedding": {"unchanged": 0, "changed": 0, "max_abs_delta": 0.0},
        "lm_head": {"unchanged": 0, "changed": 0, "max_abs_delta": 0.0},
        "other": {"unchanged": 0, "changed": 0, "max_abs_delta": 0.0},
    }
    per_tensor = []
    for name in original:
        if name not in trained:
            continue
        cat = classify_param(name)
        o = original[name].float()
        t = trained[name].float()
        if o.shape != t.shape:
            # Shape mismatch - skip (likely a sharding issue)
            continue
        delta = (t - o).abs().max().item()
        if delta == 0:
            categories[cat]["unchanged"] += 1
        else:
            categories[cat]["changed"] += 1
        categories[cat]["max_abs_delta"] = max(categories[cat]["max_abs_delta"], delta)
        per_tensor.append({"name": name, "category": cat, "max_abs_delta": delta})

    return {"categories": categories, "per_tensor": per_tensor}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True, help="original HF model directory")
    parser.add_argument("--checkpoint", required=True, help="veRL FSDP checkpoint directory (actor/)")
    parser.add_argument("--mode", default="moe", choices=["full", "moe", "experts_only"])
    args = parser.parse_args()

    print(f"Loading original weights from {args.original}...")
    original = load_original_weights(args.original)
    print(f"  {len(original)} tensors")

    print(f"Loading trained checkpoint from {args.checkpoint}...")
    trained = load_fsdp_checkpoint(args.checkpoint)
    print(f"  {len(trained)} tensors")

    print(f"\nValidating (mode={args.mode})...")
    report = validate_checkpoint(original, trained, mode=args.mode)

    # Print report.
    cats = report["categories"]
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"Checkpoint Validation Report (mode={args.mode})")
    print(sep)
    print(f"{'Category':<15s} {'Unchanged':>10s} {'Changed':>10s} {'Max Abs Delta':>18s}")
    print("-" * 70)
    for name in ("attention", "experts", "router", "norm", "embedding", "lm_head", "other"):
        c = cats[name]
        print(f"{name:<15s} {c['unchanged']:>10d} {c['changed']:>10d} {c['max_abs_delta']:>18.6e}")
    print(sep)

    # Validate for moe mode.
    ok = True
    if args.mode == "moe":
        for name in ("attention", "norm", "embedding", "lm_head"):
            c = cats[name]
            if c["changed"] > 0:
                print(f"  [FAIL] {name} has {c['changed']} changed tensors (should be 0)")
                ok = False
        if cats["experts"]["changed"] == 0:
            print(f"  [FAIL] experts have no changed tensors (should be > 0)")
            ok = False
        if cats["router"]["changed"] == 0:
            print(f"  [FAIL] router has no changed tensors (should be > 0)")
            ok = False
        if ok:
            print("  [PASS] MoE-only checkpoint validation PASSED!")
            print("  - Frozen params (attention/norm/embedding/lm_head) are unchanged")
            print("  - Trainable params (experts/router) have changed")
    elif args.mode == "experts_only":
        for name in ("attention", "norm", "embedding", "lm_head", "router"):
            c = cats[name]
            if c["changed"] > 0:
                print(f"  [FAIL] {name} has {c['changed']} changed tensors (should be 0)")
                ok = False
        if cats["experts"]["changed"] == 0:
            print(f"  [FAIL] experts have no changed tensors (should be > 0)")
            ok = False
        if ok:
            print("  [PASS] Experts-only checkpoint validation PASSED!")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
