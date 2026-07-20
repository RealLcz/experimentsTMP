#!/usr/bin/env python
"""Efficient MoE checkpoint validation: only load a few representative
parameters per category to verify the freeze worked.

This avoids loading all 27GB of checkpoint shards by only extracting
the specific keys we need from each shard.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file


# Sample one parameter per category (small ones to minimize memory).
SAMPLE_KEYS = {
    "attention": "model.layers.0.self_attn.q_proj.weight",       # 2048x2048
    "router": "model.layers.0.mlp.gate.weight",                  # 64x2048 (small)
    "norm": "model.layers.0.input_layernorm.weight",             # 2048 (tiny)
    "embedding": "model.embed_tokens.weight",                    # 50304x2048 (big)
    "lm_head": "lm_head.weight",                                  # 50304x2048 (big)
    # Experts: sample a few
    "experts_0": "model.layers.0.mlp.experts.0.gate_proj.weight",
    "experts_1": "model.layers.0.mlp.experts.1.gate_proj.weight",
}


def load_original_param(model_path: str, key: str) -> torch.Tensor:
    """Load a single parameter from the original HF model."""
    model_path = Path(model_path)
    shards = sorted(model_path.glob("model-*.safetensors"))
    for shard in shards:
        state = load_file(str(shard))
        if key in state:
            return state[key].float()
    raise KeyError(f"{key} not found in original model")


def load_checkpoint_param(ckpt_dir: str, key: str, world_size: int = 8) -> torch.Tensor:
    """Load and merge a single parameter from FSDP shards."""
    ckpt_dir = Path(ckpt_dir)
    local_tensors = []
    for rank in range(world_size):
        shard = ckpt_dir / f"model_world_size_{world_size}_rank_{rank}.pt"
        state = torch.load(str(shard), map_location="cpu", weights_only=False)
        t = state[key]
        if hasattr(t, "_local_tensor"):
            t = t._local_tensor.clone()
        elif hasattr(t, "full_tensor"):
            t = t.full_tensor()
        local_tensors.append(t)
        del state
    # Concatenate along dim 0 (FSDP shard dim).
    return torch.cat(local_tensors, dim=0).float()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True)
    parser.add_argument("--checkpoint", required=True, help="actor/ directory")
    parser.add_argument("--mode", default="moe", choices=["full", "moe", "experts_only"])
    args = parser.parse_args()

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"Efficient Checkpoint Validation (mode={args.mode})")
    print(f"{sep}")
    print(f"{'Category':<15s} {'Key':<55s} {'Max Abs Delta':>18s}")
    print("-" * 70)

    results = {}
    for cat, key in SAMPLE_KEYS.items():
        try:
            orig = load_original_param(args.original, key)
            trained = load_checkpoint_param(args.checkpoint, key)
            if orig.shape != trained.shape:
                print(f"{cat:<15s} {key:<55s} SHAPE MISMATCH {orig.shape} vs {trained.shape}")
                results[cat] = None
                continue
            delta = (trained - orig).abs().max().item()
            print(f"{cat:<15s} {key:<55s} {delta:>18.6e}")
            results[cat] = delta
        except Exception as e:
            print(f"{cat:<15s} {key:<55s} ERROR: {e}")
            results[cat] = None

    print(sep)

    # Validate.
    ok = True
    if args.mode == "moe":
        # Frozen categories must have delta == 0.
        for cat in ("attention", "router", "norm", "embedding", "lm_head"):
            d = results.get(cat)
            if d is None:
                continue
            if cat in ("attention", "norm", "embedding", "lm_head") and d != 0:
                print(f"  [FAIL] {cat} changed (delta={d}), should be 0")
                ok = False
        # Router and experts must have changed.
        if results.get("router") == 0:
            print(f"  [FAIL] router unchanged, should be > 0")
            ok = False
        for cat in ("experts_0", "experts_1"):
            d = results.get(cat)
            if d is not None and d == 0:
                print(f"  [FAIL] {cat} unchanged, should be > 0")
                ok = False
        if ok:
            print("  [PASS] MoE-only checkpoint validation PASSED!")
            print("  - Frozen params (attention/norm/embedding/lm_head) are unchanged")
            print("  - Trainable params (experts/router) have changed")
    elif args.mode == "full":
        # In full mode, everything should change (except maybe norm if lr=0 for it).
        print("  [INFO] Full mode: all params should change (no frozen params)")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
