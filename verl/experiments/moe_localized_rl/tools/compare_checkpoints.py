#!/usr/bin/env python
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

"""Compare two checkpoints and report per-category parameter deltas.

Usage:
    python compare_checkpoints.py --before <ckpt_a_dir> --after <ckpt_b_dir> \
        --model <hf_model_path_or_tiny> [--mode moe] [--output report.json]

For veRL FSDP checkpoints saved in HF format, both directories should contain
a `model.safetensors` or sharded `model-*.safetensors` files. For tiny models
used in unit tests, pass `--model tiny` to build a tiny OLMoE config.

The output table shows, per category:
    - number of changed tensors (max_abs_delta > 0)
    - max_abs_delta across all tensors in the category

This is the hard correctness gate for the MoE-localized RL experiment: frozen
categories must show 0 changed tensors and max_abs_delta == 0.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

# OLMoE classes for categorization
try:
    from transformers.models.olmoe.modeling_olmoe import (
        OlmoeAttention,
        OlmoeRMSNorm,
        OlmoeSparseMoeBlock,
    )
    HAS_OLMOE = True
except ImportError:
    HAS_OLMOE = False


def load_state_dict(path: str) -> dict[str, torch.Tensor]:
    """Load a state dict from a directory or file."""
    from safetensors.torch import load_file

    path = Path(path)
    if path.is_file():
        return load_file(str(path))
    # Directory: look for sharded safetensors.
    shards = sorted(path.glob("model-*.safetensors"))
    if shards:
        state = {}
        for shard in shards:
            state.update(load_file(str(shard)))
        return state
    single = path / "model.safetensors"
    if single.exists():
        return load_file(str(single))
    # Fallback: torch checkpoint.
    pt = path / "pytorch_model.bin"
    if pt.exists():
        return torch.load(str(pt), map_location="cpu")
    raise FileNotFoundError(f"No checkpoint found in {path}")


def classify_param(name: str) -> str:
    """Classify a parameter name into a category based on OLMoE naming."""
    if name.endswith("embed_tokens.weight") or "embed_tokens" in name:
        return "embedding"
    if name == "lm_head.weight" or name.startswith("lm_head."):
        return "lm_head"
    if ".mlp.gate" in name or name.endswith(".mlp.gate.weight"):
        return "router"
    if ".mlp.experts" in name:
        return "experts"
    if ".self_attn" in name:
        return "attention"
    if "layernorm" in name or name.endswith(".norm.weight"):
        return "norm"
    return "other"


def compare_checkpoints(before: dict, after: dict) -> dict:
    """Compare two state dicts and return per-category deltas."""
    categories = {
        "attention": {"changed_tensors": 0, "total_tensors": 0, "max_abs_delta": 0.0},
        "experts": {"changed_tensors": 0, "total_tensors": 0, "max_abs_delta": 0.0},
        "router": {"changed_tensors": 0, "total_tensors": 0, "max_abs_delta": 0.0},
        "norm": {"changed_tensors": 0, "total_tensors": 0, "max_abs_delta": 0.0},
        "embedding": {"changed_tensors": 0, "total_tensors": 0, "max_abs_delta": 0.0},
        "lm_head": {"changed_tensors": 0, "total_tensors": 0, "max_abs_delta": 0.0},
        "other": {"changed_tensors": 0, "total_tensors": 0, "max_abs_delta": 0.0},
    }
    per_tensor = []
    for name in before:
        if name not in after:
            continue
        cat = classify_param(name)
        delta = (after[name].float() - before[name].float()).abs().max().item()
        categories[cat]["total_tensors"] += 1
        if delta > 0:
            categories[cat]["changed_tensors"] += 1
        categories[cat]["max_abs_delta"] = max(categories[cat]["max_abs_delta"], delta)
        per_tensor.append({"name": name, "category": cat, "max_abs_delta": delta})
    return {"categories": categories, "per_tensor": per_tensor}


def print_report(report: dict, mode: str = "moe"):
    cats = report["categories"]
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"Checkpoint Delta Report (expected mode: {mode})")
    print(sep)
    print(f"{'Category':<15s} {'Changed':>10s} {'Total':>10s} {'Max Abs Delta':>18s}")
    print("-" * 70)
    for name in ("attention", "experts", "router", "norm", "embedding", "lm_head", "other"):
        c = cats.get(name, {"changed_tensors": 0, "total_tensors": 0, "max_abs_delta": 0.0})
        print(f"{name:<15s} {c['changed_tensors']:>10d} {c['total_tensors']:>10d} {c['max_abs_delta']:>18.6e}")
    print(sep)

    # Validation for moe mode.
    if mode == "moe":
        ok = True
        for name in ("attention", "norm", "embedding", "lm_head"):
            c = cats[name]
            if c["changed_tensors"] != 0 or c["max_abs_delta"] != 0:
                print(f"  [FAIL] {name} changed: {c['changed_tensors']} tensors, delta={c['max_abs_delta']}")
                ok = False
        if cats["experts"]["changed_tensors"] == 0:
            print(f"  [FAIL] experts did not change")
            ok = False
        if cats["router"]["changed_tensors"] == 0:
            print(f"  [FAIL] router did not change")
            ok = False
        if ok:
            print("  [PASS] MoE-only checkpoint delta validation passed.")
        else:
            print("  [FAIL] MoE-only checkpoint delta validation FAILED.")
        return ok
    elif mode == "experts_only":
        ok = True
        for name in ("attention", "norm", "embedding", "lm_head", "router"):
            c = cats[name]
            if c["changed_tensors"] != 0 or c["max_abs_delta"] != 0:
                print(f"  [FAIL] {name} changed: {c['changed_tensors']} tensors, delta={c['max_abs_delta']}")
                ok = False
        if cats["experts"]["changed_tensors"] == 0:
            print(f"  [FAIL] experts did not change")
            ok = False
        if ok:
            print("  [PASS] Experts-only checkpoint delta validation passed.")
        else:
            print("  [FAIL] Experts-only checkpoint delta validation FAILED.")
        return ok
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", required=True, help="checkpoint A directory (before)")
    parser.add_argument("--after", required=True, help="checkpoint B directory (after)")
    parser.add_argument("--mode", default="moe", choices=["full", "moe", "experts_only"])
    parser.add_argument("--output", default=None, help="write JSON report here")
    args = parser.parse_args()

    print(f"Loading before: {args.before}")
    before = load_state_dict(args.before)
    print(f"Loading after:  {args.after}")
    after = load_state_dict(args.after)

    report = compare_checkpoints(before, after)
    ok = print_report(report, mode=args.mode)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report written to {args.output}")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
