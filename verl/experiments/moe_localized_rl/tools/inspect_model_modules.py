#!/usr/bin/env python
"""Inspect the module structure of an OLMoE model (or any HF model).

Usage:
    python inspect_model_modules.py --model /path/to/OLMoE-1B-7B-0125-Instruct
    python inspect_model_modules.py --tiny  # use a tiny config (no download)
"""

from __future__ import annotations

import argparse
from collections import Counter

import torch.nn as nn


def make_tiny_olmoe():
    from transformers import OlmoeConfig, OlmoeForCausalLM

    cfg = OlmoeConfig(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        intermediate_size=128,
        vocab_size=256,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        norm_topk_prob=True,
    )
    return OlmoeForCausalLM(cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="HF model path (None -> tiny)")
    parser.add_argument("--tiny", action="store_true", help="use a tiny OLMoE config")
    args = parser.parse_args()

    if args.tiny or args.model is None:
        model = make_tiny_olmoe()
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(args.model)

    # Print module class distribution.
    class_counts = Counter()
    for module in model.modules():
        class_counts[type(module).__name__] += 1
    print("\n=== Module class distribution ===")
    for cls, n in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  {cls:<35s} {n:>5d}")

    # Print named parameters grouped by top-level prefix.
    print("\n=== Top-level parameter prefixes ===")
    prefixes = Counter()
    for name, p in model.named_parameters():
        prefix = name.split(".")[0]
        prefixes[prefix] += p.numel()
    for prefix, numel in sorted(prefixes.items(), key=lambda x: -x[1]):
        print(f"  {prefix:<25s} {numel:>12,d}")

    # Total.
    total = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total:,d}")

    # OLMoE-specific: show MoE block structure.
    try:
        from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock

        for name, module in model.named_modules():
            if isinstance(module, OlmoeSparseMoeBlock):
                print(f"\n=== MoE block: {name} ===")
                print(f"  gate (router): {module.gate}")
                print(f"  num_experts: {len(module.experts)}")
                print(f"  expert[0]: {module.experts[0]}")
                break
    except Exception as e:
        print(f"Could not inspect MoE block: {e}")


if __name__ == "__main__":
    main()
