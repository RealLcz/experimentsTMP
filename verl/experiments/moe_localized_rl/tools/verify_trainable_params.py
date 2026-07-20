#!/usr/bin/env python
"""Verify the trainable parameter distribution of a model under a policy.

Usage:
    python verify_trainable_params.py --model /path/to/OLMoE --mode moe
    python verify_trainable_params.py --tiny --mode experts_only
"""

from __future__ import annotations

import argparse

from verl.utils.parameter_update_policy import apply_parameter_update_policy


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
    parser.add_argument("--model", default=None)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--mode", default="moe", choices=["full", "moe", "experts_only"])
    parser.add_argument("--strict", action="store_true", default=True)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if args.tiny or args.model is None:
        model = make_tiny_olmoe()
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(args.model)

    summary = apply_parameter_update_policy(
        model,
        mode=args.mode,
        strict=args.strict,
        log_parameter_stats=True,
        output_dir=args.output_dir,
    )
    print(f"\nSummary: mode={summary['mode']}, "
          f"trainable={summary['trainable_parameters']:,d} / {summary['total_parameters']:,d} "
          f"({summary['trainable_ratio']*100:.2f}%)")


if __name__ == "__main__":
    main()
