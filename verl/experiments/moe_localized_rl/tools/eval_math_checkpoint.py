#!/usr/bin/env python3
"""Evaluate a veRL FSDP checkpoint or HF model on MATH test set.

Usage:
    # Merge + eval a veRL checkpoint
    python eval_math_checkpoint.py \
        --name attention_step230 \
        --actor_dir /path/to/global_step_230/actor \
        --base_model /path/to/OLMoE-1B-7B-0125-Instruct

    # Eval an existing HF model directory
    python eval_math_checkpoint.py \
        --name base_model \
        --hf_model /path/to/OLMoE-1B-7B-0125-Instruct
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

VERL_ROOT = Path(__file__).resolve().parents[3]
if str(VERL_ROOT) not in sys.path:
    sys.path.insert(0, str(VERL_ROOT))

from verl.utils.reward_score import default_compute_score


def merge_fsdp_actor(actor_dir: str, base_model: str, target_dir: str) -> str:
    target = Path(target_dir)
    if (target / "model.safetensors").exists() or list(target.glob("model-*.safetensors")):
        print(f"[merge] Reusing existing HF model at {target}")
        return str(target)
    target.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        actor_dir,
        "--target_dir",
        str(target),
        "--trust-remote-code",
    ]
    print("[merge]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return str(target)


def load_test_data(test_file: str, max_samples: int | None) -> pd.DataFrame:
    df = pd.read_parquet(test_file)
    if max_samples is not None and max_samples < len(df):
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
    return df


def format_prompt(tokenizer, messages) -> str:
    if hasattr(messages, "tolist"):
        messages = messages.tolist()
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def run_vllm_eval(
    model_path: str,
    df: pd.DataFrame,
    max_tokens: int,
    temperature: float,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
) -> list[str]:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompts = [format_prompt(tokenizer, row["prompt"]) for _, row in df.iterrows()]

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=4096,
    )
    sampling = SamplingParams(
        temperature=temperature,
        top_p=1.0,
        max_tokens=max_tokens,
        n=1,
    )
    outputs = llm.generate(prompts, sampling)
    return [o.outputs[0].text for o in outputs]


def score_outputs(df: pd.DataFrame, outputs: list[str]) -> dict:
    scores = []
    for (_, row), output in zip(df.iterrows(), outputs):
        rm = row["reward_model"]
        gt = rm["ground_truth"] if isinstance(rm, dict) else rm
        ds = row["data_source"]
        score = default_compute_score(ds, output, gt)
        if isinstance(score, dict):
            score = float(score.get("score", score.get("acc", 0.0)))
        scores.append(float(score))
    return {
        "accuracy": sum(scores) / len(scores),
        "num_samples": len(scores),
        "num_correct": int(sum(scores)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Experiment label for results JSON")
    parser.add_argument("--test_file", default="/mnt/vast/home/ym56kacy/jinhe/MendelGM/wom/data/math/test.parquet")
    parser.add_argument("--base_model", default="/mnt/vast/home/ym56kacy/jinhe/MendelGM/wom/models/OLMoE-1B-7B-0125-Instruct")
    parser.add_argument("--actor_dir", default=None, help="veRL FSDP actor checkpoint dir")
    parser.add_argument("--hf_model", default=None, help="Pre-merged HF model path")
    parser.add_argument("--merged_dir", default=None, help="Where to write merged HF weights")
    parser.add_argument("--output_dir", default="/mnt/vast/home/ym56kacy/jinhe/MendelGM/wom/outputs/pilot_v2_eval")
    parser.add_argument("--max_samples", type=int, default=None, help="Subsample test set (default: all)")
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--gpu_mem_util", type=float, default=0.45)
    parser.add_argument("--training_step", type=int, default=None)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.hf_model:
        model_path = args.hf_model
    elif args.actor_dir:
        merged_dir = args.merged_dir or str(out_root / "merged_models" / args.name)
        model_path = merge_fsdp_actor(args.actor_dir, args.base_model, merged_dir)
    else:
        raise ValueError("Provide --actor_dir or --hf_model")

    df = load_test_data(args.test_file, args.max_samples)
    t0 = time.time()
    outputs = run_vllm_eval(
        model_path,
        df,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem_util,
    )
    metrics = score_outputs(df, outputs)
    elapsed = time.time() - t0

    result = {
        "name": args.name,
        "model_path": model_path,
        "actor_dir": args.actor_dir,
        "training_step": args.training_step,
        "notes": args.notes,
        "test_file": args.test_file,
        "max_samples": len(df),
        "elapsed_sec": elapsed,
        **metrics,
    }

    result_path = out_root / f"{args.name}.json"
    gen_path = out_root / f"{args.name}_generations.jsonl"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    with open(gen_path, "w") as f:
        for i, ((_, row), output) in enumerate(zip(df.iterrows(), outputs)):
            rm = row["reward_model"]
            gt = rm["ground_truth"] if isinstance(rm, dict) else rm
            rec = {
                "idx": i,
                "data_source": row["data_source"],
                "ground_truth": gt,
                "output": output,
                "score": default_compute_score(row["data_source"], output, gt),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(json.dumps(result, indent=2))
    print(f"Saved: {result_path}")
    print(f"Generations: {gen_path}")


if __name__ == "__main__":
    main()
