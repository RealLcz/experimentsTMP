#!/usr/bin/env python
"""Generate training curves and summary from veRL training logs.

Reads the tensorboard event files from a training output directory and
generates:
    - reward_curve.png
    - kl_curve.png
    - grad_norm_curve.png
    - gpu_memory.csv
    - training_summary.json

Usage:
    python plot_training_curves.py --output_dir /path/to/outputs/pilot_full
    python plot_training_curves.py --output_dir outputs/pilot_full --output_dir2 outputs/pilot_moe
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_tensorboard_logs(output_dir: str) -> dict[str, list[tuple[int, float]]]:
    """Read scalar metrics from tensorboard event files.

    Returns a dict: metric_name -> [(step, value), ...]
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        print("tensorboard not available, trying alternative parser")
        return _read_logs_alternative(output_dir)

    metrics = {}
    # Search for event files in the output directory.
    event_files = list(Path(output_dir).rglob("events.out.tfevents.*"))
    if not event_files:
        print(f"No tensorboard event files found in {output_dir}")
        return metrics

    for ef in event_files:
        ea = EventAccumulator(str(ef.parent), size_guidance={"scalars": 0})
        ea.Reload()
        for tag in ea.Tags().get("scalars", []):
            events = ea.Scalars(tag)
            if tag not in metrics:
                metrics[tag] = []
            for e in events:
                metrics[tag].append((e.step, e.value))
    # Sort by step.
    for tag in metrics:
        metrics[tag].sort(key=lambda x: x[0])
    return metrics


def _read_logs_alternative(output_dir: str) -> dict[str, list[tuple[int, float]]]:
    """Fallback: parse the SLURM log for training metrics."""
    metrics = {}
    # Look for monitor.json or similar.
    for f in Path(output_dir).rglob("monitor.json"):
        with open(f) as fh:
            for line in fh:
                try:
                    entry = json.loads(line)
                    step = entry.get("step", 0)
                    for k, v in entry.items():
                        if isinstance(v, (int, float)) and k != "step":
                            metrics.setdefault(k, []).append((step, float(v)))
                except json.JSONDecodeError:
                    continue
    return metrics


def plot_curve(metrics: dict, tag: str, output_path: str, title: str = None, ylabel: str = None):
    if tag not in metrics or not metrics[tag]:
        print(f"  [skip] no data for {tag}")
        return
    steps, values = zip(*metrics[tag])
    plt.figure(figsize=(10, 6))
    plt.plot(steps, values, marker="o", markersize=3, linewidth=1.5)
    plt.xlabel("Step")
    plt.ylabel(ylabel or tag)
    plt.title(title or tag)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  [saved] {output_path} ({len(steps)} points)")


def generate_summary(metrics: dict, output_path: str):
    summary = {}
    for tag, vals in metrics.items():
        if not vals:
            continue
        steps, values = zip(*vals)
        summary[tag] = {
            "num_points": len(vals),
            "first_step": steps[0],
            "last_step": steps[-1],
            "first_value": values[0],
            "last_value": values[-1],
            "min_value": min(values),
            "max_value": max(values),
        }
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  [saved] {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True, help="training output directory")
    parser.add_argument("--output_dir2", default=None, help="second output directory for comparison")
    args = parser.parse_args()

    print(f"Reading metrics from {args.output_dir}...")
    metrics1 = read_tensorboard_logs(args.output_dir)
    metrics2 = read_tensorboard_logs(args.output_dir2) if args.output_dir2 else {}

    # Plot individual curves.
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    metric_map = {
        "reward/mean": ("reward_curve.png", "Reward Mean", "reward"),
        "reward/std": ("reward_std_curve.png", "Reward Std", "reward std"),
        "actor/pg_loss": ("pg_loss_curve.png", "PG Loss", "loss"),
        "actor/pg_clipfrac": ("pg_clipfrac_curve.png", "PG Clip Fraction", "clip frac"),
        "actor/ppo_kl": ("kl_curve.png", "PPO KL", "KL"),
        "actor/grad_norm": ("grad_norm_curve.png", "Grad Norm", "grad norm"),
        "response_length/mean": ("response_length_curve.png", "Response Length", "tokens"),
    }

    print("\nGenerating plots...")
    for tag, (fname, title, ylabel) in metric_map.items():
        # Try exact tag and also search for partial matches.
        matched_tags = [t for t in metrics1 if tag in t.lower() or tag.lower() in t.lower()]
        for mt in matched_tags:
            plot_curve(metrics1, mt, os.path.join(plots_dir, fname), title=title, ylabel=ylabel)
            break
        else:
            print(f"  [skip] no data for {tag}")

    # Comparison plot for reward if both dirs provided.
    if metrics2:
        print("\nGenerating comparison plots...")
        for tag in metrics1:
            if "reward" in tag.lower() and tag in metrics2:
                plt.figure(figsize=(10, 6))
                s1, v1 = zip(*metrics1[tag])
                plt.plot(s1, v1, marker="o", markersize=3, label="full", linewidth=1.5)
                s2, v2 = zip(*metrics2[tag])
                plt.plot(s2, v2, marker="s", markersize=3, label="moe-only", linewidth=1.5)
                plt.xlabel("Step")
                plt.ylabel(tag)
                plt.title(f"{tag} Comparison")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                fname = f"comparison_{tag.replace('/', '_')}.png"
                plt.savefig(os.path.join(plots_dir, fname), dpi=150)
                plt.close()
                print(f"  [saved] {fname}")

    # Generate summary.
    print("\nGenerating summary...")
    generate_summary(metrics1, os.path.join(args.output_dir, "training_summary.json"))
    if metrics2:
        generate_summary(metrics2, os.path.join(args.output_dir2, "training_summary.json"))

    print("\nDone!")


if __name__ == "__main__":
    main()
