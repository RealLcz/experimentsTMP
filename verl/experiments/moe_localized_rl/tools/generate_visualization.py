#!/usr/bin/env python
"""Generate comparison visualization of Full vs MoE-only GSPO training."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os

OUTPUT_DIR = "/mnt/vast/home/ym56kacy/jinhe/MendelGM/wom/verl/experiments/moe_localized_rl/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FULL_SMOKE = "/mnt/vast/home/ym56kacy/jinhe/MendelGM/wom/verl/tensorboard_log/moe_localized_rl_smoke/olmoe_full_smoke"
MOE_SMOKE = "/mnt/vast/home/ym56kacy/jinhe/MendelGM/wom/verl/tensorboard_log/moe_localized_rl_smoke/olmoe_moe_smoke"
FULL_PILOT = "/mnt/vast/home/ym56kacy/jinhe/MendelGM/wom/verl/tensorboard_log/moe_localized_rl_pilot/olmoe_gspo_full_seed0"
MOE_PILOT = "/mnt/vast/home/ym56kacy/jinhe/MendelGM/wom/verl/tensorboard_log/moe_localized_rl_pilot/olmoe_gspo_moe_seed0"


def load_metric(path, tag):
    ea = EventAccumulator(path, size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return [], []
    vals = ea.Scalars(tag)
    return [v.step for v in vals], [v.value for v in vals]


def load_val_metric(path):
    ea = EventAccumulator(path, size_guidance={"scalars": 0})
    ea.Reload()
    for tag in ea.Tags().get("scalars", []):
        if "val" in tag and "reward" in tag and "mean" in tag:
            vals = ea.Scalars(tag)
            return [v.step for v in vals], [v.value for v in vals]
    return [], []


# ============================================================
# Figure 1: Pilot run - Grad Norm comparison (the "explosion")
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Pilot Run: Full GSPO vs MoE-only GSPO (OLMoE-1B-7B + GSM8K)", fontsize=16, fontweight="bold")

# --- (0,0) Grad Norm - log scale ---
ax = axes[0, 0]
sx, sy = load_metric(FULL_PILOT, "actor/grad_norm")
ax.plot(sx, sy, "o-", color="#e74c3c", label="Full GSPO", markersize=4, linewidth=1.5)
sx, sy = load_metric(MOE_PILOT, "actor/grad_norm")
ax.plot(sx, sy, "s-", color="#2ecc71", label="MoE-only GSPO", markersize=4, linewidth=1.5)
ax.set_yscale("log")
ax.set_xlabel("Training Step")
ax.set_ylabel("Gradient Norm (log scale)")
ax.set_title("Gradient Norm - Full EXPLODED at step ~28")
ax.legend()
ax.grid(True, alpha=0.3, which="both")
ax.axhline(y=100, color="gray", linestyle="--", alpha=0.5, label="_")
ax.annotate("GRADIENT EXPLOSION\n(grad_norm = 1356)", xy=(28, 1356), xytext=(15, 5000),
            fontsize=10, color="red", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="red", lw=2))
ax.annotate("STABLE\n(grad_norm < 6)", xy=(90, 0.5), xytext=(60, 0.01),
            fontsize=10, color="green", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="green", lw=2))

# --- (0,1) Training Reward ---
ax = axes[0, 1]
sx, sy = load_metric(FULL_PILOT, "critic/score/mean")
ax.plot(sx, sy, "o-", color="#e74c3c", label="Full GSPO", markersize=4, linewidth=1.5)
sx, sy = load_metric(MOE_PILOT, "critic/score/mean")
ax.plot(sx, sy, "s-", color="#2ecc71", label="MoE-only GSPO", markersize=4, linewidth=1.5)
ax.set_xlabel("Training Step")
ax.set_ylabel("Training Reward (mean)")
ax.set_title("Training Reward - Full collapsed after step 28")
ax.legend()
ax.grid(True, alpha=0.3)
ax.annotate("Full CRASHED\n(reward dropped)", xy=(30, 0.66), xytext=(33, 0.2),
            fontsize=10, color="red", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="red", lw=2))
ax.annotate("MoE STABLE\n(reward = 0.80)", xy=(95, 0.80), xytext=(70, 0.35),
            fontsize=10, color="green", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="green", lw=2))

# --- (1,0) Validation Accuracy ---
ax = axes[1, 0]
sx, sy = load_val_metric(FULL_PILOT)
ax.plot(sx, sy, "o-", color="#e74c3c", label="Full GSPO", markersize=6, linewidth=2)
sx, sy = load_val_metric(MOE_PILOT)
ax.plot(sx, sy, "s-", color="#2ecc71", label="MoE-only GSPO", markersize=6, linewidth=2)
ax.axhline(y=0.357, color="gray", linestyle=":", alpha=0.7, label="Baseline (35.7%)")
ax.set_xlabel("Training Step")
ax.set_ylabel("GSM8K Validation Accuracy")
ax.set_title("Validation Accuracy - MoE-only reached 76.1%")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0.3, 0.85)
ax.annotate("MoE peak: 76.1%", xy=(65, 0.761), xytext=(75, 0.82),
            fontsize=10, color="green", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="green", lw=2))

# --- (1,1) Response Length ---
ax = axes[1, 1]
sx, sy = load_metric(FULL_PILOT, "response_length/mean")
ax.plot(sx, sy, "o-", color="#e74c3c", label="Full GSPO", markersize=4, linewidth=1.5)
sx, sy = load_metric(MOE_PILOT, "response_length/mean")
ax.plot(sx, sy, "s-", color="#2ecc71", label="MoE-only GSPO", markersize=4, linewidth=1.5)
ax.set_xlabel("Training Step")
ax.set_ylabel("Response Length (tokens)")
ax.set_title("Response Length - Full generated degenerate output")
ax.legend()
ax.grid(True, alpha=0.3)
ax.annotate("Full COLLAPSED\n(generating ~72 tokens)", xy=(30, 72), xytext=(35, 150),
            fontsize=10, color="red", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="red", lw=2))
ax.annotate("MoE STABLE\n(~280 tokens)", xy=(95, 284), xytext=(70, 180),
            fontsize=10, color="green", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="green", lw=2))

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(OUTPUT_DIR, "pilot_comparison_4panel.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUTPUT_DIR}/pilot_comparison_4panel.png")


# ============================================================
# Figure 2: Smoke test - the full picture of both runs
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Smoke Test (116 steps): Full GSPO vs MoE-only GSPO", fontsize=16, fontweight="bold")

# --- (0,0) Grad Norm ---
ax = axes[0, 0]
sx, sy = load_metric(FULL_SMOKE, "actor/grad_norm")
ax.plot(sx, sy, "o-", color="#e74c3c", label="Full GSPO", markersize=3, linewidth=1.5)
sx, sy = load_metric(MOE_SMOKE, "actor/grad_norm")
ax.plot(sx, sy, "s-", color="#2ecc71", label="MoE-only GSPO", markersize=3, linewidth=1.5)
ax.set_yscale("log")
ax.set_xlabel("Training Step")
ax.set_ylabel("Gradient Norm (log scale)")
ax.set_title("Gradient Norm - MoE EXPLODED at step ~85 in smoke test")
ax.legend()
ax.grid(True, alpha=0.3, which="both")
ax.annotate("MoE GRADIENT EXPLOSION\n(grad_norm = 23,725!)", xy=(85, 23725), xytext=(50, 100000),
            fontsize=10, color="orange", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="orange", lw=2))

# --- (0,1) Training Reward ---
ax = axes[0, 1]
sx, sy = load_metric(FULL_SMOKE, "critic/score/mean")
ax.plot(sx, sy, "o-", color="#e74c3c", label="Full GSPO", markersize=3, linewidth=1.5)
sx, sy = load_metric(MOE_SMOKE, "critic/score/mean")
ax.plot(sx, sy, "s-", color="#2ecc71", label="MoE-only GSPO", markersize=3, linewidth=1.5)
ax.set_xlabel("Training Step")
ax.set_ylabel("Training Reward (mean)")
ax.set_title("Training Reward - Both learned, MoE collapsed late")
ax.legend()
ax.grid(True, alpha=0.3)
ax.annotate("MoE peaked at 0.88\n(above Full!)", xy=(80, 0.88), xytext=(90, 0.95),
            fontsize=9, color="green", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="green", lw=1.5))
ax.annotate("MoE COLLAPSED\n(reward -> 0)", xy=(105, 0.0), xytext=(80, 0.15),
            fontsize=10, color="orange", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="orange", lw=2))

# --- (1,0) Response Length ---
ax = axes[1, 0]
sx, sy = load_metric(FULL_SMOKE, "response_length/mean")
ax.plot(sx, sy, "o-", color="#e74c3c", label="Full GSPO", markersize=3, linewidth=1.5)
sx, sy = load_metric(MOE_SMOKE, "response_length/mean")
ax.plot(sx, sy, "s-", color="#2ecc71", label="MoE-only GSPO", markersize=3, linewidth=1.5)
ax.set_xlabel("Training Step")
ax.set_ylabel("Response Length (tokens)")
ax.set_title("Response Length - MoE collapsed to 1 token (empty output)")
ax.legend()
ax.grid(True, alpha=0.3)
ax.annotate("MoE generated EMPTY\nresponses (1 token)", xy=(100, 1), xytext=(70, 100),
            fontsize=10, color="orange", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="orange", lw=2))

# --- (1,1) PG Loss ---
ax = axes[1, 1]
sx, sy = load_metric(FULL_SMOKE, "actor/pg_loss")
ax.plot(sx, sy, "o-", color="#e74c3c", label="Full GSPO", markersize=3, linewidth=1.5)
sx, sy = load_metric(MOE_SMOKE, "actor/pg_loss")
ax.plot(sx, sy, "s-", color="#2ecc71", label="MoE-only GSPO", markersize=3, linewidth=1.5)
ax.set_xlabel("Training Step")
ax.set_ylabel("PG Loss")
ax.set_title("Policy Gradient Loss - MoE spiked before collapse")
ax.legend()
ax.grid(True, alpha=0.3)
ax.annotate("Loss SPIKE\n(loss = 154)", xy=(90, 154), xytext=(60, 120),
            fontsize=10, color="orange", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="orange", lw=2))

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(OUTPUT_DIR, "smoke_comparison_4panel.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUTPUT_DIR}/smoke_comparison_4panel.png")


# ============================================================
# Figure 3: Combined timeline - what happened
# ============================================================
fig, ax = plt.subplots(figsize=(16, 8))
ax.set_title("Training Stability Timeline: What 'Gradient Explosion' Looks Like", fontsize=14, fontweight="bold")

# Plot pilot grad norms on log scale
sx, sy = load_metric(FULL_PILOT, "actor/grad_norm")
ax.plot(sx, sy, "o-", color="#e74c3c", label="Full GSPO (Pilot) - EXPLODED at step 28", markersize=5, linewidth=2)
sx, sy = load_metric(MOE_PILOT, "actor/grad_norm")
ax.plot(sx, sy, "s-", color="#2ecc71", label="MoE-only GSPO (Pilot) - STABLE for 99 steps", markersize=4, linewidth=2)

ax.set_yscale("log")
ax.set_xlabel("Training Step", fontsize=12)
ax.set_ylabel("Gradient Norm (log scale)", fontsize=12)
ax.legend(fontsize=12, loc="upper left")
ax.grid(True, alpha=0.3, which="both")
ax.set_xlim(-2, 105)
ax.set_ylim(0.01, 10000)

# Add shaded regions
ax.axvspan(0, 28, alpha=0.05, color="green", label="_")
ax.axvspan(28, 35, alpha=0.15, color="red", label="_")
ax.axvspan(35, 99, alpha=0.05, color="green", label="_")

# Annotations
ax.annotate("NORMAL TRAINING\nBoth models learning", xy=(15, 10), fontsize=11,
            ha="center", color="#555555",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
ax.annotate("Full GSPO\nGRADIENT EXPLOSION\ngrad_norm: 14 -> 1356\n(100x increase!)", 
            xy=(29, 1356), xytext=(45, 3000), fontsize=11, color="red", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="mistyrose", alpha=0.9),
            arrowprops=dict(arrowstyle="->", color="red", lw=2.5))
ax.annotate("MoE-only GSPO\nSTABLE THROUGHOUT\ngrad_norm: 0.4 - 6\n(never exceeded 6!)", 
            xy=(80, 0.5), xytext=(60, 0.05), fontsize=11, color="green", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="honeydew", alpha=0.9),
            arrowprops=dict(arrowstyle="->", color="green", lw=2.5))

# Add horizontal reference lines
ax.axhline(y=10, color="orange", linestyle="--", alpha=0.4, linewidth=1)
ax.text(100.5, 10, "10", fontsize=9, color="orange", va="center")
ax.axhline(y=100, color="red", linestyle="--", alpha=0.4, linewidth=1)
ax.text(100.5, 100, "100", fontsize=9, color="red", va="center")
ax.axhline(y=1000, color="darkred", linestyle="--", alpha=0.4, linewidth=1)
ax.text(100.5, 1000, "1000\n(DANGER)", fontsize=9, color="darkred", va="center")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "stability_timeline.png"), dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUTPUT_DIR}/stability_timeline.png")

print("\nDone! All visualizations saved to:", OUTPUT_DIR)
