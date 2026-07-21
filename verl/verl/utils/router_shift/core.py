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

"""Pure functions for RSPO router-shift weight computation.

Implements Eq. (5)-(9) and (11) from the RSPO paper:
  d^{(l)}_{i,t} = (1/K) sum_k |log r^{(l)}_phi(e^k) - log r^{(l)}_{phi_old}(e^k)|
  Delta_{i,t} = (1/L) sum_l d^{(l)}_{i,t}
  gamma_{i,t} = exp(-Delta_{i,t})
  gamma_tilde_{i,t} = sg[max(gamma_{i,t}, gamma_min)]
  w_tilde_{i,t} = w_{i,t} * gamma_tilde_{i,t}
"""

from __future__ import annotations

from typing import Any

import torch

_EPS = 1e-8


def compute_layer_router_deviation(
    old_scores: torch.Tensor,
    current_scores: torch.Tensor,
) -> torch.Tensor:
    """Per-layer routing deviation d^{(l)}_{i,t}.

    Args:
        old_scores: Routing scores at old activated experts, shape (..., K).
        current_scores: Current routing scores at the same expert indices, shape (..., K).

    Returns:
        Layer deviation of shape ``old_scores.shape[:-1]``.
    """
    old_log = torch.log(old_scores.clamp(min=_EPS))
    current_log = torch.log(current_scores.clamp(min=_EPS))
    return torch.abs(current_log - old_log).mean(dim=-1)


def compute_router_shift_ratio(layer_deviations: torch.Tensor) -> torch.Tensor:
    """Aggregate layer deviations and compute gamma = exp(-Delta).

    Args:
        layer_deviations: Per-layer deviations, shape (..., L).

    Returns:
        Router-shift ratio gamma in (0, 1], shape ``layer_deviations.shape[:-1]``.
    """
    delta = layer_deviations.mean(dim=-1)
    return torch.exp(-delta)


def process_router_shift_weight(
    gamma: torch.Tensor,
    gamma_min: float = 0.8,
    stop_gradient: bool = True,
) -> torch.Tensor:
    """Apply floor and stop-gradient to obtain gamma_tilde."""
    weight = torch.clamp(gamma, min=gamma_min)
    if stop_gradient:
        weight = weight.detach()
    return weight


def compute_router_shift_weight(
    old_router_scores: torch.Tensor,
    current_router_scores: torch.Tensor,
    *,
    gamma_min: float = 0.8,
    stop_gradient: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-token router-shift weight from old/current router scores.

    Args:
        old_router_scores: Shape (batch, seq_len, num_layers, top_k).
        current_router_scores: Same shape, scores at old expert indices.

    Returns:
        (gamma_tilde, delta) both shape (batch, seq_len).
    """
    layer_dev = compute_layer_router_deviation(old_router_scores, current_router_scores)
    gamma = compute_router_shift_ratio(layer_dev)
    gamma_tilde = process_router_shift_weight(gamma, gamma_min=gamma_min, stop_gradient=stop_gradient)
    delta = layer_dev.mean(dim=-1)
    return gamma_tilde, delta


def apply_router_shift_to_log_ratio(
    log_importance_ratio: torch.Tensor,
    router_shift_weight: torch.Tensor,
) -> torch.Tensor:
    """Rescale log importance ratio: log w_tilde = log w + log gamma_tilde."""
    return log_importance_ratio + torch.log(router_shift_weight.clamp(min=_EPS))


def gather_current_scores_at_old_indices(
    current_full_scores: torch.Tensor,
    old_indices: torch.Tensor,
) -> torch.Tensor:
    """Gather current router scores at old activated expert indices.

    Args:
        current_full_scores: (total_nnz, num_experts) or (batch, seq, num_experts).
        old_indices: Same leading dims + top_k.

    Returns:
        Scores at old indices, shape ``old_indices.shape``.
    """
    return torch.gather(current_full_scores, dim=-1, index=old_indices)


def stack_layer_traces(layer_traces: list[dict[str, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack per-layer hook outputs into (total_nnz, L, K) tensors."""
    if not layer_traces:
        raise ValueError("No router traces captured.")
    scores = torch.stack([t["scores"] for t in layer_traces], dim=-1)  # (nnz, K, L)
    indices = torch.stack([t["indices"] for t in layer_traces], dim=-1)
    scores = scores.transpose(-1, -2)  # (nnz, L, K)
    indices = indices.transpose(-1, -2)
    return scores, indices


def compute_router_shift_metrics(
    gamma_raw: torch.Tensor,
    gamma_tilde: torch.Tensor,
    delta: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    gamma_min: float = 0.8,
) -> dict[str, Any]:
    """Compute RSPO diagnostics for TensorBoard logging.

    Args:
        gamma_raw: Unclamped gamma (for floor_fraction).
        gamma_tilde: Clamped gamma used as the actual weight (for weight_mean etc).
        delta: Per-layer router deviation.
        response_mask: Boolean mask over (batch, seq_len).
        gamma_min: Floor threshold for floor_fraction computation.
    """
    mask = response_mask.to(dtype=torch.bool)
    if mask.sum() == 0:
        return {}

    gamma_tilde_flat = gamma_tilde[mask].float()
    gamma_raw_flat = gamma_raw[mask].float()
    delta_flat = delta[mask].float()
    floor_frac = (gamma_raw_flat < gamma_min).float().mean()

    def _percentile(x: torch.Tensor, q: float) -> float:
        return float(torch.quantile(x, q).item())

    return {
        "router_shift/raw_gamma_mean": float(gamma_raw_flat.mean().item()),
        "router_shift/weight_mean": float(gamma_tilde_flat.mean().item()),
        "router_shift/weight_min": float(gamma_tilde_flat.min().item()),
        "router_shift/weight_p10": _percentile(gamma_tilde_flat, 0.10),
        "router_shift/weight_p50": _percentile(gamma_tilde_flat, 0.50),
        "router_shift/floor_fraction": float(floor_frac.item()),
        "router_shift/deviation_mean": float(delta_flat.mean().item()),
        "router_shift/deviation_p90": _percentile(delta_flat, 0.90),
        "router_shift/deviation_p99": _percentile(delta_flat, 0.99),
    }
