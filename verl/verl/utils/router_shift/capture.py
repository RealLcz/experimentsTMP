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

"""Forward-hook based router trace capture for MoE models."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from verl.utils.router_shift.core import stack_layer_traces


def _get_router_hook_targets(model: nn.Module) -> list[tuple[str, nn.Module, str]]:
    """Return (name, module, kind) for router modules we can hook."""
    targets: list[tuple[str, nn.Module, str]] = []

    # GPT-OSS: hook TopKRouter inside each MLP block.
    try:
        from transformers.models.gpt_oss.modeling_gpt_oss import GptOssTopKRouter

        for name, module in model.named_modules():
            if isinstance(module, GptOssTopKRouter):
                targets.append((name, module, "gpt_oss_router"))
    except Exception:
        pass

    # Gemma 4: hook the text router. Its forward output is
    # (full probabilities, post-processed top-k weights, top-k indices). RSPO
    # compares r_phi(e | c), so cache the full probability assigned to each
    # old-policy top-k expert, not the normalized/scaled dispatch weight.
    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRouter

        for name, module in model.named_modules():
            if isinstance(module, Gemma4TextRouter):
                targets.append((name, module, "gemma4_router"))
    except Exception:
        pass

    # OLMoE: hook gate (router) inside sparse MoE blocks.
    try:
        from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock

        for name, module in model.named_modules():
            if isinstance(module, OlmoeSparseMoeBlock) and hasattr(module, "gate"):
                targets.append((name, module, "olmoe_block"))
    except Exception:
        pass

    # Qwen3-MoE: hook the gate (Qwen3MoeTopKRouter) inside each MoE block.
    # The router forward returns (router_logits, router_scores, router_indices).
    # We recompute full softmax probabilities from router_logits for gathering.
    try:
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

        for name, module in model.named_modules():
            if isinstance(module, Qwen3MoeSparseMoeBlock) and hasattr(module, "gate"):
                targets.append((name, module, "qwen3_moe_block"))
    except Exception:
        pass

    return targets


class RouterTraceCapture:
    """Capture per-layer router scores/indices during a forward pass."""

    def __init__(self) -> None:
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._layer_traces: list[dict[str, torch.Tensor]] = []

    def clear(self) -> None:
        self._layer_traces = []

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []
        self.clear()

    def enable(self, model: nn.Module) -> None:
        self.remove()
        targets = _get_router_hook_targets(model)
        if not targets:
            raise ValueError(
                "RouterTraceCapture: no supported MoE router modules found in model. "
                "Register a hook target for this architecture."
            )
        for _name, module, kind in targets:
            if kind == "gpt_oss_router":
                self._hooks.append(module.register_forward_hook(self._gpt_oss_router_hook))
            elif kind == "gemma4_router":
                self._hooks.append(module.register_forward_hook(self._gemma4_router_hook))
            elif kind == "olmoe_block":
                self._hooks.append(module.register_forward_hook(self._olmoe_block_hook))
            elif kind == "qwen3_moe_block":
                self._hooks.append(module.register_forward_hook(self._qwen3_moe_block_hook))

    def _gpt_oss_router_hook(self, _module, _inputs, output) -> None:
        router_scores, router_indices = output
        # Hooks are removed before activation-checkpoint recomputation.  Any
        # autograd-recorded operation here would therefore make the original
        # forward and recompute save a different number of tensors.  Router
        # traces are diagnostics/stop-gradient inputs, so operate only on
        # detached values under no_grad.
        with torch.no_grad():
            router_scores = router_scores.detach()
            router_indices = router_indices.detach()
            scores_at_k = torch.gather(router_scores, dim=-1, index=router_indices)
        self._layer_traces.append(
            {
                "scores": scores_at_k,
                "indices": router_indices,
                "router_scores_full": router_scores,
            }
        )

    def _gemma4_router_hook(self, _module, _inputs, output) -> None:
        if not isinstance(output, tuple) or len(output) != 3:
            raise ValueError(
                "Gemma4TextRouter output changed: expected "
                "(router_probabilities, top_k_weights, top_k_indices)."
            )
        router_probabilities, _top_k_weights, router_indices = output
        with torch.no_grad():
            router_probabilities = router_probabilities.detach()
            router_indices = router_indices.detach()
            scores_at_k = torch.gather(router_probabilities, dim=-1, index=router_indices)
        self._layer_traces.append(
            {
                "scores": scores_at_k,
                "indices": router_indices,
                "router_scores_full": router_probabilities,
            }
        )

    def _olmoe_block_hook(self, module, inputs, output) -> None:
        with torch.no_grad():
            hidden_states = inputs[0].detach()
            _batch_size, _sequence_length, hidden_dim = hidden_states.shape
            hidden_states = hidden_states.view(-1, hidden_dim)
            router_logits = module.gate(hidden_states)
            routing_weights_full = torch.nn.functional.softmax(router_logits, dim=-1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(routing_weights_full, module.top_k, dim=-1)
            routing_weights = routing_weights.to(hidden_states.dtype)
        self._layer_traces.append(
            {
                "scores": routing_weights,
                "indices": selected_experts,
                "router_scores_full": routing_weights_full.to(hidden_states.dtype),
            }
        )

    def _qwen3_moe_block_hook(self, module, inputs, output) -> None:
        """Qwen3-MoE: recompute full softmax from router logits for gathering.

        Qwen3MoeTopKRouter.forward returns (router_logits, router_scores, router_indices).
        router_scores are already normalized by top-k sum, but RSPO needs the raw
        softmax probability for each expert so we can gather current scores at old
        expert indices. We recompute from router_logits.
        """
        with torch.no_grad():
            hidden_states = inputs[0].detach()
            _batch_size, _sequence_length, hidden_dim = hidden_states.shape
            hidden_states = hidden_states.view(-1, hidden_dim)
            gate_output = module.gate(hidden_states)
            # Qwen3MoeTopKRouter returns (router_logits, router_scores, router_indices)
            if isinstance(gate_output, tuple):
                router_logits = gate_output[0]
            else:
                router_logits = gate_output
            router_probs_full = torch.nn.functional.softmax(router_logits, dim=-1, dtype=torch.float)
            top_k = module.gate.top_k
            routing_weights, selected_experts = torch.topk(router_probs_full, top_k, dim=-1)
            routing_weights = routing_weights.to(hidden_states.dtype)
        self._layer_traces.append(
            {
                "scores": routing_weights,
                "indices": selected_experts,
                "router_scores_full": router_probs_full.to(hidden_states.dtype),
            }
        )

    def get_trace(self) -> Optional[dict[str, torch.Tensor]]:
        if not self._layer_traces:
            return None
        scores, indices = stack_layer_traces(self._layer_traces)
        return {"router_scores": scores, "router_indices": indices}

    def select_token_rows(self, row_indices: torch.Tensor) -> None:
        """Keep the same flattened token rows in every captured layer.

        Padded SDPA forwards flatten ``[batch, sequence]`` inside each MoE
        router, so hooks also see padding rows. Callers remove those rows
        before constructing jagged traces or gathering current probabilities
        at old-policy expert indices.
        """
        if row_indices.ndim != 1 or row_indices.dtype != torch.long:
            raise ValueError("row_indices must be a one-dimensional torch.long tensor")
        for layer_idx, trace in enumerate(self._layer_traces):
            for key in ("scores", "indices", "router_scores_full"):
                value = trace.get(key)
                if value is None:
                    continue
                if value.ndim < 1:
                    raise ValueError(f"Captured router {key} at layer {layer_idx} has no token dimension")
                if row_indices.numel() and int(row_indices.max()) >= value.shape[0]:
                    raise ValueError(
                        f"Router row selection exceeds layer {layer_idx} {key} rows: "
                        f"max_index={int(row_indices.max())}, rows={value.shape[0]}"
                    )
                trace[key] = value.index_select(0, row_indices.to(value.device))

    def get_current_scores_at_old_indices(
        self,
        old_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Gather current scores at old expert indices from the latest forward.

        Args:
            old_indices: (total_nnz, L, K) expert indices from old policy.

        Returns:
            (total_nnz, L, K) scores at those indices.
        """
        if len(self._layer_traces) != old_indices.shape[-2]:
            raise ValueError(
                f"Layer count mismatch: captured {len(self._layer_traces)} layers, "
                f"old_indices has {old_indices.shape[-2]}."
            )
        gathered = []
        for layer_idx, trace in enumerate(self._layer_traces):
            router_scores_full = trace.get("router_scores_full")
            if router_scores_full is not None:
                idx = old_indices[:, layer_idx, :]
                gathered.append(torch.gather(router_scores_full, dim=-1, index=idx))
            else:
                # OLMoE path: recompute from stored top-k (indices must match)
                gathered.append(trace["scores"])
        return torch.stack(gathered, dim=1)
