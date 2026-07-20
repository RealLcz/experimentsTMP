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

import torch
import torch.nn as nn

from verl.utils.router_shift.core import (
    apply_router_shift_to_log_ratio,
    compute_router_shift_weight,
    process_router_shift_weight,
)
from verl.utils.router_shift.capture import RouterTraceCapture


def test_identical_routing_gives_gamma_one():
    old = torch.tensor([[[[0.5, 0.5], [0.4, 0.6]]]])
    current = old.clone()
    gamma_tilde, _ = compute_router_shift_weight(old, current, gamma_min=0.8)
    assert torch.allclose(gamma_tilde, torch.ones_like(gamma_tilde), atol=1e-5)


def test_large_deviation_decreases_gamma():
    old = torch.tensor([[[[0.9, 0.1]]]])
    current = torch.tensor([[[[0.1, 0.9]]]])
    gamma_tilde, _ = compute_router_shift_weight(old, current, gamma_min=0.0)
    assert gamma_tilde.item() < 0.5


def test_gamma_floor():
    old = torch.tensor([[[[0.99, 0.01]]]])
    current = torch.tensor([[[[0.01, 0.99]]]])
    gamma_tilde, _ = compute_router_shift_weight(old, current, gamma_min=0.8)
    assert gamma_tilde.item() >= 0.8


def test_stop_gradient():
    old = torch.tensor([[[[0.5, 0.5]]]], requires_grad=True)
    current = torch.tensor([[[[0.4, 0.6]]]], requires_grad=True)
    gamma_tilde, _ = compute_router_shift_weight(old, current, gamma_min=0.8, stop_gradient=True)
    assert not gamma_tilde.requires_grad


def test_apply_router_shift_in_log_space():
    log_ratio = torch.tensor([[0.5, -0.2]])
    weight = torch.tensor([[0.9, 0.8]])
    adjusted = apply_router_shift_to_log_ratio(log_ratio, weight)
    expected = log_ratio + torch.log(weight)
    assert torch.allclose(adjusted, expected)


def test_gemma4_router_trace_uses_raw_router_probabilities():
    gemma4 = __import__(
        "transformers.models.gemma4.modeling_gemma4",
        fromlist=["Gemma4TextRouter"],
    )
    config_module = __import__(
        "transformers.models.gemma4.configuration_gemma4",
        fromlist=["Gemma4TextConfig"],
    )
    config = config_module.Gemma4TextConfig(
        hidden_size=8,
        num_experts=4,
        top_k_experts=2,
        rms_norm_eps=1e-6,
    )

    class TinyGemma4Routers(nn.Module):
        def __init__(self):
            super().__init__()
            self.routers = nn.ModuleList([gemma4.Gemma4TextRouter(config) for _ in range(2)])

        def forward(self, x):
            return [router(x) for router in self.routers]

    model = TinyGemma4Routers()
    capture = RouterTraceCapture()
    capture.enable(model)
    outputs = model(torch.randn(5, config.hidden_size))
    trace = capture.get_trace()
    assert trace is not None
    assert trace["router_scores"].shape == (5, 2, 2)
    assert trace["router_indices"].shape == (5, 2, 2)

    for layer, (probabilities, dispatch_weights, indices) in enumerate(outputs):
        expected = torch.gather(probabilities, -1, indices)
        assert torch.allclose(trace["router_scores"][:, layer], expected)
        # Gemma 4 normalizes/scales dispatch weights after top-k; RSPO must not
        # accidentally cache those post-processed expert weights.
        assert not torch.allclose(expected, dispatch_weights)

    current_at_old = capture.get_current_scores_at_old_indices(trace["router_indices"])
    assert torch.allclose(current_at_old, trace["router_scores"])
    capture.remove()


def test_router_capture_does_not_break_activation_checkpoint_recompute():
    gemma4 = __import__(
        "transformers.models.gemma4.modeling_gemma4",
        fromlist=["Gemma4TextRouter"],
    )
    config_module = __import__(
        "transformers.models.gemma4.configuration_gemma4",
        fromlist=["Gemma4TextConfig"],
    )
    config = config_module.Gemma4TextConfig(
        hidden_size=8,
        num_experts=4,
        top_k_experts=2,
        rms_norm_eps=1e-6,
    )

    class CheckpointedRouter(nn.Module):
        def __init__(self):
            super().__init__()
            self.router = gemma4.Gemma4TextRouter(config)

        def forward(self, x):
            _probabilities, dispatch_weights, _indices = self.router(x)
            return dispatch_weights.square().sum()

    model = CheckpointedRouter()
    capture = RouterTraceCapture()
    capture.enable(model)
    x = torch.randn(5, config.hidden_size, requires_grad=True)
    loss = torch.utils.checkpoint.checkpoint(model, x, use_reentrant=False)
    assert capture.get_trace() is not None
    # The production path removes hooks after forward and before checkpoint
    # recomputation during backward.
    capture.remove()
    loss.backward()
    assert x.grad is not None


def test_router_capture_select_token_rows_compacts_all_trace_fields():
    capture = RouterTraceCapture()
    capture._layer_traces = [
        {
            "scores": torch.arange(12, dtype=torch.float32).reshape(6, 2),
            "indices": torch.arange(12, dtype=torch.long).reshape(6, 2),
            "router_scores_full": torch.arange(30, dtype=torch.float32).reshape(6, 5),
        }
    ]

    capture.select_token_rows(torch.tensor([0, 1, 3, 4], dtype=torch.long))

    trace = capture.get_trace()
    assert trace["router_scores"].shape == (4, 1, 2)
    assert trace["router_indices"].shape == (4, 1, 2)
    old_indices = torch.zeros((4, 1, 2), dtype=torch.long)
    current = capture.get_current_scores_at_old_indices(old_indices)
    assert current.shape == (4, 1, 2)
    assert torch.equal(current[:, 0, 0], torch.tensor([0.0, 5.0, 15.0, 20.0]))


def test_disabled_equivalence_zero_shift():
    """gamma=1 should not change log ratio."""
    log_ratio = torch.tensor([[0.3, -0.1]])
    weight = torch.ones_like(log_ratio)
    adjusted = apply_router_shift_to_log_ratio(log_ratio, weight)
    assert torch.allclose(adjusted, log_ratio)
