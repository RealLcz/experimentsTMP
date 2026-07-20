"""One-GPU smoke test for Gemma 4 router hooks after FSDP2 wrapping."""

import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

from verl.utils.router_shift.capture import RouterTraceCapture, _get_router_hook_targets


def main() -> None:
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    config = Gemma4TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        global_head_dim=8,
        num_global_key_value_heads=2,
        vocab_size_per_layer_input=64,
        hidden_size_per_layer_input=0,
        enable_moe_block=True,
        num_experts=4,
        top_k_experts=2,
        moe_intermediate_size=16,
        layer_types=["full_attention", "full_attention"],
        rope_parameters={"full_attention": {"rope_type": "default", "rope_theta": 10000.0}},
    )
    config._attn_implementation = "eager"
    model = Gemma4ForCausalLM(config).cuda()
    mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("fsdp",))
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for layer in model.model.layers:
        fully_shard(layer, mesh=mesh, mp_policy=policy)
    fully_shard(model, mesh=mesh, mp_policy=policy)

    targets = _get_router_hook_targets(model)
    print("TARGETS", [(name, type(module).__name__) for name, module, _ in targets], flush=True)
    capture = RouterTraceCapture()
    capture.enable(model)
    model(input_ids=torch.randint(0, 64, (2, 5), device="cuda"), use_cache=False)
    trace = capture.get_trace()
    if trace is None:
        raise RuntimeError("FSDP2 Gemma4 router hooks did not fire")
    print("TRACE", {key: tuple(value.shape) for key, value in trace.items()}, flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
