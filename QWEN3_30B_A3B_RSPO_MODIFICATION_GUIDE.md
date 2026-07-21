# Qwen3-30B-A3B：Attention-only RSPO vs Full RSPO
## 代码修改与实验启动清单

## 1. 目标

将当前仓库从：

```text
Gemma4-26B-A4B
+
Full / Attention-only
+
GSPO + Router Shift
```

切换为：

```text
Qwen3-30B-A3B
+
Full
vs.
Attention-only
+
RSPO / GSPO+RS
```

核心研究问题：

> 在同样的 MoE routing stabilization 下，Full-parameter RL 是否仍然必要，还是仅更新 Attention 就可以获得接近的 RL adaptation？

在正式训练前，需要先解决以下代码与实验定义问题。

---

# 2. 修改总览

## 必须修改

```text
P0-1  新增 Qwen3-MoE Parameter Update Policy
P0-2  新增 Qwen3-MoE Router Trace Capture
P0-3  新建 Qwen3 专用训练 Launcher
P0-4  明确并修正 RSPO objective 定义
```

## 强烈建议修改

```text
P1-1  修复 Router Shift floor_fraction diagnostics
P1-2  清理与 Gemma4 / GPT-OSS 相关的临时 patch
P1-3  补齐 Qwen3-specific correctness tests
P1-4  增加正式 1-step smoke pipeline
```

---

# 3. P0-1：新增 Qwen3-MoE Parameter Update Policy

当前：

```python
MODEL_PARAMETER_POLICIES = {
    "olmoe": ...,
    "gemma4": ...,
    "gpt_oss": ...,
}
```

缺少：

```python
"qwen3_moe"
```

因此：

```text
Qwen3 + full
```

可能因为 unknown model + full 而继续运行。

但是：

```text
Qwen3 + attention_only
```

会被 strict policy 拒绝。

---

## 3.1 需要新增 model detection

建议：

```python
def _is_qwen3_moe_model(model: nn.Module) -> bool:
    try:
        from transformers.models.qwen3_moe.modeling_qwen3_moe import (
            Qwen3MoeForCausalLM,
        )
    except Exception:
        return False

    return isinstance(model, Qwen3MoeForCausalLM) or any(
        isinstance(m, Qwen3MoeForCausalLM)
        for m in model.modules()
    )
```

---

## 3.2 需要新增 parameter policy

建议支持：

```text
full
moe
experts_only
attention_only
```

目标定义：

### Full

```text
Attention       Trainable
Experts         Trainable
Router          Trainable
Norm            Trainable
Embedding       Trainable
LM Head         Trainable
```

### Attention-only

```text
Attention       Trainable

Experts         Frozen
Router          Frozen
Norm            Frozen
Embedding       Frozen
LM Head         Frozen
```

建议逻辑：

```python
if mode == "full":
    for p in model.parameters():
        p.requires_grad = True
else:
    for p in model.parameters():
        p.requires_grad = False
```

然后：

```python
if mode == "attention_only":
    for module in model.modules():
        if isinstance(module, Qwen3MoeAttention):
            _set_requires_grad(module, True)
```

对于：

```text
moe
experts_only
```

识别：

```text
Qwen3MoeSparseMoeBlock
Router / Gate
Experts
```

并分别打开。

---

## 3.3 注册到 Registry

新增：

```python
MODEL_PARAMETER_POLICIES["qwen3_moe"] = {
    "is_model": _is_qwen3_moe_model,
    "apply": apply_qwen3_moe_policy,
}
```

---

## 3.4 Audit 必须覆盖

至少统计：

```text
attention
experts
router
norm
embedding
lm_head
```

Attention-only 必须验证：

```text
attention_trainable > 0

experts_trainable == 0
router_trainable == 0
norm_trainable == 0
embedding_trainable == 0
lm_head_trainable == 0
```

---

# 4. P0-2：新增 Qwen3 Router Trace Capture

当前 RouterTraceCapture 只支持：

```text
GPT-OSS
Gemma4
OLMoE
```

缺少：

```text
Qwen3-MoE
```

因此打开：

```yaml
router_shift:
  enabled: true
```

之后，Qwen3 会直接报：

```text
RouterTraceCapture:
no supported MoE router modules found
```

---

# 5. Qwen3 Router Capture 应该保存什么

RSPO 需要比较：

```text
Old Policy Routing
vs.
Current Policy Routing
```

并且是在：

```text
old policy 当时激活的 experts
```

上比较 routing probability。

因此 Old Policy 至少保存：

```text
old_router_indices
old_router_scores
```

概念维度：

```text
tokens
×
moe_layers
×
top_k
```

Current Policy forward 需要能够得到：

```text
current full router probabilities
```

然后：

```text
gather current probabilities
at old_router_indices
```

最终得到：

```text
current_router_scores_at_old_experts
```

---

# 6. Qwen3 Router Capture 推荐实现

建议优先沿用现有 abstraction：

```text
RouterTraceCapture
```

新增：

```text
qwen3_moe_router
```

或者：

```text
qwen3_moe_block
```

hook target。

概念流程：

```text
Qwen3 MoE block
    ↓
Router logits / router probabilities
    ↓
Softmax over all experts
    ↓
Top-K expert indices
    ↓
Old scores at Top-K experts
```

保存：

```python
{
    "scores": scores_at_k,
    "indices": selected_experts,
    "router_scores_full": routing_weights_full,
}
```

这样 current forward 时可以：

```python
torch.gather(
    router_scores_full,
    dim=-1,
    index=old_indices,
)
```

---

# 7. Router Capture 必须满足的测试

## Test 1

Old Policy forward：

```text
router_indices.shape
=
[tokens, num_moe_layers, top_k]
```

---

## Test 2

Old scores：

```text
router_scores.shape
=
[tokens, num_moe_layers, top_k]
```

---

## Test 3

Current forward：

```text
current_scores_at_old_indices
```

shape 与 old scores 完全一致。

---

## Test 4

如果：

```text
old policy == current policy
```

则 Router Shift 应接近：

```text
gamma ≈ 1
```

---

## Test 5

如果人工改变 router scores：

```text
routing deviation ↑
```

则：

```text
gamma ↓
```

---

# 8. P0-3：必须新建 Qwen3 专用 Launcher

不要直接把：

```text
run_pilot_gemma4_full.sh
```

里的：

```text
MODEL_PATH
```

改成 Qwen3。

Gemma4 launcher 中包含 Gemma4-specific patch。

例如：

```bash
data.continuous_token.model_family=gemma4
```

以及：

```bash
actor_rollout_ref.model.use_remove_padding=False
```

以及：

```bash
+actor_rollout_ref.model.override_config.attn_implementation=sdpa
```

这些不应直接继承到 Qwen3。

---

# 9. 建议新增 Launcher

```text
run_pilot_qwen3_full_rs.sh
run_pilot_qwen3_attention_only_rs.sh
```

或者更推荐只维护一个：

```text
run_pilot_qwen3_rs.sh
```

通过：

```bash
PARAM_MODE=full
```

和：

```bash
PARAM_MODE=attention_only
```

切换。

这样两组实验真正只有：

```text
parameter_update.mode
```

不同。

---

# 10. Qwen3 Launcher 建议基础配置

模型：

```bash
MODEL_PATH=/path/to/Qwen3-30B-A3B
```

建议恢复：

```bash
actor_rollout_ref.model.use_remove_padding=True
```

不要继承 Gemma4-specific：

```bash
attn_implementation=sdpa
```

除非 Qwen3 实际 smoke test 证明必须这么做。

如果使用 continuous token：

```bash
data.continuous_token.model_family=qwen3
```

如果：

```text
continuous_token.enable=false
```

则这项不是关键。

---

# 11. P0-4：必须明确“Canonical RSPO”还是“GSPO + RS”

这是当前最重要的 Research Correctness 问题。

你现在的实现：

```python
negative_approx_kl = log_prob - old_log_prob

negative_approx_kl += log(gamma)

sequence_ratio =
exp(
    mean_over_tokens(
        negative_approx_kl
    )
)

GSPO sequence clipping
```

等价于：

```text
token importance ratio
    ↓
× gamma
    ↓
geometric mean over sequence
    ↓
GSPO sequence-level clipping
```

这更接近：

```text
GSPO + Router-Shift Rescaling
```

而不是 canonical RSPO objective。

---

# 12. 两种合法路线

## Route A：继续当前实现

正式命名：

```text
Full GSPO+RS
vs.
Attention-only GSPO+RS
```

不要简单叫：

```text
RSPO
```

研究问题：

> Router-Shift rescaling 作为 GSPO plug-in 时，Attention-only 是否可以保留 Full RL performance？

这条路线改动更小。

---

## Route B：严格复现 Canonical RSPO

新增：

```text
loss_mode=rspo
```

或者单独：

```python
compute_policy_loss_rspo(...)
```

严格按照论文 objective 实现。

概念流程：

```text
token importance ratio
    ↓
advantage-sign-aware token clipping
    ↓
router shift gamma
    ↓
geometric mean aggregation
    ↓
sequence-level objective
```

不要继续把：

```text
gamma
```

简单乘进 GSPO sequence ratio 后再做 sequence clipping。

如果论文实验标题要写：

```text
Attention-only RSPO
vs.
Full RSPO
```

建议走 Route B。

---

# 13. 推荐选择

如果当前目标是：

```text
快速验证 Attention-only 是否有效
```

建议：

```text
Route A:
GSPO+RS
```

因为已有大部分代码。

如果目标是：

```text
复现 RSPO
+
做正式 paper experiment
```

建议：

```text
Route B:
Canonical RSPO
```

最重要的是：

> 不要让代码做 GSPO+RS，但论文里写成 RSPO。

---

# 14. P1-1：修复 Router Shift floor_fraction

当前逻辑：

```python
gamma_tilde = clamp(
    gamma,
    min=gamma_min,
)
```

然后 metrics 又做：

```python
floor_fraction =
(gamma < gamma_min).mean()
```

如果传进 metrics 的是已经 clamp 后的：

```text
gamma_tilde
```

那么：

```text
floor_fraction
```

理论上永远接近：

```text
0
```

---

# 15. 正确设计

建议：

```python
gamma_raw
gamma_tilde
delta
```

全部保留。

例如：

```python
return gamma_raw, gamma_tilde, delta
```

然后：

```python
floor_fraction = (
    gamma_raw < gamma_min
).float().mean()
```

日志：

```text
router_shift/raw_gamma_mean
router_shift/weight_mean
router_shift/floor_fraction
router_shift/deviation_mean
```

---

# 16. P1-2：清理 Gemma4 / GPT-OSS 临时 patch

切换 Qwen3 后，以下代码不要继续影响主实验。

包括：

```text
Gemma4-specific SDPA workaround
Gemma4 remove_padding workaround
Gemma4 continuous token model_family
GPT-OSS-specific router hook
临时 Cursor debug log
临时 absolute debug path
```

保留这些 adapter 没问题。

但 Qwen3 launcher 不应该默认继承它们。

---

# 17. P1-3：补齐 Qwen3 Parameter Correctness Tests

至少增加：

```text
test_qwen3_full_requires_grad
test_qwen3_attention_only_requires_grad
```

Attention-only：

```text
Attention grad != None

Experts grad == None
Router grad == None
Norm grad == None
Embedding grad == None
LM Head grad == None
```

Checkpoint delta：

```text
Attention delta > 0

Experts delta == 0
Router delta == 0
Norm delta == 0
Embedding delta == 0
LM Head delta == 0
```

---

# 18. P1-4：增加 Qwen3 Router Correctness Tests

建议独立测试：

```text
Qwen3 model
    ↓
RouterTraceCapture.enable()
    ↓
Forward
    ↓
get_trace()
```

验证：

```text
number of captured layers
==
number of MoE layers
```

并验证：

```text
indices.max() < num_experts
```

---

# 19. 正式训练前必须通过的 Gate

## Gate 0：环境检查

确认：

```text
Transformers
vLLM
PyTorch
veRL
```

可以加载 Qwen3-30B-A3B。

---

## Gate 1：Standalone vLLM

```text
Qwen3-30B-A3B
+
vLLM
```

成功生成一条 response。

---

## Gate 2：Vanilla veRL

```text
Qwen3
+
Full
+
GSPO
+
RS disabled
```

运行：

```text
1 training step
```

必须成功完成：

```text
rollout
→ reward
→ old_log_prob
→ actor forward
→ loss
→ backward
→ optimizer.step
```

如果 Gate 2 失败：

> 问题与 RS 无关。

---

## Gate 3：Router Capture

打开：

```text
RouterTraceCapture
```

只检查：

```text
old router trace
current router trace
```

不要先跑正式训练。

---

## Gate 4：Full + RS

```text
Qwen3
+
Full
+
GSPO+RS / RSPO
```

只跑：

```text
1 step
```

---

## Gate 5：Attention-only + RS

```text
Qwen3
+
Attention-only
+
GSPO+RS / RSPO
```

只跑：

```text
1 step
```

---

## Gate 6：正式实验

最终：

```text
Full RS seed0
vs.
Attention-only RS seed0
```

之后再增加：

```text
seed1
seed2
```

---

# 20. Full 30B OOM 风险仍然存在

切换 Qwen3 可以显著减少软件兼容问题。

但是：

```text
Qwen3-30B-A3B
```

仍然是约 30B total parameter model。

即使：

```text
active params ≈ 3B
```

Full AdamW 仍然需要处理大部分 trainable parameters 的：

```text
gradients
optimizer states
FSDP shards
```

所以：

```text
Full
```

仍然可能 OOM。

Attention-only 的 memory advantage 会非常明显。

---

# 21. Full 模式建议的显存策略

第一版 smoke test 可以降低：

```text
TRAIN_BATCH_SIZE
PPO_MINI_BATCH_SIZE
ROLLOUT_N
MAX_RESPONSE_LENGTH
PPO_MAX_TOKEN_LEN_PER_GPU
```

必要时开启：

```text
optimizer offload
parameter offload
FSDP2 offload policy
```

先以：

```text
能跑 1 step
```

为目标。

不要一开始直接使用正式 MATH 8k context 配置。

---

# 22. 推荐的 Qwen3 Smoke 配置

第一阶段：

```text
TRAIN_BATCH_SIZE=2~4
PPO_MINI_BATCH_SIZE=1~2
ROLLOUT_N=4
MAX_PROMPT_LENGTH=1024
MAX_RESPONSE_LENGTH=1024
TOTAL_TRAINING_STEPS=1
```

先验证工程链。

正式实验再恢复：

```text
MATH
long response
larger rollout group
```

---

# 23. TensorBoard 必须保留的 Metrics

基础：

```text
reward
response_length

actor/pg_loss
actor/pg_clipfrac
actor/ppo_kl
actor/grad_norm
```

Rollout mismatch：

```text
rollout_corr/kl
rollout_corr/ppl_ratio
```

Router Shift：

```text
router_shift/raw_gamma_mean
router_shift/weight_mean
router_shift/weight_min
router_shift/weight_p10
router_shift/weight_p50
router_shift/floor_fraction
router_shift/deviation_mean
router_shift/deviation_p90
router_shift/deviation_p99
```

如果实现 importance ratio diagnostics：

```text
importance_ratio/p50
importance_ratio/p90
importance_ratio/p99
importance_ratio/max
```

---

# 24. 最终推荐的仓库结构

```text
verl/
├── verl/
│   ├── utils/
│   │   ├── parameter_update_policy.py
│   │   └── router_shift/
│   │       ├── core.py
│   │       └── capture.py
│   │
│   └── trainer/
│       └── ppo/
│           └── core_algos.py
│
└── experiments/
    └── moe_localized_rl/
        ├── scripts/
        │   ├── run_pilot_qwen3_rs.sh
        │   └── slurm_qwen3_rs.sh
        │
        └── configs/
            ├── qwen3_full_rs.yaml
            └── qwen3_attention_only_rs.yaml
```

---

# 25. 推荐实验命名

如果继续当前 plugin 实现：

```text
qwen3_30b_a3b_full_gspo_rs_seed0
qwen3_30b_a3b_attention_only_gspo_rs_seed0
```

如果严格实现 canonical RSPO：

```text
qwen3_30b_a3b_full_rspo_seed0
qwen3_30b_a3b_attention_only_rspo_seed0
```

---

# 26. Definition of Done

## Qwen3 Parameter Policy

- [ ] 能识别 `Qwen3MoeForCausalLM`
- [ ] `full` 全参数 trainable
- [ ] `attention_only` 只有 Attention trainable
- [ ] Experts frozen
- [ ] Router frozen
- [ ] Norm frozen
- [ ] Embedding frozen
- [ ] LM Head frozen
- [ ] strict validation 通过
- [ ] checkpoint delta 通过

---

## Qwen3 Router Capture

- [ ] 能识别 Qwen3 MoE router
- [ ] Old expert indices 正确
- [ ] Old scores 正确
- [ ] Current scores at old expert indices 正确
- [ ] Layer 数量一致
- [ ] Top-K shape 正确
- [ ] 无 padding / token alignment 错误

---

## RS / RSPO

- [ ] 明确算法是 `GSPO+RS` 还是 canonical `RSPO`
- [ ] 不混用实验命名
- [ ] Router Shift 使用 Old Policy，而不是 Reference Policy
- [ ] Stop-gradient 正确
- [ ] gamma_min 正确
- [ ] floor_fraction 使用 raw gamma
- [ ] RS disabled 时 vanilla GSPO 数值不变

---

## Launcher

- [ ] 使用 Qwen3 模型路径
- [ ] 不继承 Gemma4 SDPA workaround
- [ ] 不继承 Gemma4 remove-padding workaround
- [ ] `continuous_token.model_family=qwen3`
- [ ] Full / Attention-only 唯一核心差异是 `parameter_update.mode`

---

## Smoke Test

- [ ] Standalone Qwen3 vLLM 能生成
- [ ] Qwen3 Vanilla GSPO 1-step 成功
- [ ] Full + RS 1-step 成功
- [ ] Attention-only + RS 1-step 成功
- [ ] weight sync 正常
- [ ] 无 NaN / Inf
- [ ] 无 RouterTrace shape mismatch
- [ ] 无 FSDP optimizer param mismatch

---

# 27. 最终建议

当前不要直接：

```text
Gemma4 launcher
→ 改 MODEL_PATH
→ Qwen3
→ 正式训练
```

正确顺序应该是：

```text
1. Qwen3 Parameter Policy
        ↓
2. Qwen3 Router Capture
        ↓
3. 明确 GSPO+RS vs canonical RSPO
        ↓
4. 新建 Qwen3 Launcher
        ↓
5. Vanilla GSPO 1-step
        ↓
6. Full + RS 1-step
        ↓
7. Attention-only + RS 1-step
        ↓
8. 正式 Full vs Attention-only 实验
```

最重要的两个原则：

> **第一，不要把 Gemma4-specific workaround 带进 Qwen3。**

> **第二，不要让代码实际运行 GSPO+RS，但实验和论文中写成 canonical RSPO。**

完成以上修改后，Qwen3-30B-A3B 会比 Gemma4-26B-A4B 更适合作为当前 Full vs Attention-only 大模型验证平台。
