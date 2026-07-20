# Attention-only GSPO+RS vs Full GSPO+RS
## 基于 veRL 的 RSPO 实验 Vibe Coding 指导书

## 1. 实验目标

当前项目已经在 veRL 基础上实现了以下 Parameter Update Policy：

- `full`：更新全部参数
- `moe`：仅更新 Experts + Router
- `experts_only`：仅更新 Experts
- `attention_only`：仅更新 Attention

此前在 OLMoE + GSPO 实验中观察到：

- `Full GSPO`：可能发生 collapse
- `MoE-only GSPO`：可能发生 collapse
- `Attention-only GSPO`：训练相对稳定

本轮实验希望引入 **RSPO 中的 Router-Shift stabilization**，在相同稳定化机制下比较：

```text
Full GSPO + Router Shift
vs.
Attention-only GSPO + Router Shift
```

本轮核心研究问题：

> **当 MoE-specific routing instability 被 Router-Shift mechanism 缓解后，是否仍然只需要更新 Attention 参数，就可以获得接近 Full-parameter RL 的性能？**

因此实验中的唯一主要变量应该是：

```text
parameter_update.mode = full
```

vs.

```text
parameter_update.mode = attention_only
```

其余所有训练配置保持完全一致。

---

## 2. 实验组定义

### 2.1 Full GSPO+RS

```text
Attention       Trainable
Experts         Trainable
Router          Trainable
LayerNorm       Trainable
Embedding       Trainable
LM Head         Trainable
```

训练目标：

```text
GSPO
+
Router-Shift Rescaling
```

### 2.2 Attention-only GSPO+RS

```text
Attention       Trainable
Experts         Frozen
Router          Frozen
LayerNorm       Frozen
Embedding       Frozen
LM Head         Frozen
```

训练目标完全相同：

```text
GSPO
+
Router-Shift Rescaling
```

注意：

> Router 参数被冻结，不代表 routing decisions 不会变化。

因为 Attention 参数更新后：

```text
Attention output changes
        ↓
Hidden state changes
        ↓
Router input changes
        ↓
Router scores change
        ↓
Top-K experts may change
```

因此 `attention_only` 模式下仍然必须计算 Router Shift。

---

## 3. 总体工程原则

本项目继续基于现有 veRL 修改。

**不要创建新的 RL Trainer。**

正确架构：

```text
Existing veRL
        │
        ▼
Existing GSPO
        │
        ▼
Compute Importance Ratio
        │
        ▼
[NEW]
Router-Shift Rescaling
        │
        ▼
Existing GSPO Clipping
        │
        ▼
Existing Loss Aggregation
        │
        ▼
Backward
```

禁止：

```text
复制 PPO Trainer
复制 FSDP Worker
复制 GSPO 实现
重新实现 rollout engine
单独创建一套 RSPO Trainer
```

Router Shift 应该实现为一个 **可插拔的 importance-ratio adjustment module**。

---

## 4. 第一阶段：Repository Inspection

Cursor 首先不要修改代码。

先检查当前 veRL checkout，定位：

```text
1. actor model initialization
2. old_log_prob computation
3. current_log_prob computation
4. GSPO importance ratio computation
5. GSPO clipping
6. GSPO policy loss aggregation
7. rollout DataProto / batch fields
8. rollout actor 与 training actor 的生命周期
9. parameter_update_policy 应用位置
10. model forward output 中 Router 信息的位置
```

输出文件：

```text
RSPO_IMPLEMENTATION_MAP.md
```

格式：

```markdown
# RSPO Implementation Map

## Old Log Probability
File:
Function:
Input:
Output:

## Current Log Probability
File:
Function:

## GSPO Policy Loss
File:
Function:

## Importance Ratio
File:
Function:

## GSPO Clipping
File:
Function:

## Rollout Data Transport
File:
Class:

## Parameter Update Policy
File:
Function:

## Router Outputs
Model:
Module:
Output fields:
```

完成 inspection 后再开始修改代码。

---

## 5. Old Policy 与 Reference Policy 必须区分

RSPO 所比较的是：

```text
Old Policy Routing
vs.
Current Policy Routing
```

不是：

```text
Reference Policy Routing
vs.
Current Policy Routing
```

必须确保：

```text
old_log_prob
```

和：

```text
old_router_trace
```

来自同一个 old-policy snapshot。

正确的数据流：

```text
Old / Rollout Policy
        │
        ├── old_log_prob
        │
        └── old_router_trace
                │
                ▼
           Training Batch
                │
                ▼
Current Actor Forward
        │
        ├── current_log_prob
        │
        └── current_router_outputs
                │
                ▼
          Router Shift
                │
                ▼
       Adjust Importance Ratio
                │
                ▼
              GSPO
```

禁止使用 Reference Model 的 routing 信息代替 Old Policy Routing。

---

## 6. Router Trace 设计

不要保存完整：

```text
batch
× sequence
× layers
× num_experts
```

Router logits。

这会带来不必要的显存和通信开销。

优先保存：

```text
old_expert_indices
old_router_scores
```

概念维度：

```text
batch
× sequence
× moe_layers
× top_k
```

即只保存 old policy 当时激活的 experts 以及对应 routing scores。

建议结构：

```python
@dataclass
class RouterTrace:
    expert_indices: torch.Tensor
    router_scores: torch.Tensor
    attention_mask: torch.Tensor | None = None
```

Current policy forward 时：

```text
Current Router Logits
        │
        ▼
Gather scores at old_expert_indices
        │
        ▼
Compare with old_router_scores
        │
        ▼
Router Shift
```

---

## 7. Router Trace Adapter

不要把代码写死为 OLMoE。

建议新增：

```text
verl/
└── utils/
    └── router_trace/
        ├── __init__.py
        ├── base.py
        ├── olmoe.py
        └── gpt_oss.py
```

统一接口：

```python
class RouterTraceAdapter:
    def enable_capture(self, model):
        ...

    def get_trace(self):
        ...

    def clear(self):
        ...
```

或者：

```python
def extract_router_trace(
    model_output,
    model_type: str,
) -> RouterTrace:
    ...
```

当前如果先在 OLMoE 上实现：

```text
OLMoE adapter
```

未来切换：

```text
GPT-OSS-20B
```

只需要新增：

```text
GPT-OSS adapter
```

不要修改 RSPO 核心算法。

---

## 8. 新增配置

建议在现有 Actor Config 中加入：

```yaml
actor_rollout_ref:
  actor:

    parameter_update:
      mode: full

    policy_loss:
      loss_mode: gspo

      router_shift:
        enabled: true
        gamma_min: 0.8
        stop_gradient: true
        log_diagnostics: true
```

Attention-only：

```yaml
actor_rollout_ref:
  actor:

    parameter_update:
      mode: attention_only

    policy_loss:
      loss_mode: gspo

      router_shift:
        enabled: true
        gamma_min: 0.8
        stop_gradient: true
        log_diagnostics: true
```

默认：

```yaml
router_shift:
  enabled: false
```

必须保证：

```text
enabled = false
```

时：

```text
Modified veRL == Original GSPO
```

---

## 9. 不要创建 `loss_mode=rspo`

推荐：

```yaml
loss_mode: gspo

router_shift:
  enabled: true
```

而不是：

```yaml
loss_mode: rspo
```

因为 Router Shift 应该是一个 plug-in：

```text
GRPO + RS
GSPO + RS
GMPO + RS
```

都可以复用。

代码逻辑：

```python
importance_ratio = compute_importance_ratio(...)

if config.router_shift.enabled:
    router_shift_weight = compute_router_shift_weight(...)

    importance_ratio = apply_router_shift_rescaling(
        importance_ratio=importance_ratio,
        router_shift_weight=router_shift_weight,
    )

loss = compute_gspo_loss(
    importance_ratio=importance_ratio,
    ...
)
```

禁止：

```python
def rspo_loss(...):
    # copy all GSPO implementation here
```

---

## 10. RSPO 数学实现

Cursor 必须严格根据原始 RSPO 论文公式实现。

不要根据自然语言描述猜公式。

建议拆成纯函数：

```python
compute_layer_router_deviation(...)
```

↓

```python
compute_router_shift_ratio(...)
```

↓

```python
process_router_shift_weight(...)
```

↓

```python
apply_router_shift_rescaling(...)
```

完整流程：

```text
Old activated experts
        │
        ▼
Old router scores
        │
        │
Current router scores
on old activated experts
        │
        ▼
Layer-wise Router Deviation
        │
        ▼
Aggregate over MoE layers
        │
        ▼
Router Shift Ratio
        │
        ▼
Apply lower-bound γ_min
        │
        ▼
Stop Gradient / detach
        │
        ▼
Router Shift Weight
        │
        ▼
Rescale Importance Ratio
        │
        ▼
GSPO Clipping
```

默认：

```text
gamma_min = 0.8
```

Router Shift Weight 必须：

```python
router_shift_weight = router_shift_weight.detach()
```

禁止 Router Shift branch 产生额外 gradient path。

---

## 11. RS 的插入位置

必须确认执行顺序：

```text
Original Importance Ratio
        │
        ▼
Router-Shift Rescaling
        │
        ▼
GSPO Clipping
        │
        ▼
GSPO Surrogate Objective
        │
        ▼
Loss Aggregation
```

不要放在：

```text
GSPO loss 计算完成之后
```

也不要简单：

```text
final_loss *= router_shift_weight
```

必须严格按照 RSPO 论文定义修改 importance-ratio 相关计算。

---

## 12. Attention-only Parameter Policy

已有 `attention_only` 时检查其定义。

必须是：

```text
Attention       Trainable
Experts         Frozen
Router          Frozen
Norm            Frozen
Embedding       Frozen
LM Head         Frozen
```

不要把：

```text
pre-attention norm
post-attention norm
```

划入 Attention-only。

保持最严格的定义：

```text
Self-Attention Module Parameters Only
```

---

## 13. Attention-only 正确性测试

必须存在：

```python
test_attention_only_requires_grad()
```

验证：

```text
attention_trainable > 0

expert_trainable == 0
router_trainable == 0
norm_trainable == 0
embedding_trainable == 0
lm_head_trainable == 0
```

Backward test：

```text
Attention grad != None

Expert grad == None
Router grad == None
Norm grad == None
Embedding grad == None
LM Head grad == None
```

Checkpoint delta：

```text
Attention delta > 0

Expert delta == 0
Router delta == 0
Norm delta == 0
Embedding delta == 0
LM Head delta == 0
```

---

## 14. Router Shift Unit Tests

### Test 1：Identical Routing

输入：

```text
Old Routing == Current Routing
```

验证：

```text
Router Shift 表示最大 trust
```

并与论文公式预期一致。

### Test 2：Large Routing Deviation

构造：

```text
Current routing scores
```

明显偏离：

```text
Old routing scores
```

验证：

```text
Router Shift Weight decreases
```

### Test 3：Gamma Floor

验证：

```python
router_shift_weight >= gamma_min
```

默认：

```text
gamma_min = 0.8
```

### Test 4：Stop Gradient

验证：

```python
router_shift_weight.requires_grad is False
```

或者确认：

```text
RS branch 不产生 gradient
```

### Test 5：Disabled Equivalence

配置：

```yaml
router_shift:
  enabled: false
```

同一个输入下：

```text
Original GSPO Loss
```

必须与：

```text
Modified GSPO with RS disabled
```

数值一致。

使用：

```python
torch.testing.assert_close(...)
```

---

## 15. Old Router Trace 数据传输测试

新增 batch fields，例如：

```text
old_router_indices
old_router_scores
```

必要时：

```text
old_router_mask
```

必须严格与 response tokens 对齐。

特别注意：

```text
Prompt
Response
Padding
Packed Sequence
```

最终 Router Shift 只能作用于：

```text
有效 response tokens
```

必须复用现有 GSPO：

```text
loss_mask
```

建议增加：

```python
assert router_shift_weight.shape == loss_mask.shape
```

或在明确完成 layer/top-k aggregation 后检查。

禁止利用 PyTorch 自动 broadcast 掩盖 shape bug。

---

## 16. Integration Test

使用 tiny MoE 完整测试：

```text
Old Policy Forward
        │
        ▼
old_log_prob
old_router_trace
        │
        ▼
Current Policy Forward
        │
        ▼
current_log_prob
current_router_trace
        │
        ▼
Router Shift Weight
        │
        ▼
Adjusted Importance Ratio
        │
        ▼
GSPO Loss
        │
        ▼
Backward
        │
        ▼
Optimizer Step
```

分别测试：

```text
mode = full
```

和：

```text
mode = attention_only
```

两种模式必须均能正常完成。

---

## 17. Distributed Smoke Test

完成 Unit Test 后运行真实 veRL pipeline。

第一阶段只跑：

```text
1–2 training steps
```

验证：

```text
rollout
→ old router trace
→ reward
→ current actor forward
→ router shift
→ GSPO update
→ weight sync
→ new rollout
```

分别测试：

```text
Full GSPO+RS
Attention-only GSPO+RS
```

确认：

```text
无 NaN
无 Inf
无 shape mismatch
无 DataProto key missing
无 loss_mask 错误
```

---

## 18. TensorBoard Metrics

必须增加 Router Shift diagnostics。

至少记录：

```text
router_shift/weight_mean
router_shift/weight_min
router_shift/weight_p10
router_shift/weight_p50

router_shift/floor_fraction

router_shift/deviation_mean
router_shift/deviation_p90
router_shift/deviation_p99
```

如果需要观察 layer：

```text
router_shift/first_layer_mean
router_shift/middle_layer_mean
router_shift/last_layer_mean
```

不要每 step 为全部 MoE layers 写大量 TensorBoard metrics。

同时继续记录：

```text
reward

response_length

actor/pg_loss
actor/pg_clipfrac
actor/ppo_kl
actor/grad_norm

rollout_corr/kl
rollout_corr/ppl
```

如现有代码已经加入 importance-ratio diagnostics，继续保存：

```text
importance_ratio/mean
importance_ratio/p90
importance_ratio/p99
importance_ratio/max
```

---

## 19. 正式实验配置

创建：

```text
full_gspo_rs.yaml
attention_only_gspo_rs.yaml
```

两份配置必须只有：

```yaml
parameter_update:
  mode:
```

不同。

例如：

```diff
- mode: full
+ mode: attention_only
```

以下必须完全相同：

```text
Model
Dataset
Seed

Learning Rate
Optimizer

Train Batch Size
Mini Batch Size

Rollout N

Max Prompt Length
Max Response Length

GSPO Clip Parameters

Router Shift gamma_min

Training Epochs
Training Steps

Reward Function

Evaluation Protocol
```

---

## 20. 实验命名

推荐：

```text
full_gspo_rs_seed0
attention_only_gspo_rs_seed0
```

不要简单命名：

```text
full_rspo
attention_rspo
```

因为当前实际算法结构是：

```text
GSPO + Router Shift
```

后续如果实验：

```text
GRPO + RS
GMPO + RS
```

不会混淆。

---

## 21. 本轮实验要回答的问题

核心问题只有一个：

> **在相同 Router-Shift stabilization 下，Attention-only RL 是否可以达到接近 Full-parameter RL 的性能？**

比较：

```text
Performance(Attention-only + RS)
```

和：

```text
Performance(Full + RS)
```

同时比较：

```text
Peak GPU Memory
Optimizer Memory
Training Throughput
```

---

## 22. 结果解释

### Case A：Attention-only RS ≈ Full RS

这是最理想的结果。

如果同时：

```text
Memory_Attention << Memory_Full
```

那么支持：

> RL adaptation in pretrained MoEs can be largely localized to the attention subspace, even when MoE-specific routing instability is explicitly controlled.

这时 Attention-only 有：

```text
Performance
+
Stability
+
Memory Efficiency
```

三个潜在优势。

### Case B：Full RS >> Attention-only RS

说明：

```text
MoE parameters 确实提供额外 RL plasticity
```

之前 Full GSPO collapse 可能只是因为：

```text
GSPO 无法稳定利用 MoE parameter updates
```

而不是：

```text
MoE parameters 不需要更新
```

### Case C：两者都稳定，但 Attention-only 提升较小

说明 Attention-only 可能只是因为：

```text
可训练参数少
→ policy movement 小
→ 不容易 collapse
```

而不是：

```text
Attention 是最优 RL adaptation subspace
```

此时应继续研究：

```text
Attention + Selected Experts
```

### Case D：Full RS / Attention-only RS 仍然 collapse

说明当前 instability 可能不主要由 Router Shift 引起。

下一步应该重新检查：

```text
Importance Ratio
Policy Drift
Rollout-Training Mismatch
GSPO Update Dynamics
```

而不是继续增加 Router stabilization。

---

## 23. 本轮实验背后的逻辑

已有实验：

```text
Vanilla GSPO
        │
        ├── Full → Collapse
        │
        └── Attention-only → Stable
```

现在的问题是：

> Full 的失败是不是单纯因为更新 MoE 后产生了 routing instability？

因此：

```text
                Vanilla GSPO
                     │
       ┌─────────────┴─────────────┐
       │                           │
      Full                  Attention-only
       │                           │
   Collapse                       Stable
       │
       └──────────────┬────────────┘
                      │
                      ▼
             Add Router Shift
                      │
          ┌───────────┴───────────┐
          │                       │
      Full + RS          Attention-only + RS
```

如果：

```text
Full + RS >> Attention-only + RS
```

说明：

> MoE plasticity 是有价值的，只是 vanilla GSPO 无法稳定优化它。

如果：

```text
Full + RS ≈ Attention-only + RS
```

说明：

> 即使 MoE parameters 可以被稳定训练，它们对当前 RL adaptation 仍然不是必要的。

这就是本轮实验真正需要区分的两个 hypothesis。

---

## 24. Cursor 执行顺序

### Task 1

```text
Repository Inspection
```

输出：

```text
RSPO_IMPLEMENTATION_MAP.md
```

### Task 2

检查：

```text
attention_only
```

parameter policy correctness。

### Task 3

实现：

```text
Router Trace Extraction
```

### Task 4

实现：

```text
Old Router Trace
```

在 rollout → training batch 的数据传输。

### Task 5

根据论文实现纯函数：

```text
Router Deviation
Router Shift Ratio
Router Shift Weight
```

### Task 6

将 Router Shift 插入：

```text
Existing GSPO Importance Ratio
```

与 clipping 之间。

### Task 7

完成所有 Unit Tests。

### Task 8

完成 Tiny MoE Integration Test。

### Task 9

运行：

```text
Full GSPO+RS
```

和：

```text
Attention-only GSPO+RS
```

各 1–2 step distributed smoke test。

### Task 10

确认所有 diagnostics 正确后运行正式实验。

---

## 25. Definition of Done

### Algorithm Correctness

- [ ] RS 使用 Old Policy routing，而不是 Reference Policy routing
- [ ] Old Router Trace 与 old_log_prob 来自同一个 policy snapshot
- [ ] RS 数学公式严格按照论文实现
- [ ] Router Shift Weight 使用 stop-gradient
- [ ] `gamma_min=0.8`
- [ ] RS 在 GSPO clipping 前作用于 importance ratio
- [ ] Attention-only 即使 Router frozen 仍计算 Router Shift
- [ ] `router_shift.enabled=false` 时与 Vanilla GSPO 数值一致

### Parameter Correctness

- [ ] Full 模式所有参数可训练
- [ ] Attention-only 仅 Attention 参数可训练
- [ ] Experts frozen
- [ ] Router frozen
- [ ] Norm frozen
- [ ] Embedding frozen
- [ ] LM Head frozen
- [ ] Checkpoint delta 验证通过

### Engineering

- [ ] Router Trace 不保存不必要的完整 expert logits
- [ ] Router Trace 与 response loss mask 正确对齐
- [ ] Full GSPO+RS 可以完成 distributed smoke test
- [ ] Attention-only GSPO+RS 可以完成 distributed smoke test
- [ ] Actor → rollout weight sync 正常
- [ ] TensorBoard Router Shift metrics 正常

### Experiment

- [ ] Full 与 Attention-only 使用相同模型
- [ ] 相同数据
- [ ] 相同 seed
- [ ] 相同 GSPO 参数
- [ ] 相同 RS 参数
- [ ] 相同 training token budget
- [ ] 相同 evaluation protocol
- [ ] 两份 config 只有 `parameter_update.mode` 不同

最终回答：

> **After controlling MoE routing instability with Router-Shift stabilization, is full-model RL adaptation still necessary, or can attention-only adaptation retain comparable performance at substantially lower training memory cost?**
