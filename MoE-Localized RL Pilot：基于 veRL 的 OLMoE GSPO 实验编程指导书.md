# MoE-Localized RL Pilot：基于 veRL 的 OLMoE GSPO 实验编程指导书

## 1. 项目目标

本项目研究以下核心问题：

> **在 pretrained native MoE language model 上进行 RL 时，是否必须更新 Attention/shared parameters，还是只更新原生 MoE experts 与 router 就可以获得接近 full-parameter RL 的学习效果？**

第一阶段不追求 SOTA，不实现新的 RL 算法，也不加入 RSPO。

第一阶段只做一个低成本的 **go / no-go feasibility experiment**：

```text
OLMoE-1B-7B-0125-Instruct
        +
       GSPO
        |
        +-------------------+
        |                   |
   Full-GSPO          MoE-only-GSPO
        |                   |
 Update all params    Update experts + router
```

核心比较：

```text
Full GSPO
vs.
MoE-only GSPO
```

如果 MoE-only 能保留 Full GSPO 的大部分 RL improvement，再进入 Qwen3-30B-A3B 等现代大型 MoE。

------

# 2. 为什么基于 veRL 修改

不要自行实现 RL training loop。

使用当前 veRL 作为基础设施，保留其现有：

- rollout generation；
- reward computation；
- group sampling；
- GRPO advantage estimator；
- GSPO policy loss；
- reference policy；
- actor rollout weight synchronization；
- FSDP/FSDP2；
- checkpoint；
- logging；
- evaluation。

当前 veRL 已经实现 GSPO policy loss，并支持通过配置切换；其 worker 架构也支持 FSDP/FSDP2/Megatron 训练 backend，以及 vLLM/SGLang 等 rollout backend。

**本项目的原则是：**

> 尽量不修改 veRL 的 RL algorithm 和 trainer control flow，只增加一个可配置的 `Parameter Update Policy`。

理想情况下：

```text
vanilla veRL
    |
    +-- model initialization
    |
    +-- [NEW] apply_parameter_update_policy()
    |
    +-- FSDP/FSDP2 wrapping
    |
    +-- optimizer creation
    |
    +-- existing GSPO training
```

所有实验尽量通过 config 切换，不维护两套 trainer。

------

# 3. 模型

第一阶段使用：

```text
allenai/OLMoE-1B-7B-0125-Instruct
```

原因：

- 7B total parameters；
- 约 1B active parameters；
- native sparse MoE；
- Hugging Face Transformers 原生支持；
- 成本远低于 Qwen3-30B-A3B；
- 足够用于验证 parameter-localized RL 是否存在基本 signal。

OLMoE 当前 Transformers 架构中：

```text
OlmoeDecoderLayer
├── input_layernorm
├── self_attn
├── post_attention_layernorm
└── mlp: OlmoeSparseMoeBlock
      ├── gate: OlmoeTopKRouter
      └── experts: OlmoeExperts
```

即每层基本执行：

```text
Hidden State
    │
    ▼
LayerNorm
    │
    ▼
Self Attention
    │
    + Residual
    │
    ▼
LayerNorm
    │
    ▼
MoE Router
    │
    ▼
Top-K Experts
    │
    + Residual
    │
    ▼
Next Layer
```

该结构可以直接用于区分：

```text
Attention
Experts
Router
LayerNorm
Embedding
LM Head
```

当前 Transformers 实现中，`OlmoeDecoderLayer.self_attn` 对应 Attention，`OlmoeDecoderLayer.mlp` 是 `OlmoeSparseMoeBlock`，其内部进一步包含 `gate` 和 `experts`。

------

# 4. 第一阶段实验定义

第一阶段必须实现三个模式，但实际正式 pilot 先运行前两个。

## Mode A：Full

```text
mode = full
```

所有参数：

```python
requires_grad = True
```

包括：

```text
Attention
Experts
Router
LayerNorm
Embedding
LM Head
```

这是标准 GSPO baseline。

------

## Mode B：MoE-only

```text
mode = moe
```

仅允许：

```text
OlmoeExperts
OlmoeTopKRouter
```

参数更新。

即：

```text
Attention       Frozen
Experts         Trainable
Router          Trainable
LayerNorm       Frozen
Embedding       Frozen
LM Head         Frozen
```

这是本项目核心实验。

------

## Mode C：Experts-only

```text
mode = experts_only
```

只允许：

```text
OlmoeExperts
```

更新。

即：

```text
Attention       Frozen
Experts         Trainable
Router          Frozen
LayerNorm       Frozen
Embedding       Frozen
LM Head         Frozen
```

第一轮不必立即跑这一组。

只有当：

```text
MoE-only ≈ Full
```

时，再运行 Experts-only，用于回答：

> Router adaptation 是否必要？

------

# 5. 第一阶段明确不做的事情

以下内容不要在 MVP 中实现：

```text
RSPO
Routing Replay
Selective Experts
Top-K trainable experts
Advantage-weighted expert selection
Attention LoRA
Router regularization
Partial weight synchronization optimization
New GSPO implementation
New rollout engine
New reward model
```

尤其：

> **不要修改 GSPO 数学实现。**

直接使用 veRL 已有 GSPO。

第一阶段必须把唯一核心变量控制为：

```text
Trainable Parameter Subspace
```

即：

```text
Full
vs.
MoE-only
```

------

# 6. 推荐代码结构

尽量减少对 veRL core 的侵入。

建议增加：

```text
verl/
├── verl/
│   └── utils/
│       └── parameter_update_policy.py        # NEW
│
├── tests/
│   └── models/
│       └── test_parameter_update_policy.py   # NEW
│
├── experiments/
│   └── moe_localized_rl/
│       ├── README.md
│       ├── configs/
│       │   ├── olmoe_full_gspo.yaml
│       │   ├── olmoe_moe_gspo.yaml
│       │   └── olmoe_experts_only_gspo.yaml
│       │
│       ├── scripts/
│       │   ├── run_smoke_full.sh
│       │   ├── run_smoke_moe.sh
│       │   ├── run_pilot_full.sh
│       │   └── run_pilot_moe.sh
│       │
│       └── tools/
│           ├── inspect_model_modules.py
│           ├── verify_trainable_params.py
│           └── compare_checkpoints.py
```

如果 veRL 当前 repository 已有更合适的实验目录规范，遵守现有规范，不强制创建上述路径。

核心原则：

```text
核心逻辑进入 verl/
实验配置进入 examples/ 或 experiments/
研究专用分析工具不要污染 trainer
```

------

# 7. 新增配置

给 Actor 增加一个参数：

```yaml
actor_rollout_ref:
  actor:
    parameter_update:
      mode: full
      strict: true
      log_parameter_stats: true
```

支持：

```text
full
moe
experts_only
```

默认必须是：

```yaml
mode: full
```

从而保证：

> 不指定该配置时，veRL 行为与原始版本完全一致。

这是 backward compatibility 的硬性要求。

------

# 8. Parameter Update Policy 实现

新增：

```python
apply_parameter_update_policy(
    model,
    mode: str,
    strict: bool = True,
)
```

基本逻辑：

```python
if mode == "full":
    enable_all_parameters(model)

elif mode == "moe":
    freeze_all_parameters(model)
    enable_olmoe_experts(model)
    enable_olmoe_router(model)

elif mode == "experts_only":
    freeze_all_parameters(model)
    enable_olmoe_experts(model)

else:
    raise ValueError(...)
```

不要一开始通过简单字符串：

```python
if "mlp" in name
```

判断。

优先使用 module class。

对于 OLMoE：

```python
OlmoeExperts
OlmoeTopKRouter
```

因为 `mlp` 这样的字符串未来可能同时匹配普通 dense FFN。

建议实现：

```python
for module in model.modules():

    if isinstance(module, OlmoeExperts):
        for p in module.parameters():
            p.requires_grad = True

    if mode == "moe" and isinstance(module, OlmoeTopKRouter):
        for p in module.parameters():
            p.requires_grad = True
```

同时保留一个 model-type dispatch：

```python
MODEL_PARAMETER_POLICIES = {
    "olmoe": apply_olmoe_policy,
}
```

方便未来扩展：

```text
qwen3_moe
deepseek
gpt_oss
```

不要把整个 utility 写死成 OLMoE-only。

------

# 9. Parameter Policy 必须在什么时候执行

Cursor 必须找到当前 veRL 中：

```text
Actor model constructed
        ↓
Distributed wrapping
        ↓
Optimizer constructed
```

的准确路径。

Parameter Policy 必须发生在：

```text
Model load
        ↓
apply_parameter_update_policy()
        ↓
FSDP/FSDP2 wrap
        ↓
Optimizer construction
```

即：

```text
Model initialization 后
Optimizer creation 前
```

不允许在 optimizer 创建之后才 freeze。

同时 optimizer 应只接受：

```python
p for p in model.parameters() if p.requires_grad
```

需要检查 veRL 当前 optimizer builder 是否已经这么处理。

如果没有，应修改 optimizer 参数输入，但不要改 optimizer 算法。

------

# 10. FSDP/FSDP2 注意事项

Pilot 优先使用：

```text
FSDP2
```

原因是 partial freezing 会产生：

```text
frozen parameters
+
trainable parameters
```

混合存在。

PyTorch FSDP 文档指出，FSDP1 在 `use_orig_params=True` 时支持 frozen 和 non-frozen 参数混用，但这种方式可能带来额外 gradient memory；FSDP2 的 per-parameter sharding 对 frozen parameters 的限制更少。

当前 veRL 配置也区分 FSDP/FSDP2，而 FSDP1 默认 `use_orig_params=false`。

因此优先：

```yaml
actor_rollout_ref:
  actor:
    strategy: fsdp2
```

如果当前 OLMoE + veRL checkout 对 FSDP2 存在兼容问题，再测试：

```text
FSDP1
+
use_orig_params=true
```

但不要默认使用 FSDP1 + `use_orig_params=false` 进行 mixed frozen/trainable parameters 实验。

------

# 11. Parameter Audit

每次训练启动必须打印：

```text
====================================================
Parameter Update Policy
====================================================
Mode: moe
Model type: olmoe

Total parameters:        X
Trainable parameters:    Y
Frozen parameters:       Z
Trainable ratio:         XX.XX %

By category:

Attention:       0 trainable / XXX total
Experts:         XXX trainable / XXX total
Router:          XXX trainable / XXX total
LayerNorm:       0 trainable / XXX total
Embedding:       0 trainable / XXX total
LM Head:         0 trainable / XXX total
====================================================
```

并写入：

```text
parameter_policy.json
```

示例：

```json
{
  "mode": "moe",
  "total_parameters": 7000000000,
  "trainable_parameters": 0,
  "categories": {
    "attention": {},
    "experts": {},
    "router": {},
    "norm": {},
    "embedding": {},
    "lm_head": {}
  }
}
```

实际数值由代码自动统计。

不要手写参数量。

------

# 12. Strict Validation

当：

```yaml
strict: true
```

时，MoE-only 必须自动 assert：

```python
assert trainable_expert_params > 0
assert trainable_router_params > 0

assert trainable_attention_params == 0
assert trainable_norm_params == 0
assert trainable_embedding_params == 0
assert trainable_lm_head_params == 0
```

Experts-only：

```python
assert trainable_expert_params > 0

assert trainable_router_params == 0
assert trainable_attention_params == 0
```

任何 assertion failure：

```text
直接终止训练
```

禁止 silent fallback。

------

# 13. 单元测试

## Test 1：Full policy

使用 tiny OLMoE config 创建一个非常小的随机模型，例如：

```text
hidden_size ≈ 64
layers = 2
experts = 4
top-k = 2
```

不要下载 7B checkpoint。

执行：

```python
apply_parameter_update_policy(model, "full")
```

验证：

```python
all(p.requires_grad for p in model.parameters())
```

------

## Test 2：MoE-only policy

执行：

```python
apply_parameter_update_policy(model, "moe")
```

验证：

```text
OlmoeExperts      requires_grad = True
OlmoeTopKRouter   requires_grad = True

OlmoeAttention    requires_grad = False
OlmoeRMSNorm      requires_grad = False
Embedding         requires_grad = False
LM Head           requires_grad = False
```

------

## Test 3：Experts-only policy

验证：

```text
Experts     True
Router      False
Attention   False
Norm        False
```

------

## Test 4：Backward gradient test

对 tiny model：

```text
forward
→ dummy LM loss
→ backward
```

MoE-only 模式下验证：

```python
expert.grad is not None
router.grad is not None

attention.grad is None
norm.grad is None
lm_head.grad is None
```

注意：

某些 expert 在当前 batch 中可能没有被 router 激活，因此不能要求：

```python
所有 expert grad != None
```

正确判断应该是：

```text
至少一个被激活 expert 获得 gradient
所有 frozen parameter 均无 gradient
```

------

# 14. 最关键的 Integration Test：参数是否真的只更新 MoE

这是整个项目最重要的 correctness test。

执行：

```text
Load tiny model
↓
Save checkpoint A
↓
Run one optimizer step
↓
Save checkpoint B
```

计算：

```python
delta = parameter_after - parameter_before
```

MoE-only 必须满足：

```text
Experts:

max_abs_delta > 0


Router:

max_abs_delta > 0


Attention:

max_abs_delta == 0


LayerNorm:

max_abs_delta == 0


Embedding:

max_abs_delta == 0


LM Head:

max_abs_delta == 0
```

建议实现：

```text
experiments/moe_localized_rl/tools/compare_checkpoints.py
```

输出：

```text
Parameter Category        Changed Params
-----------------------------------------
Attention                 0
Experts                   XXXXX
Router                    XXXXX
LayerNorm                 0
Embedding                 0
LM Head                   0
```

这是进入正式 RL 实验前的硬性 gate。

------

# 15. Rollout Weight Sync 测试

由于 veRL 的 Actor 和 rollout engine 之间需要同步权重，需要确认 partial training 不影响：

```text
Actor updated experts/router
        ↓
rollout engine receives updated weights
```

不要在 MVP 阶段实现“只同步变化参数”。

继续使用 veRL 原始 weight synchronization。

测试目标只是：

```text
MoE-only optimizer step 后
updated model 可以成功同步到 rollout backend
且下一轮 rollout 正常完成
```

至少完成：

```text
1 rollout
→ 1 update
→ weight sync
→ 1 new rollout
```

无报错。

不要为了节省通信修改现有 weight sync。

这是未来 optimization，不属于当前研究问题。

------

# 16. GSPO 配置

使用 veRL 当前已有 GSPO implementation。

保持：

```text
advantage estimator = GRPO-style group advantage
policy loss = GSPO
```

当前 veRL 的 GSPO 实现通过注册的 `gspo` policy loss 完成 sequence-level importance ratio 和 sequence-level loss aggregation。

核心配置形式：

```yaml
algorithm:
  adv_estimator: grpo

actor_rollout_ref:
  actor:
    policy_loss:
      loss_mode: gspo
```

具体 GSPO clipping hyperparameters：

> 优先继承当前 veRL 官方 GSPO recipe 或当前 checkout 的推荐配置。

不要由 Cursor 自行“根据 GRPO 猜一个参数”。

Full 和 MoE-only 必须使用完全相同：

```text
learning rate
batch size
group size
clip range
temperature
max response length
training prompts
random seed
training steps
```

唯一变量：

```text
parameter_update.mode
```

------

# 17. 数据与实验分两阶段

## Stage A：Engineering Smoke Test

第一步使用 veRL 已有支持最成熟的简单数学 RLVR 数据。

优先：

```text
GSM8K
```

原因不是为了做研究结果，而是验证：

```text
OLMoE
+
veRL
+
GSPO
+
reward
+
rollout
+
parameter freezing
```

整个 pipeline 能跑通。

veRL 官方 quickstart 本身提供 GSM8K function-based reward 的 RL 示例，因此适合作为工程 smoke test。

Smoke 数据量：

```text
32–128 prompts
```

只需要：

```text
2–5 optimizer steps
```

分别运行：

```text
Full
MoE-only
```

目标：

```text
无 OOM
无 NaN
reward 正常计算
GSPO loss 正常
参数更新检查通过
checkpoint 正常
```

------

## Stage B：Experiment 0

工程验证通过后，再运行真正 pilot。

训练集可以从：

```text
DAPO-Math-style verifiable math data
```

或适合当前 OLMoE 难度的数学 RLVR 数据中选择。

不要直接假设 OLMoE 能解决大量困难 DAPO prompts。

首先进行 difficulty scan：

```text
随机抽取候选训练问题
每题 rollout 8 次
```

估计：

```text
p_correct
```

优先保留：

```text
0 < p_correct < 1
```

即一个 group 中存在 reward variance 的题。

Pilot 数据规模建议：

```text
1K–3K prompts
```

目标不是训练 SOTA，而是获得清晰 learning curve。

------

# 18. 正式 Pilot Run

第一轮只跑：

```text
Run A: Full GSPO
Run B: MoE-only GSPO
```

不要立即跑 Experts-only。

保证：

```text
same initial checkpoint
same dataset
same seed
same rollout n
same training tokens
same optimizer hyperparameters
same GSPO hyperparameters
```

Run name：

```text
olmoe_gspo_full_seed0
olmoe_gspo_moe_seed0
```

首先跑：

```text
1 seed
```

只有出现值得继续的 signal，再增加：

```text
seed1
seed2
```

------

# 19. 必须记录的 Metrics

## RL Metrics

至少记录：

```text
reward/mean
reward/std

actor/pg_loss
actor/pg_clipfrac
actor/ppo_kl
actor/grad_norm

response_length
entropy（如果启用）
```

veRL 当前已经提供 actor loss、KL、clip fraction、grad norm 等训练指标。

------

## Parameter Metrics

新增：

```text
params/trainable_total
params/trainable_ratio

params/expert_trainable
params/router_trainable
params/attention_trainable
```

------

## Efficiency Metrics

记录：

```text
peak_gpu_memory
step_time
rollout_time
update_time
tokens_per_second
```

第一阶段只记录。

不要提前假设：

```text
MoE-only 一定更省大量显存
```

------

# 20. MoE Diagnostics

MVP 完成后增加简单 diagnostics。

第一版只需要：

```text
expert utilization
router load entropy
```

可以通过 lightweight forward hook 监听：

```text
OlmoeTopKRouter
```

其 forward 当前会输出：

```text
router_logits
router_scores
router_indices
```

因此可以统计每层 Top-K expert selection，而不必长期保存完整 router logits。

统计：

```text
tokens_per_expert
expert_selection_frequency
load_entropy
```

第一阶段不要实现完整 RSPO Router Shift Ratio。

RSPO-style：

```text
old policy routing
vs.
current policy routing
```

需要对同一 trajectory 保存和比较 route 信息，复杂度明显更高。

只有发现：

```text
MoE-only training unstable
```

或者：

```text
routing distribution 明显漂移
```

之后再实现。

------

# 21. Experiment 0 要回答的问题

Experiment 0 只回答：

## RQ1

> 当 Full GSPO 可以学习时，仅更新 pretrained native MoE experts + router，是否也能够产生显著 RL improvement？

比较：

```text
ΔReward_full
vs.
ΔReward_moe
```

定义：

```text
MoE RL Retention Ratio

          MoE final - Base
rho = -------------------------
          Full final - Base
```

这里不需要在代码中硬编码 threshold。

分析阶段观察：

```text
rho 接近 1
```

还是：

```text
rho 接近 0
```

------

# 22. 结果决策树

## Case A

```text
MoE-only ≈ Full
```

结论：

```text
Positive signal
```

下一步：

```text
Qwen3-30B-A3B
+
GSPO
+
Full vs MoE-only
```

同时补：

```text
Experts-only
```

研究 router necessity。

------

## Case B

```text
Base < MoE-only < Full
```

仍然继续。

下一步增加：

```text
MoE + small Attention LoRA
```

研究：

> 最少需要多少 non-MoE plasticity？

------

## Case C

```text
MoE-only ≈ Base
Full >> Base
```

说明纯 MoE-localized RL hypothesis 在 OLMoE 上不成立。

不要马上烧 Qwen3 算力。

先增加：

```text
Freeze-Attention-only
```

即：

```text
Attention Frozen
Experts Trainable
Router Trainable
Norm Trainable
LM Head Trainable
```

判断失败来自：

```text
Attention
```

还是其他 shared parameters。

------

## Case D

```text
MoE-only 学习但明显不稳定
```

检查：

```text
router utilization
KL
clip fraction
importance ratio
```

之后才考虑：

```text
RSPO
```

不要提前把 RSPO 绑定到核心方法。

------

# 23. Cursor 执行顺序

Cursor 必须严格按下面顺序工作。

## Task 1：Repository Inspection

先不要写代码。

检查当前 checkout：

```text
Actor model initialization 在哪里
FSDP/FSDP2 wrapping 在哪里
Optimizer creation 在哪里
GSPO config 在哪里
Actor → rollout weight sync 在哪里
```

输出：

```text
IMPLEMENTATION_MAP.md
```

说明计划修改的准确文件和函数。

完成后再进入 Task 2。

------

## Task 2：Baseline Reproduction

不修改任何训练逻辑。

运行：

```text
tiny GSM8K
+
OLMoE
+
Full GSPO
```

确认原始 veRL pipeline 可以运行。

如果 baseline 不能运行：

> 优先解决 OLMoE compatibility。

不要同时开发 parameter freezing。

------

## Task 3：Parameter Policy

实现：

```text
full
moe
experts_only
```

要求：

```text
默认 full 完全 backward compatible。
```

------

## Task 4：Unit Tests

完成：

```text
mode selection test
requires_grad test
backward gradient test
```

所有测试通过才能继续。

------

## Task 5：Checkpoint Delta Test

完成：

```text
one optimizer step
+
before/after parameter comparison
```

证明：

```text
只有目标参数发生变化。
```

------

## Task 6：Distributed Smoke Test

使用：

```text
FSDP2
```

完成：

```text
rollout
→ reward
→ GSPO update
→ weight sync
→ rollout
```

分别测试：

```text
Full
MoE-only
```

------

## Task 7：Experiment Config

创建：

```text
olmoe_full_gspo.yaml
olmoe_moe_gspo.yaml
```

确保通过 diff 检查：

```text
除 parameter_update.mode 外，
所有实验相关 hyperparameter 相同。
```

------

## Task 8：Pilot

运行：

```text
Full GSPO
MoE-only GSPO
```

生成：

```text
reward_curve.png
kl_curve.png
grad_norm_curve.png
gpu_memory.csv
training_summary.json
```

------

## Task 9：Result Validation

训练结束后自动执行 checkpoint diff。

MoE-only checkpoint 必须再次验证：

```text
Attention unchanged
Norm unchanged
Embedding unchanged
LM Head unchanged

Experts changed
Router changed
```

如果不满足：

```text
本次实验无效
```

不得用于研究结论。

------

# 24. Cursor 开发原则

Cursor 在整个任务中必须遵循：

### 原则 1

不要重写 veRL 已有功能。

### 原则 2

不要修改 GSPO 数学实现。

### 原则 3

所有新功能必须 config-driven。

### 原则 4

默认配置必须保持 vanilla veRL 行为。

### 原则 5

先写测试，再运行 expensive experiment。

### 原则 6

禁止通过“训练没有报错”判断 freeze 成功。

必须通过：

```text
requires_grad
+
gradient
+
checkpoint parameter delta
```

三层验证。

### 原则 7

不要在本阶段实现 RSPO。

### 原则 8

不要提前优化 partial weight synchronization。

### 原则 9

遇到 veRL 主干 API 与本文档不一致时：

```text
优先适配当前 checkout 的官方接口，
不要为了匹配本文档复制旧版 veRL 代码。
```

### 原则 10

所有实验必须保存：

```text
git commit hash
veRL commit hash
transformers version
torch version
CUDA version
model checkpoint
完整 Hydra config
random seed
```

保证可复现。

------

# 25. Definition of Done

整个 Experiment 0 只有满足以下条件才算完成。

## Engineering

-  OLMoE 可以在 veRL 中完成 GSPO training
-  Full mode 正常
-  MoE-only mode 正常
-  FSDP2 distributed training 正常
-  rollout weight sync 正常
-  checkpoint save/load 正常

## Correctness

-  Full 所有目标参数可训练
-  MoE-only Attention 无梯度
-  MoE-only Norm 无梯度
-  MoE-only Embedding/LM Head 无梯度
-  Experts 有梯度
-  Router 有梯度
-  checkpoint delta 验证 frozen 参数 bitwise/数值不变

## Experiment

-  Full GSPO 出现明确 learning signal
-  MoE-only 使用完全相同训练数据
-  MoE-only 使用完全相同 token budget
-  两组 learning curve 可直接比较
-  记录 peak memory 和 training speed
-  保存最终 checkpoint 与完整配置

## Research Decision

最终只输出一个核心结论：

```text
Does MoE-only GSPO retain the RL improvement of Full GSPO on OLMoE?
```

如果答案是：

```text
Yes / Mostly Yes
```

进入大型现代 MoE：

```text
Qwen3-30B-A3B
```

如果答案是：

```text
No
```

优先研究：

```text
Freeze-Attention
MoE + Attention LoRA
```

而不是直接扩大模型规模。

------

# 26. 最终预期代码改动规模

理想情况下，核心 veRL 改动应该非常小：

```text
1 个 parameter policy utility
1 个 config extension
1 个 actor model initialization hook
若干 unit tests
```

其余全部放在：

```text
experiment configs
scripts
analysis tools
```

如果 Cursor 最终需要：

```text
大规模重写 PPO trainer
复制一份 ray_trainer
复制一份 FSDP worker
自己实现 GSPO
```

说明设计方向错误，应停止并重新检查集成点。

本项目的核心工程目标是：

> **Make parameter plasticity a configurable property of an existing veRL training run, rather than creating a new RL framework.**