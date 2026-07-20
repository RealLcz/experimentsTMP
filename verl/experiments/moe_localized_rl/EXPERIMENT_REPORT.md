# MoE-Localized RL 实验报告

## 1. 研究问题

> 在 pretrained native MoE language model 上进行 RL 时，是否必须更新 Attention/shared parameters，还是只更新原生 MoE experts 与 router 就可以获得接近 full-parameter RL 的学习效果？

本实验为 go/no-go feasibility experiment，比较两种模式：

- **Full GSPO**：更新所有参数（vanilla veRL）
- **MoE-only GSPO**：仅更新 MoE experts + router，冻结 Attention / LayerNorm / Embedding / LM Head

唯一变量为 `parameter_update.mode`，其余超参数完全相同。

---

## 2. 实验设置

### 2.1 模型

| 项目 | 值 |
|------|-----|
| 模型 | `allenai/OLMoE-1B-7B-0125-Instruct` |
| 总参数量 | 6,919,161,856 (~6.92B) |
| 架构 | OlmoeForCausalLM (native sparse MoE) |
| 层数 | 16 |
| Expert 数 | 64 |
| Top-K | 8 |
| Hidden size | 2048 |

### 2.2 参数分类

| 类别 | 参数量 | 占比 | Full | MoE-only |
|------|--------|------|------|----------|
| Attention | 268,500,992 | 3.88% | Trainable | **Frozen** |
| Experts | 6,442,450,944 | 93.13% | Trainable | Trainable |
| Router (gate) | 2,097,152 | 0.03% | Trainable | Trainable |
| LayerNorm | 67,584 | 0.001% | Trainable | **Frozen** |
| Embedding | 103,022,592 | 1.49% | Trainable | **Frozen** |
| LM Head | 103,022,592 | 1.49% | Trainable | **Frozen** |

MoE-only 模式可训练参数：6,444,548,096 (93.14%)，冻结参数：474,613,760 (6.86%)。

### 2.3 RL 算法

| 项目 | 值 |
|------|-----|
| RL 框架 | veRL 0.9.0.dev (commit `1bfd02c4`) |
| Policy Loss | GSPO (sequence-level importance ratio) |
| Advantage Estimator | GRPO (group-relative) |
| Clip ratio low | 3e-4 |
| Clip ratio high | 4e-4 |
| Clip ratio c (dual-clip) | 10.0 |
| Loss aggregation | seq-mean-token-mean |
| KL in reward | False |
| KL loss | False |
| Entropy coefficient | 0 |
| Learning rate | 1e-6 |
| Optimizer | AdamW |

### 2.4 训练配置

两组实验配置：

| 配置项 | Smoke Test (工程验证) | Pilot Test (正式实验) |
|--------|----------------------|----------------------|
| 数据集 | GSM8K (train 7473, test 1319) | GSM8K (train 7473, test 1319) |
| Train batch size | 64 | 256 |
| PPO mini batch size | 32 | 64 |
| Rollout n | 8 | 16 |
| Max prompt length | 512 | 1024 |
| Max response length | 512 | 1024 |
| PPO max token len / GPU | 8192 | 16384 |
| Rollout TP | 2 | 2 |
| Rollout GPU mem util | 0.5 | 0.6 |
| Total epochs | 1 | 15 |
| Save freq | 100 | 10 |
| Test freq | 100 | 5 |
| Seed | 42 | 42 |

### 2.5 基础设施

| 项目 | 值 |
|------|-----|
| GPU | 8x NVIDIA H100 80GB (每 run 1 节点) |
| 训练 backend | FSDP2 (fully_shard) |
| Rollout backend | vLLM 0.11.0 |
| PyTorch | 2.8.0+cu128 |
| Transformers | 4.56.1 |
| Ray | 2.55.0 |
| Python | 3.12.0 |

### 2.6 正确性验证

三层验证确保 parameter freezing 生效：

1. **requires_grad 检查**：16 个单元测试全部通过，验证各模式下参数的 `requires_grad` 状态正确。
2. **梯度检查**：MoE-only 模式下 backward 后，frozen 参数 (attention/norm/embedding/lm_head) 的 `grad` 为 `None`，trainable 参数 (experts/router) 的 `grad` 不为 `None`。
3. **Checkpoint delta 验证**：在真实 7B 模型上训练后，对比原始权重与训练后 checkpoint：

| 类别 | Smoke (step 116) max_abs_delta | Pilot (step 30) max_abs_delta | 预期 |
|------|-------------------------------|------------------------------|------|
| Attention | 0.0 | 0.0 | 不变 |
| LayerNorm | 0.0 | 0.0 | 不变 |
| Embedding | 0.0 | 0.0 | 不变 |
| LM Head | 0.0 | 0.0 | 不变 |
| Router | 8.97e-05 | 5.77e-05 | 变化 |
| Experts | 1.08e-04 | 7.02e-05 | 变化 |

Frozen 参数 bitwise identical（delta = 0.0），trainable 参数有变化。验证通过。

---

## 3. 实验结果

### 3.1 Smoke Test 结果（116 步，小 batch 配置）

| 指标 | Full GSPO | MoE-only GSPO |
|------|-----------|---------------|
| 完成步数 | 116 / 116 | 116 / 116 |
| 训练 reward (首) | 0.3555 | 0.3594 |
| 训练 reward (末) | 0.8223 | 0.0000 |
| 训练 reward (峰值) | 0.8926 | 0.8809 |
| 验证准确率 (末) | 0.7346 | 0.0000 |
| Grad norm (范围) | 1.6 ~ 289.9 | 0.0 ~ 28831.6 |
| Response length (末) | 313.6 | 1.0 |
| Peak GPU memory | 18.40 GB | 17.53 GB |
| 训练状态 | 全程稳定 | **step ~85 梯度爆炸，模型崩溃** |

**Smoke test 结论**：Full GSPO 全程稳定完成；MoE-only GSPO 在 step 85 前学习良好（reward 峰值 0.88，与 Full 相当），之后发生梯度爆炸（grad_norm 飙升至 28,831），模型输出退化为 1 token 空回复，reward 归零。

### 3.2 Pilot Test 结果（大 batch 配置）

| 指标 | Full GSPO | MoE-only GSPO |
|------|-----------|---------------|
| 完成步数 | 31 (崩溃退出) | 99 (磁盘满退出) |
| 训练 reward (首) | 0.3650 | 0.3623 |
| 训练 reward (末) | 0.6614 | 0.7981 |
| 训练 reward (峰值) | 0.8550 (step 21) | 0.8792 (step 40) |
| 验证准确率 (首) | 0.3571 | 0.3571 |
| 验证准确率 (峰值) | 0.7392 (step 20) | 0.7612 (step 65) |
| 验证准确率 (末) | 0.5671 (step 30) | 0.7445 (step 95) |
| Grad norm (范围) | 4.7 ~ 1356.2 | 0.2 ~ 40.7 |
| Grad norm (末) | 1356.2 | 0.37 |
| Response length (末) | 72.4 | 283.7 |
| PG loss (末) | 61.07 | 0.075 |
| PPO KL (末) | 0.259 | 0.045 |
| PG clip frac (末) | 0.334 | 0.239 |
| Peak GPU memory | 20.84 GB | 19.85 GB |
| Step time (末) | 47.6s | 77.0s |
| 训练状态 | **step ~28 梯度爆炸，崩溃** | 全程稳定 |

**Pilot test 结论**：Full GSPO 在 step 28 左右发生梯度爆炸（grad_norm 从 ~10 飙至 1356），reward 下跌、response length 从 258 缩短至 72 token，程序在 step 31 因 `KeyError: loss_mask` 退出。MoE-only GSPO 全程 99 步保持稳定，验证准确率从 35.7% 提升至 76.1%（2.13x），最终因磁盘空间不足而终止，非训练失败。

### 3.3 两阶段结果对比

| | Smoke (小 batch) | Pilot (大 batch) |
|---|---|---|
| Full GSPO | 稳定完成 | 梯度爆炸崩溃 (step 31) |
| MoE-only GSPO | 梯度爆炸崩溃 (step ~85) | 稳定运行 (99 步) |

两组实验中"谁崩溃"的结果不一致，说明两种模式的稳定性均依赖超参数配置（batch size、rollout 数、response 长度等），不能简单断言某一种模式更稳定。

### 3.4 MoE RL Retention Ratio

由于 Full GSPO 在 pilot 中过早崩溃，无法直接计算完整的 retention ratio。以可比步数估算：

| 指标 | Full (step 20, 崩溃前) | MoE-only (step 20) |
|------|----------------------|---------------------|
| 训练 reward | 0.845 | 0.816 |
| 验证准确率 | 0.739 | 0.726 |

在可比步数上，MoE-only 的 reward 和 accuracy 与 Full 接近。估算 retention ratio ρ ≈ 0.95~1.0。

---

## 4. 分析与讨论

### 4.1 RQ1 回答

> 当 Full GSPO 可以学习时，仅更新 pretrained native MoE experts + router，是否也能够产生显著 RL improvement？

**是的。** MoE-only GSPO 在 pilot 配置下产生了显著的 RL improvement：

- 验证准确率：35.7% → 76.1%（提升 2.13x）
- 训练 reward：0.36 → 0.80（提升 2.2x）
- 学习曲线清晰、稳定

这证明仅更新 MoE experts + router 确实可以获得 RL learning signal。

### 4.2 稳定性问题

两种模式在不同配置下都出现了梯度爆炸：

- Smoke（小 batch 64, rollout 8）：MoE-only 在 step 85 爆炸，Full 稳定
- Pilot（大 batch 256, rollout 16）：Full 在 step 28 爆炸，MoE-only 稳定

爆炸的共同特征是 grad_norm 突然飙升 2~3 个数量级，随后模型输出退化为极短或空回复，reward 归零。这属于 RL 训练中的 mode collapse。可能的原因包括：

- Learning rate (1e-6) 对于某些配置可能偏高
- Gradient clipping 阈值 (clip_grad=1.0) 可能不足以防止极端梯度
- GSPO 的 sequence-level clipping 在特定 batch 配置下可能不够稳定

后续实验应尝试降低 learning rate、加强 gradient clipping、或加入 KL penalty 来提高两种模式的稳定性。

### 4.3 显存对比

| 配置 | Full GSPO | MoE-only GSPO | 节省 |
|------|-----------|---------------|------|
| Smoke peak GPU memory | 18.40 GB | 17.53 GB | 0.87 GB |
| Pilot peak GPU memory | 20.84 GB | 19.85 GB | 0.99 GB |

MoE-only 模式每 GPU 节省约 1 GB 显存，主要来自优化器不再为 frozen 参数分配 state（exp_avg, exp_avg_sq）。由于 OLMoE 中 93% 的参数是 experts（trainable），frozen 的 attention/embedding 等只占 7%，因此显存节省幅度有限。

### 4.4 Optimizer State 验证

通过对比 checkpoint 中 optimizer state 文件大小确认了 optimizer 参数过滤生效：

| 模式 | optim shard 大小 (每 rank) |
|------|---------------------------|
| Full GSPO | 6,922,817,529 bytes |
| MoE-only GSPO | 6,448,033,352 bytes |

MoE-only 的 optimizer state 更小，因为只包含 trainable 参数（experts + router）的 AdamW state。

---

## 5. 代码改动

### 5.1 核心改动（4 个文件）

| 文件 | 改动 |
|------|------|
| `verl/utils/parameter_update_policy.py` | **新增**：`apply_parameter_update_policy(model, mode, strict)` 函数，支持 full/moe/experts_only 三种模式，基于 `isinstance(module, OlmoeSparseMoeBlock)` 进行 class-based dispatch |
| `verl/workers/config/actor.py` | **新增** `ParameterUpdateConfig` dataclass + 添加到 `ActorConfig` |
| `verl/workers/engine/fsdp/transformer_impl.py` | 在 `_build_model_optimizer` 中 model load 后、FSDP wrap 前插入 policy 应用；`_build_optimizer` 改为 `[p for p in module.parameters() if p.requires_grad]` |
| `verl/trainer/config/actor/actor.yaml` | 新增 `parameter_update` YAML block，默认 `mode: full` |

### 5.2 向后兼容性

默认 `mode: full` 为 no-op，不指定该配置时 veRL 行为与原始版本完全一致。单元测试 `test_unknown_model_full_ok` 验证了未知模型在 full 模式下正常工作。

### 5.3 测试

| 测试文件 | 测试数 | 状态 |
|----------|--------|------|
| `tests/models/test_parameter_update_policy.py` | 12 | 全部通过 |
| `tests/models/test_checkpoint_delta.py` | 4 | 全部通过 |

---

## 6. 可复现性信息

| 项目 | 值 |
|------|-----|
| veRL commit | `1bfd02c4359a44310759aab7c0382e2bf5836756` |
| 本项目改动 commit | `ee5719374c02b49b87ad2b88ae4d8c0221dca4ec` |
| PyTorch | 2.8.0+cu128 |
| Transformers | 4.56.1 |
| vLLM | 0.11.0 |
| CUDA | 12.8 |
| GPU | NVIDIA H100 80GB x8 |
| 模型 checkpoint | `allenai/OLMoE-1B-7B-0125-Instruct` |
| 数据 | GSM8K (openai/gsm8k) |
| Seed | 42 |
| 完整 Hydra config | 见 SLURM 日志 `outputs/slurm/` |

---

## 7. 结论与下一步

### 结论

1. **MoE-only GSPO 能产生显著 RL improvement**：仅更新 experts + router，验证准确率从 35.7% 提升至 76.1%，证明 parameter-localized RL hypothesis 在 OLMoE 上成立。
2. **Retention ratio ρ ≈ 0.95~1.0**：在可比步数上，MoE-only 与 Full 的 reward 和 accuracy 接近。
3. **稳定性未决**：两种模式在不同配置下都出现过梯度爆炸，不能断言某一种更稳定。需要进一步调参。
4. **显存节省有限**：由于 OLMoE 中 93% 参数是 experts（trainable），MoE-only 模式仅节省约 1 GB/GPU。
5. **Checkpoint 验证通过**：frozen 参数在训练后 bitwise identical，parameter freezing 机制正确。

### 下一步建议

1. **提高稳定性**：降低 learning rate 至 5e-7，加强 gradient clipping，或加入 KL penalty，使两种模式都能稳定完成 435 步。
2. **跑完完整 pilot**：当前 pilot 因磁盘空间和崩溃未能完成 435 步，需要在稳定后重跑。
3. **运行 Experts-only 模式**：如果 MoE-only 稳定，进一步运行 `experts_only`（冻结 router），回答"router adaptation 是否必要"。
4. **扩展到 Qwen3-30B-A3B**：如果 OLMoE 上 MoE-only 持续显示 positive signal，扩展到更大的现代 MoE 模型。
