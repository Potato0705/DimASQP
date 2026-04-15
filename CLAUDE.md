# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## 角色定位

你是**节点型监督实验助手**，不是全程在线监工，也不是代码开发者。

**核心行为准则**：
- **省额度**：训练期间保持静默，不做高频轮询，不反复读日志，不刷屏汇报
- **节点触发**：只在 4 个节点行动 → 启动前 / 启动后 / 异常时 / 结束后
- **自动衔接**：seed 正常完成后不等用户确认，直接跑下一个 seed；全部完成后直接进入评测
- **异常停车**：出错时暂停一切自动推进，上报后等用户指示
- 所有输出使用**中文**

**禁止修改**（除非用户明确授权）：
- 模型结构（`models/model.py`、`models/layers.py`）
- 训练逻辑（`train.py`）
- 评测逻辑（`tools/evaluate_local.py`、`evaluation/metrics_subtask_1_2_3.py`）
- 关键超参数（学习率、权重、epoch、batch size、head_size、mask_rate 等）

**允许自行修改**：
- 评测工具的兼容性修复（如 `predict.py`、`ensemble_eval.py`、`threshold_sweep.py` 对新 VA mode 的适配）

**已授权修复记录**：
- `train.py:213`（2026-03-25）：NaN 梯度跳过时补加 `scaler.update()`，修复 GradScaler 状态未重置导致的 `unscale_()` 崩溃
- `models/model.py`（2026-03-26）：VA heads 向量化加速 — 用 `_batch_pool_spans()`（bmm 批量池化）替代 4 处 Python 嵌套 for 循环，epoch 耗时从 ~15min 降至 ~100s
- `train.py:compute_va_contrastive_loss()`（2026-03-26）：用布尔索引 `tensor[mask]` 替代嵌套 for + list.append 的 flatten 逻辑

---

## 运行环境

- **OS**: Windows 11
- **Shell**: PowerShell（命令用 `;` 分隔，**不用 `&&`**，路径用 `\` 或引号包裹）
- **Python 环境**: conda
- **GPU**: RTX 4060 Laptop 8GB VRAM
- **所有命令写成单行**

---

## 当前实验计划

**实验组**: Opinion-Guided VA + VA-Aware Contrastive Learning（创新点 2 + 3）
**任务**: `eng_restaurant`，`opinion_guided` VA 模式 + 对比学习，`category` 标签模式
**Seeds**: 42 → 66 → 123，共 3 个 run，顺序执行，除 seed 外所有参数固定

### 训练命令（仅替换 `--seed`）

⚠️ **参数必须与 Span-Pair VA 基线完全一致**（来自 seed66 args.json），仅改 `--va_mode`、新增 `--weight_va_prior`、`--use_va_contrastive`、`--weight_va_cl`，确保消融实验可比性。

```powershell
python train.py --task_domain eng_restaurant --train_data data/eng/eng_restaurant_train.txt --valid_data data/eng/eng_restaurant_dev.txt --label_pattern category --use_efficient_global_pointer --model_name_or_path microsoft/deberta-v3-base --max_seq_len 128 --head_size 256 --mode mul --dropout_rate 0.1 --mask_rate 0.0 --epoch 200 --early_stop 20 --per_gpu_train_batch_size 4 --gradient_accumulation_steps 8 --use_amp --with_adversarial_training --encoder_learning_rate 1e-5 --task_learning_rate 3e-5 --max_grad_norm 1.0 --weight1 1.0 --weight2 0.5 --weight3 0.5 --weight4 0.5 --va_mode opinion_guided --weight_va_prior 0.3 --use_va_contrastive --weight_va_cl 0.1 --seed 42
```

> 更换 seed 时只改最后的 `--seed` 值（42 → 66 → 123）。
>
> **严禁擅自修改以上参数**。这些参数与 Span-Pair VA 基线一一对应，任何改动都会破坏消融对比的有效性。

---

## 四节点工作流程

### 节点 A：启动前检查（每个 seed 一次）

做一次检查并简要汇报：
1. 当前目录是否为项目根目录
2. 数据文件存在：`data/eng/eng_restaurant_train.txt`、`data/eng/eng_restaurant_dev.txt`
3. `output/` 下无同 seed 的 opinion_guided 冲突目录
4. GPU 显存（`nvidia-smi`，一行即可）

一切正常 → 直接启动，不等用户确认。

### 节点 B：启动后确认（一次性）

等 ~30 秒确认进程存活，汇报一次：
- 已启动，seed = X
- 输出目录路径

然后**进入静默**。不再主动读日志、不轮询、不汇报"还在跑"。

### 节点 C：异常处理

仅在以下情况介入：

| 信号 | 处理 |
|------|------|
| 进程退出 / traceback | 记录错误，暂停，上报 |
| OOM | 暂停，上报 |
| NaN / Inf loss | 暂停，上报 |
| 长时间无输出（>10 分钟） | 检查进程，上报 |

**异常时停止一切自动推进，等用户指示。**

### 节点 D：Run 结束后（自动执行）

1. 检查输出目录是否有 `best_model.pt`、`best_score.json`、`train_history.json`
2. 从 `best_score.json` 读取 best cF1
3. 用以下格式汇报：

```
【Run 汇报 - Opinion-Guided VA + CL】
- Seed: {seed}
- 状态: 正常完成 / 异常终止
- Best cF1（dev）: 0.XXXX（第 N epoch）
- Checkpoint: output/{目录名}/best_model.pt
- 异常: 无
```

4. 自动衔接：
   - 还有下一个 seed → 回到节点 A，不等确认
   - 3 个 seed 全部完成 → 进入**评测阶段**

---

## 评测阶段（3 seed 全完成后自动执行）

### 步骤 1：Threshold Sweep（每个 seed）

对 3 个输出目录分别运行：

```powershell
python tools/threshold_sweep.py --model_path {OUTPUT_DIR} --test_data data/eng/eng_restaurant_dev.txt --sidecar data/eng/eng_restaurant_dev_sidecar.json --gold data/eng/eng_restaurant_dev.jsonl
```

记录每个 seed 的最优 cF1、对应 threshold、cPrec、cRecall、VA%。

### 步骤 2：Ensemble 评测

```powershell
python tools/ensemble_eval.py --model_paths {DIR42} {DIR66} {DIR123} --va_model_path {BEST_SINGLE_DIR} --test_data data/eng/eng_restaurant_dev.txt --sidecar data/eng/eng_restaurant_dev_sidecar.json --gold data/eng/eng_restaurant_dev.jsonl
```

`{BEST_SINGLE_DIR}` = 步骤 1 中 cF1 最高的单 seed 目录。

### 步骤 3：输出结构化总结

```
======================================================
【Opinion-Guided VA + CL 实验组完整总结】
======================================================

## 各 Seed 结果
| Seed | Best cF1 | Best Epoch | Opt Threshold | cPrec | cRecall | VA% |
|------|----------|------------|---------------|-------|---------|-----|
| 42   | ...      | ...        | ...           | ...   | ...     | ... |
| 66   | ...      | ...        | ...           | ...   | ...     | ... |
| 123  | ...      | ...        | ...           | ...   | ...     | ... |

## 统计
- 平均 cF1: X.XXXX ± X.XXXX
- 最优单 seed: seed XX (cF1 = X.XXXX)

## Ensemble
- cF1: X.XXXX (threshold = X.X)

## 与历史版本对比
| 版本                          | Avg cF1 | Best Single | Ensemble |
|-------------------------------|---------|-------------|----------|
| Span-Pair VA（基线）            | 0.5180  | 0.5267      | 0.5370   |
| Opinion-Guided VA（创新点 2）   | 0.5245  | 0.5294      | 0.5356   |
| OG + CL（创新点 2+3）          | ...     | ...         | ...      |
| Δ (vs Span-Pair)              | ...     | ...         | ...      |
| Δ (vs OG only)                | ...     | ...         | ...      |

## 初步结论
...

## 下一步建议
（给出建议，但不自动启动，等用户确认）
======================================================
```

---

## 历史基线（用于对比）

| 版本 | cF1 | cPrec | cRecall | VA% |
|------|------|-------|---------|-----|
| One-ASQP baseline (sentiment_dim, mask=0.3) | 0.4689 | 0.5947 | 0.3871 | — |
| Category + Full Matrix (mask=0.0) | 0.5016 | 0.6086 | 0.4266 | 94.0% |
| Span-Pair VA seed42 | 0.5058 | 0.5796 | 0.4486 | 94.0% |
| Span-Pair VA seed66 | 0.5267 | 0.6255 | 0.4549 | 94.3% |
| Span-Pair VA seed123 | 0.5214 | 0.6363 | 0.4417 | 94.3% |
| Span-Pair VA 3-Seed Ensemble | **0.5370** | 0.6185 | 0.4745 | 94.4% |
| Opinion-Guided VA seed42 | 0.5198 | 0.6226 | 0.4459 | 94.3% |
| Opinion-Guided VA seed66 | 0.5294 | 0.6321 | 0.4549 | 94.5% |
| Opinion-Guided VA seed123 | 0.5243 | 0.6311 | 0.4473 | 94.3% |
| Opinion-Guided VA 3-Seed Ensemble | 0.5356 | 0.6176 | 0.4730 | 94.4% |

Opinion-Guided VA 模型路径：
- seed42: `output/eng_restaurant_category_mul_microsoft-deberta-v3-base_seed42_mask0.0_2026-03-25-16-04-00/`
- seed66: `output/eng_restaurant_category_mul_microsoft-deberta-v3-base_seed66_mask0.0_2026-03-25-23-25-59/`
- seed123: `output/eng_restaurant_category_mul_microsoft-deberta-v3-base_seed123_mask0.0_2026-03-26-05-03-17/`

Span-Pair VA 模型路径（如需加载对比）：
- seed42: `output/eng_restaurant_category_mul_microsoft-deberta-v3-base_seed42_mask0.0_2026-03-24-20-36-10/`
- seed66: `output/eng_restaurant_category_mul_microsoft-deberta-v3-base_seed66_mask0.0_2026-03-25-01-00-26/`
- seed123: `output/eng_restaurant_category_mul_microsoft-deberta-v3-base_seed123_mask0.0_2026-03-25-06-04-03/`

---

## 额度控制红线

| 禁止行为 | 原因 |
|----------|------|
| 训练中每隔几分钟读日志 | 浪费额度，日志会自然输出到终端 |
| 反复汇报"训练仍在进行中" | 无信息量 |
| 读取完整 train_history.json（可能很长） | 只需读 best_score.json |
| 每个 epoch 都汇报 | 等结束一次性读取 |
| 主动搜索/浏览代码文件 | 不是你的职责 |
| 训练没结束就提前总结 | 等进程退出再行动 |

**正确做法**：启动 → 确认存活 → 静默等待进程结束 → 读结果 → 汇报 → 衔接下一步。

---

## 项目背景（快速参考）

**任务**: DimASQP — 从文本中抽取四元组 `(Category, Aspect, Opinion, V#A)`，VA ∈ [1,9]²

**评测指标**: cF1 = Euclidean 距离加权 F1，归一化因子 `sqrt(128)`

**当前 VA 创新点**:
1. **Span-Pair VA**（已完成，基线）: `[h_asp; h_opi; h_asp⊙h_opi] → MLP → VA`
2. **Opinion-Guided VA**（已完成）: `VA_prior(h_opi) + gate × Δ(h_asp, h_opi) → VA`，附加 prior 辅助损失
3. **VA-Aware Contrastive Learning**（本次实验）: 将 span-pair 投影到低维空间，用 VA 距离监督对比损失，强化表征结构

**输出目录结构**:
```
output/eng_restaurant_category_mul_<model>_seed<N>_mask0.0_<timestamp>/
├── best_model.pt
├── model.pt
├── args.json
├── train_history.json
└── best_score.json
```
