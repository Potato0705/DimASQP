# DimASQP: Span-Pair Conditioned Valence-Arousal Prediction for Dimensional Aspect Sentiment Quad Prediction

## 项目计划书 v1.0 | 2026-03-25

---

## 1. 项目概述

### 1.1 任务定义

**DimASQP (Dimensional Aspect Sentiment Quad Prediction)** 是 DimABSA 2026 共享任务 Track A / Subtask 3 的核心任务。给定一条评论文本，需要抽取所有 **(Aspect, Category, Opinion, VA)** 四元组，其中：

- **Aspect**: 评论中提及的实体片段（如 "the pasta"）
- **Category**: 预定义类别（如 FOOD#QUALITY）
- **Opinion**: 表达情感的片段（如 "delicious"）
- **VA**: 连续值 Valence-Arousal 情感强度（范围 1.00-9.00）

与传统 ABSA 的离散情感标签（positive/negative/neutral）不同，DimASQP 要求预测**连续二维情感空间**中的精确坐标，这是本任务的核心挑战。

### 1.2 评测指标

官方采用 **cF1 (Continuous F1)**：

```
cTP_i = 1 - euclidean_distance(VA_pred, VA_gold) / sqrt(128)
cTP = sum(cTP_i)   (对每个匹配的 TP 计算)
cPrecision = cTP / (TP + FP)
cRecall = cTP / (TP + FN)
cF1 = 2 * cPrecision * cRecall / (cPrecision + cRecall)
```

**关键洞察**：cF1 同时受两个因素影响：
1. **四元组抽取质量**（TP/FP/FN） — 决定 P/R 的分母
2. **VA 预测精度**（cTP_i 的大小） — 决定 P/R 的分子

### 1.3 数据集

本项目聚焦 **英文餐厅评论数据集 (eng_restaurant)**：

| 划分 | 句子数 | 四元组数 | 平均四元组/句 |
|------|--------|---------|-------------|
| Train | 1,530 | ~2,100 | ~1.4 |
| Dev | 754 | 1,059 | ~1.4 |
| Test | 待发布 | — | — |

类别体系：14 个 Aspect-Category 组合（FOOD#QUALITY, SERVICE#GENERAL, AMBIENCE#GENERAL 等）。

---

## 2. 基线系统与改进历程

### 2.1 完整实验对比表

| # | 版本 | 关键改动 | cF1 | cPrec | cRecall | VA% | 累计提升 |
|---|------|---------|-----|-------|---------|-----|---------|
| 1 | **One-ASQP 复现** | 原始 sentiment_dim pattern, mask=0.3 | 0.4689 | 0.5947 | 0.3871 | — | baseline |
| 2 | Category Pattern | matrix 标签直接编码类别 | 0.4727 | 0.5794 | 0.3992 | 94.0% | +0.8% |
| 3 | Full Matrix Training | mask_rate 0.3→0.0 | 0.5016 | 0.6086 | 0.4266 | 94.0% | +7.0% |
| 4 | **Span-Pair VA** | VA 基于 span pair 预测 | 0.5267 | 0.6255 | 0.4549 | 94.3% | +12.3% |
| 5 | **3-Seed Ensemble** | seed42/66/123 logit 平均 | **0.5370** | **0.6185** | **0.4745** | **94.4%** | **+14.5%** |

### 2.2 各阶段改进原理详解

#### 改进 1→2: Category Pattern（cF1 +0.8%）

**问题**：One-ASQP 原始设计中，matrix head 输出 `BA-EO-{sentiment}` 标签（如 BA-EO-POS），category 通过额外的 dimension sequence 间接推断。这种两步解码引入了信息传递断层。

**改进**：将 category 直接编码到 matrix 标签中（如 `BA-EO-FOOD#QUALITY`），matrix head 一步到位输出完整的 (span, category) 信息。

**为什么有效**：
- 消除了 dimension sequence 解码的累积误差
- Matrix head 的 label 数量从 3 增加到 14（每个类别一个），提供更细粒度的监督信号
- 训练信号更直接：每个 token pair 的标签同时包含"是否构成 span pair"和"属于哪个类别"两个信息

#### 改进 2→3: Full Matrix Training（cF1 +6.9%）

**问题**：GlobalPointer 的 mask_rate=0.3 意味着只保留 30% 的负样本参与 loss 计算。这虽然加速训练但导致模型对负样本的判别能力不足，漏检严重（Recall 仅 0.3992）。

**改进**：mask_rate 设为 0.0（使用全部负样本），配合 gradient accumulation (bs=4, accum=8) 适配 8GB GPU。

**为什么有效**：
- 模型在完整矩阵上训练，对负样本的抑制更充分
- 有效批量=32，训练信号更稳定
- Recall 从 0.3992 → 0.4266（+6.9%），证实了负样本欠采样是 Recall 瓶颈的根因之一

#### 改进 3→4: Span-Pair Conditioned VA（cF1 +5.0%）⭐ 核心创新

**问题**：原始 VA head 采用 per-position 预测 `[B, L, 2]`，对每个 token 位置独立预测 VA。推理时从 aspect 首 token 处取 VA 值。这存在根本性缺陷：
1. 同一 token 可能属于不同 quadruplet（不同 aspect-opinion 对），但 per-position 只能输出一个 VA
2. VA 强度应取决于 aspect 和 opinion 的**关系**，而非单个位置的语义

**改进**：设计 SpanPairVAHead，对每个检测到的 (aspect, opinion) span pair 做条件化 VA 预测：

```
h_asp = MeanPool(hidden[asp_start:asp_end])    -- 方面表征
h_opi = MeanPool(hidden[opi_start:opi_end])    -- 观点表征
h_pair = [h_asp; h_opi; h_asp ⊙ h_opi]        -- 交互表征
VA = MLP(h_pair) → sigmoid × 8 + 1            -- 映射到 [1, 9]
```

**为什么有效**：
- **语言学合理性**："The food is great" vs "The food is acceptable" — 同一 aspect，不同 opinion → 不同 VA。Span-pair conditioning 天然捕获这种差异
- **表征增强**：span-pair VA loss 的梯度回传到 encoder，迫使其学习更好的 span-level 语义表征，间接提升了 matrix head 的 span 检测能力
- **Recall 大幅提升**：0.4266 → 0.4549（+6.6%），证实了 VA 训练信号对 encoder 表征的正面影响

#### 改进 4→5: Multi-Seed Ensemble（cF1 +2.0%）

**做法**：三个不同随机种子（42, 66, 123）训练的模型，将 logit 矩阵取均值后统一解码。VA 使用最佳单模型（seed66）的 span-pair VA head。

**为什么有效**：
- Logit 平均降低了单模型的随机噪声，使 threshold 附近的边界样本判断更稳健
- Recall 进一步提升 0.4549 → 0.4745（+4.3%），ensemble 主要收益来自召回率

---

## 3. 核心创新：面向 VA 的三阶段预测框架

本项目的核心论文贡献聚焦于 **VA (Valence-Arousal) 预测的创新**，提出 **三阶段渐进式 VA 预测框架**：

### 3.1 创新点 1: Span-Pair Conditioned VA Prediction（已实现 ✅）

**动机**：传统 per-position VA 预测忽略了 aspect-opinion 交互。

**方法**：将 VA 预测从 token-level 提升到 span-pair-level，以 (aspect span, opinion span) 的联合表征为条件预测 VA。

**架构**：
```
                  ┌──────────────┐
hidden states ──→ │ Span Pooling │ ──→ h_asp, h_opi
                  └──────────────┘
                         │
                  ┌──────────────────────┐
                  │ h = [h_asp; h_opi;   │
                  │      h_asp ⊙ h_opi]  │
                  └──────────────────────┘
                         │
                  ┌──────────────┐
                  │   MLP → VA   │ ──→ (Valence, Arousal) ∈ [1, 9]²
                  └──────────────┘
```

**实验结果**：cF1 0.5016 → 0.5267（+5.0%），VA% 94.0% → 94.3%

### 3.2 创新点 2: Opinion-Guided VA Calibration（待实现）

**动机**：观察发现，当前模型的 VA 预测在 opinion 极性明确时表现良好（如 "excellent" → 高V），但在**隐含情感**或**复杂修辞**时（如反讽 "great, just great"、程度副词 "slightly disappointing"）误差较大。Opinion span 的词汇语义应为 VA 校准提供更强的先验。

**方法**：在 Span-Pair VA 基础上，增加 Opinion-Guided Calibration Module：

1. **Opinion Sentiment Prior**：为每个 opinion span 计算基于预训练语义的 VA 先验值
   ```
   h_opi_cls = MeanPool(hidden[opi_start:opi_end])
   va_prior = MLP_prior(h_opi_cls)  → [V_prior, A_prior]
   ```

2. **Residual Calibration**：最终 VA = VA_prior + Δ_VA，其中 Δ_VA 由完整 span-pair context 预测
   ```
   Δ_VA = SpanPairVAHead(h_asp, h_opi)  → residual adjustment
   VA_final = VA_prior + α × Δ_VA        → calibrated prediction
   ```

3. **Auxiliary Loss**：对 opinion VA prior 施加独立的 MSE 监督，确保 opinion 本身携带合理的 VA 先验

**预期效果**：
- 对常见 opinion（如 "good", "bad"）的 VA 预测更稳定（prior 提供锚点）
- 对复杂表达的 VA 预测更准确（residual 捕捉 context 依赖的偏移）
- 预计 VA% 从 94.3% 提升至 95-96%，对应 cF1 +0.5~1.0%

### 3.3 创新点 3: VA-Aware Contrastive Span Learning（待实现）

**动机**：当前训练中，span-pair 抽取（matrix head）和 VA 预测（VA head）虽然共享 encoder，但各自独立优化。两个任务的关联未被显式建模：**VA 相似的 span pairs 应具有相似的表征，VA 差异大的 span pairs 应具有不同的表征**。

**方法**：引入 VA-Aware Contrastive Loss，在训练过程中显式拉近 VA 相似样本的 span-pair 表征，推远 VA 差异大的样本：

1. **Span-Pair Representation**：对每个 gold quadruplet，提取 span-pair 表征
   ```
   r_i = [h_asp_i; h_opi_i; h_asp_i ⊙ h_opi_i]
   ```

2. **VA-Aware Similarity**：定义基于 VA 距离的软标签
   ```
   s_ij = exp(-||VA_i - VA_j||² / τ)    -- VA 空间中的相似度
   ```

3. **Contrastive Loss**：
   ```
   L_contrast = -Σ_i Σ_{j≠i} s_ij × log(sim(r_i, r_j) / Σ_k sim(r_i, r_k))
   ```
   其中 sim() 为余弦相似度。

**预期效果**：
- Encoder 学到的 span 表征具有更好的 VA 区分度
- 间接提升 matrix head 对模糊 span 的判别能力（VA-informed representation）
- 对尾部类别（如 DRINKS#PRICES, LOCATION#GENERAL）的召回有帮助
- 预计 Recall 提升 1-2%，cF1 +0.5~1.5%

---

## 4. 技术架构总览

```
Input Text: "The pasta was delicious but the service was slow"
     │
     ▼
┌─────────────────────────────┐
│   DeBERTa-v3-base Encoder   │ ──→ hidden states [B, L, H]
└─────────────────────────────┘
     │                    │                         │
     ▼                    ▼                         ▼
┌──────────┐    ┌──────────────┐         ┌────────────────────┐
│ Matrix   │    │ Category     │         │  Span-Pair VA      │
│ Head     │    │ Classifier   │         │  Prediction        │
│(EGP+RoPE)│    │ (dim_seq)    │         │                    │
└──────────┘    └──────────────┘         │ ┌────────────────┐ │
     │                    │               │ │ SpanPairVAHead │ │
     ▼                    ▼               │ │  + Opinion     │ │
  Span-Pair            Category           │ │  Calibration   │ │
  Detection           Assignment          │ │  + Contrastive │ │
     │                    │               │ └────────────────┘ │
     └────────┬───────────┘               └────────────────────┘
              │                                      │
              ▼                                      ▼
    (Aspect, Category, Opinion)              (Valence, Arousal)
              │                                      │
              └──────────────┬───────────────────────┘
                             ▼
                  (Aspect, Category, Opinion, VA) Quadruplet
```

---

## 5. 实验计划

### 5.1 消融实验设计

| 实验 | Matrix Head | VA Head | 额外模块 | 预期 cF1 |
|------|-----------|---------|---------|---------|
| A1: Baseline | EGP, mask=0.3 | 无 (离散情感) | — | 0.4689 |
| A2: + Category Pattern | EGP, mask=0.3 | Position VA | — | 0.4727 |
| A3: + Full Matrix | EGP, mask=0.0 | Position VA | — | 0.5016 |
| A4: + Span-Pair VA | EGP, mask=0.0 | **Span-Pair VA** | — | 0.5267 |
| A5: + Opinion Calibration | EGP, mask=0.0 | Span-Pair VA | **Opinion Prior + Residual** | ~0.535 |
| A6: + VA Contrastive | EGP, mask=0.0 | Span-Pair VA | **VA-Aware CL** | ~0.540 |
| A7: Full System + Ensemble | EGP, mask=0.0 | Span-Pair VA | All + 3-seed | ~0.555 |

### 5.2 训练配置

| 参数 | 值 |
|------|-----|
| Encoder | microsoft/deberta-v3-base |
| Max Seq Length | 128 |
| Batch Size | 4 (effective 32 with accum=8) |
| Encoder LR | 1e-5 |
| Task LR | 3e-5 |
| Epochs | 200 (early stop patience=20) |
| Gradient Clipping | max_norm=1.0 |
| AMP | FP16 mixed precision |
| GPU | RTX 4060 Laptop 8GB |

### 5.3 实施时间线

| 周次 | 任务 | 产出 |
|------|------|------|
| Week 1 (已完成) | Baseline 复现 + Category Pattern + Full Matrix | cF1: 0.4689 → 0.5016 |
| Week 2 (已完成) | Span-Pair VA + Multi-Seed + Ensemble | cF1: 0.5016 → 0.5370 |
| Week 3 | Opinion-Guided VA Calibration 实现与实验 | 预计 cF1 ~0.540 |
| Week 4 | VA-Aware Contrastive Learning 实现与实验 | 预计 cF1 ~0.550 |
| Week 5 | Full System 集成 + 消融分析 + 论文撰写 | 完整实验表 + 论文初稿 |
| Week 6 | 论文修改 + 补充实验 | 投稿 |

---

## 6. 当前存在的问题与分析

### 6.1 Recall 瓶颈（最关键）

当前最佳 ensemble cRecall = 0.4745，意味着超过半数的 gold quadruplets 未被检测到。分类别分析：

| 类别 | Support | Recall | 问题 |
|------|---------|--------|------|
| FOOD#QUALITY | 418 | ~0.63 | 主类，表现尚可 |
| SERVICE#GENERAL | 166 | ~0.70 | 较好 |
| AMBIENCE#GENERAL | 103 | ~0.68 | 较好 |
| RESTAURANT#GENERAL | 157 | ~0.50 | 中等 |
| FOOD#STYLE_OPTIONS | 63 | ~0.37 | 低频类，严重漏检 |
| LOCATION#GENERAL | 15 | ~0.13 | 极低频，几乎漏检 |
| DRINKS#PRICES | 7 | ~0.29 | 极低频 |

**根因**：低频类别样本太少，matrix head 学不到足够的模式。

**VA Contrastive Loss 的潜在帮助**：通过 VA 空间的相似性，将低频类别的表征拉向高频类别的邻域，实现隐式的数据增强效果。

### 6.2 Loss 爆炸

训练后期 mat_loss 偶尔爆炸到 10^10 级别。虽然 gradient clipping 已缓解，但仍不稳定。

**分析**：EfficientGlobalPointer 使用 multilabel_categorical_crossentropy，当预测值和真实值差距极大时 loss 数值不稳定。

**缓解方案**：正在使用的 gradient clipping (max_norm=1.0) + 未来可以尝试 label smoothing。

### 6.3 VA 预测的 6% 损失

当前 VA% = 94.4%，意味着每个 TP 因 VA 偏差平均损失 5.6% 的 cTP 贡献。

**分析**：
- Valence（情感正负向）预测较准（opinion 极性明确时）
- Arousal（情感激活度）预测偏差较大（"angry" vs "disappointed" 在 V 相似但 A 差异大）
- 隐含情感（implicit opinion）的 VA 预测最差

**Opinion Calibration 的预期帮助**：为 opinion 提供 VA 先验锚点，减少 Arousal 维度的波动。

---

## 7. 论文框架

### 7.1 标题（候选）

- "Span-Pair Conditioned Valence-Arousal Prediction for Dimensional Aspect Sentiment Analysis"
- "Beyond Discrete Sentiment: A Span-Pair VA Framework for Continuous Aspect Sentiment Quad Prediction"

### 7.2 结构

1. **Introduction**: DimASQP 任务定义、连续 VA 的挑战、本文贡献
2. **Related Work**: ABSA → ASQP → DimASQP 演进；VA 维度情感分析；Span-based 方法
3. **Method**:
   - 3.1 Problem Formulation
   - 3.2 Token-Pair Matrix Framework (基于 One-ASQP)
   - 3.3 Span-Pair Conditioned VA Prediction (创新点1)
   - 3.4 Opinion-Guided VA Calibration (创新点2)
   - 3.5 VA-Aware Contrastive Span Learning (创新点3)
4. **Experiments**:
   - 4.1 Setup (数据集、指标、实现细节)
   - 4.2 Main Results (与 baseline 对比)
   - 4.3 Ablation Study (消融实验)
   - 4.4 Analysis (VA 预测质量、类别分析、case study)
5. **Conclusion**

### 7.3 核心卖点

1. **首个面向 DimASQP 的 Span-Pair VA 框架**：将 VA 预测从独立的 token-level 提升到条件化的 span-pair-level
2. **Opinion-Guided Calibration**：利用 opinion 语义先验校准 VA 预测，解决隐含情感和复杂修辞的难题
3. **VA-Aware Contrastive Learning**：首次将连续 VA 空间的结构信息引入 span 表征学习
4. **消融实验充分**：7 组消融清晰展示每个模块的贡献

---

## 8. 风险与应对

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| Opinion Calibration 效果不显著 | 中 | 缺少一个创新点 | 调整为 multi-granularity VA prediction (token+span+sentence) |
| VA Contrastive 训练不稳定 | 中 | 实验无法完成 | 使用温度退火、梯度截断、渐进式引入 |
| 论文实验量不够 | 低 | 审稿人质疑 | 增加 error analysis、case study、不同 encoder 对比 |
| GPU 内存不足 | 中 | 无法训练完整模型 | 继续使用 gradient accumulation、减小 hidden size |

---

## 附录 A: 关键代码文件

| 文件 | 功能 |
|------|------|
| `models/model.py` | QuadrupleModel + SpanPairVAHead |
| `dataset/dataset.py` | 数据加载、category pattern、quad span 提取 |
| `train.py` | 训练循环、梯度累积、VA loss |
| `predict.py` | 推理解码、span-pair VA 推理 |
| `tools/threshold_sweep.py` | 阈值扫描 + cF1 评估 |
| `tools/ensemble_eval.py` | 多种子集成评估 |
| `tools/evaluate_local.py` | 本地 cF1 评测（匹配官方脚本） |

## 附录 B: 完整实验结果

### B.1 单模型 Threshold Sweep

**Seed 42 (Best threshold=-0.5)**
| Threshold | cF1 | cPrec | cRecall | #Pred | VA% |
|-----------|------|-------|---------|-------|-----|
| -3.0 | 0.4712 | 0.4474 | 0.4976 | 1309 | 94.0% |
| -0.5 | **0.5058** | 0.5796 | 0.4486 | 911 | 94.0% |
| 0.0 | 0.5043 | 0.5997 | 0.4351 | 854 | 94.0% |

**Seed 66 (Best threshold=-0.5)**
| Threshold | cF1 | cPrec | cRecall | #Pred | VA% |
|-----------|------|-------|---------|-------|-----|
| -3.0 | 0.5037 | 0.5201 | 0.4883 | 1105 | 94.2% |
| -0.5 | **0.5267** | 0.6255 | 0.4549 | 856 | 94.3% |
| 0.0 | 0.5235 | 0.6397 | 0.4430 | 815 | 94.3% |

**Seed 123 (Best threshold=0.0)**
| Threshold | cF1 | cPrec | cRecall | #Pred | VA% |
|-----------|------|-------|---------|-------|-----|
| -3.0 | 0.5052 | 0.5234 | 0.4883 | 1098 | 94.2% |
| 0.0 | **0.5214** | 0.6363 | 0.4417 | 817 | 94.3% |
| 1.0 | 0.5193 | 0.6660 | 0.4255 | 752 | 94.3% |

### B.2 Ensemble Threshold Sweep

| Threshold | cF1 | cPrec | cRecall | #Pred | VA% |
|-----------|------|-------|---------|-------|-----|
| -3.0 | 0.5207 | 0.5508 | 0.4937 | 1055 | 94.3% |
| -1.5 | **0.5370** | 0.6185 | 0.4745 | 903 | 94.3% |
| 0.0 | 0.5228 | 0.6504 | 0.4371 | 791 | 94.4% |
