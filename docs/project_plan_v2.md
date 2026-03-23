# DimABSA 2026 Track A / Subtask 3 (DimASQP) 项目计划书 v2

> 更新日期：2026-03-23
> GitHub: https://github.com/Potato0705/DimASQP

---

## 一、项目概述

### 1.1 赛题说明

**DimASQP (Dimensional Aspect Sentiment Quad Prediction)** 要求从文本中抽取四元组：

```
(Aspect, Category, Opinion, VA)
```

其中 VA 是连续的 Valence-Arousal 值（范围 [1.00, 9.00]），而非传统的离散情感标签。

### 1.2 官方评测指标

**cF1 (Continuous F1)**：
- 匹配键：`(Aspect, Opinion, Category)` 精确匹配（大小写不敏感）
- 匹配成功后：`cTP = 1 - euclidean_distance(VA_pred, VA_gold) / sqrt(128)`
- `cPrecision = Σ cTP / (TP + FP)`，`cRecall = Σ cTP / (TP + FN)`
- `cF1 = 2 * cPrecision * cRecall / (cPrecision + cRecall)`

### 1.3 参赛语言与领域

| 语言 | 领域 | 训练集大小 | 类别数 | 资源等级 |
|------|------|-----------|--------|---------|
| eng | restaurant | 1,530 | 14 | 高 |
| eng | laptop | 2,934 | 121 | 高 |
| zho | restaurant | 5,180 | 12 | 高 |
| zho | laptop | 2,813 | 113 | 高 |
| jpn | hotel | 1,360 | 44 | 中 |
| rus | restaurant | 1,092 | 12 | 低 |
| tat | restaurant | 1,092 | 12 | 低 |
| ukr | restaurant | 1,092 | 12 | 低 |

### 1.4 硬件约束

- GPU: NVIDIA RTX 4060 Laptop (8GB VRAM)
- 只能使用 base 级别模型
- batch_size ≤ 8-16，必须开 AMP + 梯度累积

---

## 二、技术方案

### 2.1 基线方法：One-ASQP (ACL 2023) 改造

基于 Token-Pair Matrix 方法：
- **编码器**：DeBERTa-v3-base（英文）/ 多语言 PLM（其他语言）
- **Matrix Head**：EfficientGlobalPointer + RoPE 构建 [N_labels, L, L] 矩阵
  - BA-BO 定位 aspect 起始 → opinion 起始
  - EA-EO 定位 aspect 结束 → opinion 结束
  - BA-EO-{category} 定位 aspect 起始 → opinion 结束，同时编码类别
- **VA 回归头**：`Linear → ReLU → Linear → Sigmoid*8+1` 映射到 [1, 9]
- **辅助头**：CLS 维度分类（BCE）、维度序列标注

### 2.2 模型架构（当前实现）

```
DeBERTa Encoder
    ├── Matrix Head (EfficientGlobalPointer)
    │     → [B, N_labels, L, L]  # BA-BO + EA-EO + BA-EO-{cat}
    ├── Dimension CLS Head (Linear → BCE)
    │     → [B, N_dim]  # 篇章级类别存在性
    ├── Dimension Sequence Head (Linear → GlobalPointer Loss)
    │     → [B, N_dim, L]  # token 级类别标注
    ├── Sentiment Sequence Head (保留，category 模式下不使用)
    │     → [B, 3, L]
    └── VA Regression Head (NEW)
          → [B, L, 2]  # Sigmoid*8+1 → [1, 9]
```

### 2.3 label_pattern 模式

| 模式 | Matrix 标签含义 | BA-EO 标签数 | 适用场景 |
|------|----------------|-------------|---------|
| `sentiment_dim` | BA-EO-{cat}-{sent} | cat × 3 | 原始 ACOS 复现 |
| `sentiment` | BA-EO-{sent} | 3 | Phase 1 baseline |
| **`category`** | **BA-EO-{cat}** | **cat 个** | **Phase 2 当前方案** |
| `raw` | BA-EO | 1 | 最简单模式 |

---

## 三、项目进展

### 3.1 已完成工作

#### Phase 1：数据管线搭建（已完成 ✅）

| 步骤 | 文件/工具 | 说明 |
|------|----------|------|
| 数据划分 | `tools/split_dataset.py` | 智能划分：eng 按 ID 标记，其余按文档分组随机 |
| 格式转换 | `tools/convert_dimasqp.py` | JSONL → TXT + sidecar JSON，0 span miss |
| 配置生成 | `tools/generate_configs.py` | 扫描 JSONL 自动生成 category→id 映射 |
| 提交生成 | `tools/generate_submission.py` | 预测 JSON → 官方 JSONL 提交格式 |
| 评测集成 | `tools/evaluate_local.py` | 官方 cF1 逻辑本地复现 |
| 官方脚本 | `evaluation/metrics_subtask_1_2_3.py` | 原版评测脚本备份 |
| 训练适配 | `train.py` | 梯度累积、loss 重构、兼容性修复 |
| 兼容性 | `dataset.py`, `layers.py`, `utils.py` | MySQL 依赖移除、空 metrics 处理、PyTorch 2.6 兼容 |

#### Phase 2：架构改造（已完成 ✅）

| 改动 | 文件 | 说明 |
|------|------|------|
| category pattern | `dataset.py` | `get_label_types()` 新增 `"category"` 模式 |
| VA targets | `dataset.py` | `__getitem__` 新增 `va_targets`/`va_mask` 张量 |
| VA 回归头 | `model.py` | `va_linear + va_output + sigmoid*8+1` |
| VA 回归 loss | `train.py` | 加权 MSE loss，仅在 va_mask 标记位置计算 |
| category 解码 | `predict.py` | `create_pred_answer` 支持 category 模式 |
| VA 提取 | `predict.py` | `attach_va_to_pred_answer` 从 VA head 提取连续值 |

### 3.2 实验结果

#### eng_restaurant dev 集评测

| 指标 | Phase 1 (sentiment) | Phase 2 (category+VA) | 变化 |
|------|:---:|:---:|:---:|
| **cF1** | 0.4065 | **0.4689** | **+15.3%** ↑ |
| cPrecision | 0.4804 | **0.5947** | +23.8% ↑ |
| cRecall | 0.3524 | **0.3871** | +9.9% ↑ |
| Triplet F1 | 0.4913 | **0.4992** | +1.6% ↑ |
| VA 折扣因子 | 82.5% | **93.9%** | +11.4% ↑ |
| TP | 483 | 485 | — |
| FP | 306 | 281 | -8.2% ↓ (好) |
| FN | 694 | 692 | — |
| Gold 总四元组 | 1177 | 1177 | — |
| Pred 总四元组 | 789 | 766 | — |

**结论**：
- VA 回归头效果显著：93.9% 的折扣因子 vs 82.5% 的离散质心映射
- category 直接编码提升了 Precision（减少了 FP）
- **Recall 是当前主要瓶颈**（0.39），模型预测 766 个 vs 实际 1177 个，漏检严重

#### 训练细节

| 参数 | Phase 1 | Phase 2 |
|------|---------|---------|
| label_pattern | sentiment | category |
| 最佳 epoch | 58 / 100 | 100 / 100（未收敛） |
| matrix F1 (val) | 0.6817 | 0.5945 |
| batch_size | 8 × 4 = 32 | 8 × 4 = 32 |
| max_seq_len | 128 | 128 |
| mask_rate | 0.3 | 0.3 |
| encoder_lr | 1e-5 | 1e-5 |

### 3.3 Git 提交历史

```
e221d3f Fix predict.py for category pattern + VA string compatibility
3c7b3ac Phase 2: category label pattern + VA regression head
36fdeca Phase 1 data pipeline: split, convert, configs, evaluation, training adaptation
b865684 Initial commit: One-ASQP baseline + DimABSA 2026 Task 3 training data
```

### 3.4 项目文件结构

```
DimASQP/
├── configs/                      # 各语言-领域的 category→id 配置
│   ├── eng_restaurant.json (14)
│   ├── eng_laptop.json (121)
│   ├── zho_restaurant.json (12)
│   ├── zho_laptop.json (113)
│   ├── jpn_hotel.json (44)
│   ├── rus_restaurant.json (12)
│   ├── tat_restaurant.json (12)
│   └── ukr_restaurant.json (12)
├── data/                         # 训练数据（原始 JSONL + 转换后 TXT + sidecar）
│   ├── eng/ (restaurant + laptop)
│   ├── zho/ (restaurant + laptop)
│   ├── jpn/ (hotel)
│   ├── rus/ (restaurant)
│   ├── tat/ (restaurant)
│   └── ukr/ (restaurant)
├── dataset/dataset.py            # 数据集类（支持 4 种 label_pattern + VA targets）
├── models/
│   ├── model.py                  # QuadrupleModel（含 VA 回归头）
│   └── layers.py                 # EfficientGlobalPointer + RoPE + MetricsCalculator
├── losses/losses.py              # GlobalPointer CrossEntropy + 负采样
├── train.py                      # 训练循环（梯度累积 + 多任务加权 loss）
├── predict.py                    # 推理 + 解码（支持 category + VA 提取）
├── evaluation/                   # 官方评测脚本
│   └── metrics_subtask_1_2_3.py
├── tools/                        # 数据处理工具链
│   ├── split_dataset.py          # 训练集划分
│   ├── convert_dimasqp.py        # JSONL → TXT 转换
│   ├── generate_configs.py       # 类别配置生成
│   ├── generate_submission.py    # 提交文件生成
│   └── evaluate_local.py         # 本地 cF1 评测
├── scripts/                      # 训练启动脚本
│   └── train_eng_restaurant.sh
├── submission/                   # 提交文件
├── output/                       # 模型输出（best_model.pt, train_history.json）
└── utils/                        # 工具函数
    ├── argparse.py
    ├── adversarial.py (FGM)
    └── utils.py
```

---

## 四、问题分析与瓶颈

### 4.1 当前主要瓶颈：Recall 不足

| 问题 | 分析 | 影响 |
|------|------|------|
| **预测数量不足** | 766 pred vs 1177 gold = 仅覆盖 65% | Recall 上限被压低 |
| **隐式要素漏检** | NULL aspect/opinion 的 matrix 编码依赖 [SEP] position | 隐式四元组召回率低 |
| **category 标签空间大** | eng_restaurant 14 类，eng_laptop 121 类 | matrix 稀疏，学习困难 |
| **Phase 2 未收敛** | 100 epoch 跑满，score 仍在上升趋势 | 需要更多 epoch |

### 4.2 VA 预测表现

VA 折扣因子 93.9% 表明 VA 回归头已经相当准确，进一步提升空间有限。当前 cF1 的提升主要取决于三元组匹配的 Precision 和 Recall。

### 4.3 laptop 领域挑战

eng_laptop 有 121 个类别，matrix head 需要 123 个标签类型（121 + BA-BO + EA-EO），这对 8GB GPU 来说是严峻挑战（matrix 张量 [123, 128, 128]）。

---

## 五、后续优化计划

### Phase 3：Recall 提升（优先级最高）

| 编号 | 方案 | 原理 | 预期收益 | 工作量 |
|------|------|------|---------|--------|
| 3.1 | **延长训练 + 调参** | Phase 2 100 epoch 未收敛，增加到 200-300 epoch | Recall +5~10% | 低 |
| 3.2 | **降低 matrix 阈值** | 当前 >0 判正，改为 >-0.5 或可学习阈值 | Recall +3~5% | 低 |
| 3.3 | **降低 mask_rate** | 当前 0.3 负采样过于激进，降到 0.1 | Recall +2~3% | 低 |
| 3.4 | **数据增强** | 同义词替换、随机删除、回译 | Recall +3~5% | 中 |
| 3.5 | **多种子集成** | 3-5 个 seed 训练，union 预测结果 | Recall +5~8% | 中 |

**执行顺序**：3.1 → 3.2 → 3.3 → 3.5 → 3.4

### Phase 4：多语言扩展

| 编号 | 方案 | 说明 |
|------|------|------|
| 4.1 | **切换多语言 PLM** | XLM-RoBERTa-base 或 mDeBERTa-v3-base |
| 4.2 | **逐语言独立训练** | 用相同的 category pattern + VA head |
| 4.3 | **多语言联合训练** | 共享编码器，不同 category head |
| 4.4 | **低资源语言迁移** | 用高资源语言预训练，再在低资源上微调 |

**执行顺序**：4.1 → 4.2（先出 6 语言 baseline）→ 4.3 → 4.4

### Phase 5：创新架构（论文级改进）

| 编号 | 方案 | 原理 |
|------|------|------|
| 5.1 | **Span-Pair Conditioned VA** | 用 aspect-opinion span pair 的表征做 VA 预测，而非单 token 位置 |
| 5.2 | **2D Matrix Refinement** | 在 GlobalPointer 输出上加卷积/attention 层做局部细化 |
| 5.3 | **Category-VA Joint Head** | category 和 VA 共享中间层，互相增强 |
| 5.4 | **对比学习 VA 校准** | 用 (positive, neutral, negative) 样本的 VA 分布做对比学习 |

---

## 六、近期执行计划（1-2 周）

### Week 1：eng_restaurant 调优 + 全语言 baseline

| 天 | 任务 | 预期产出 |
|----|------|---------|
| Day 1 | Phase 2 延长训练（200 epoch）+ 调参实验 | cF1 目标 0.50+ |
| Day 2 | 阈值/mask_rate 消融实验 | 最优超参组合 |
| Day 3 | 多种子集成（3 seed） | cF1 目标 0.52+ |
| Day 4 | 切换 mDeBERTa-v3-base，训练 zho_restaurant | 中文 baseline |
| Day 5 | 训练 jpn_hotel + eng_laptop | 日文/laptop baseline |
| Day 6-7 | 训练 rus/tat/ukr（可能用多语言联合） | 全 6 语言 baseline |

### Week 2：提升 + 提交

| 天 | 任务 | 预期产出 |
|----|------|---------|
| Day 8-9 | Phase 5.1 Span-Pair Conditioned VA 实现 | 架构改进 |
| Day 10 | 全语言调优 + 集成 | 各语言最优模型 |
| Day 11-12 | 生成正式提交文件 + 验证 | 提交文件 |
| Day 13-14 | 实验报告 + 论文初稿 | 技术文档 |

---

## 七、风险与应对

| 风险 | 影响 | 应对措施 |
|------|------|---------|
| laptop 121 类别显存溢出 | 无法训练 eng_laptop/zho_laptop | 缩小 max_seq_len=96，降低 head_size |
| 低资源语言数据不足 | rus/tat/ukr 仅 1092 条 | 多语言联合训练，跨语言迁移 |
| 多语言 PLM 性能下降 | 英文不如 DeBERTa | 英文保持 DeBERTa，其余用 mDeBERTa |
| 训练时间过长 | 每个实验 ~2h（100 epoch） | 优先调参策略，避免全量搜索 |

---

## 八、关键指标追踪

当前 **eng_restaurant dev** 最佳结果：

| 指标 | 当前值 | 短期目标 | 最终目标 |
|------|--------|---------|---------|
| cF1 | **0.4689** | 0.52 | 0.58+ |
| cPrecision | 0.5947 | 0.62 | 0.65+ |
| cRecall | 0.3871 | 0.45 | 0.52+ |
| VA 折扣 | 93.9% | 95% | 96%+ |
