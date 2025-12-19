# DimASQP / One_ASQP  
## SemEval-2026 Task 3（DimABSA / ACQP）多语言四元组抽取系统

本仓库为本人在 **SemEval-2026 Task 3（Dimensional Aspect-based Sentiment Analysis, ACQP）** 比赛中的实验代码仓库，  
以 **One_ASQP** 作为基础框架，在其上进行了**多项方法级与工程级改进**，目标评价指标为官方 **cF1**（使用 `metrics_subtask_1_2_3.py -t 3`）。

系统目标：  
从文本中抽取 **四元组（Quadruplet）**：

> **(Aspect, Category, Opinion, VA)**  
> 其中 VA = Valence + Arousal（连续值）

---

## 1. 项目整体思路

### 1.1 Baseline
- 采用 **One_ASQP** 作为基础框架
- Encoder + 多头解码结构（实体 / 关系 / Category / VA）
- 使用 GlobalPointer / span-based 解码思想

### 1.2 本项目的主要改进方向
在 baseline 之上，本项目主要围绕以下几个方向进行扩展和优化：

1. **多语言统一建模**
   - 英文 / 中文 / 日文 / 俄文 / 乌克兰文 / 鞑靼文
   - 针对中文与多语言场景，引入 **mDeBERTa-v3-base**

2. **Category 决策的多源融合**
   - Head 直接预测（Cat Head）
   - Pair → Category 统计先验
   - Aspect → Category 统计先验
   - Global Category 兜底
   - 可配置的 `cat_source = head / prior / mix`

3. **解码阶段 anti-collapse 机制**
   - 防止大量样本塌缩为 `*_#GENERAL`
   - 在 global 或低置信度场景下，引入：
     - head top-k
     - asp→cat top-k
   - 明确区分 **prior 来源（pair / asp / global）**

4. **可解释的诊断统计（Diag）**
   - global 使用比例
   - null aspect / null opinion 比例
   - pair_hit / asp_hit
   - category 分布 Top-K
   - 防止“看似 cF1 提升但实际输出异常”

5. **参数可复现实验体系**
   - 所有实验配置集中在 `configs/`
   - sweep 搜索只影响推理，不破坏训练稳定性
   - 输出结构固定，便于对比

---

## 2. 目录结构说明（重点）

DimASQP/
├── train.py # 训练入口
├── predict.py # 推理入口
├── metrics_subtask_1_2_3.py # 官方风格评测脚本
├── sweep_predict.py # 推理阶段参数 sweep
│
├── configs/ # 各语言/领域配置
│ ├── eng-Laptop/
│ ├── eng-Restaurant/
│ ├── zho-Laptop/
│ └── ...
│
├── data/
│ ├── eng-Laptop/
│ │ ├── local/ # 本地 train/valid/test
│ │ └── official/ # 官方 train/dev/alltasks
│ ├── zho-Laptop/
│ └── ...
│
├── models/ # 模型结构定义
├── dataset/ # 数据读取 + 对齐 + 标签构建
├── utils/ # 通用工具
├── tools/ # 辅助脚本
│
├── output/
│ └── runs/ # 每次实验的完整输出（不进 Git）
│
├── sample_submission_files/ # 提交格式参考
└── requirements.txt

yaml
复制代码

---

## 3. 环境配置

### 3.1 创建环境
```bash
conda create -n asqp python=3.10 -y
conda activate asqp
pip install -r requirements.txt
3.2 GPU 说明
推荐 GPU

8GB 显存可运行

batch ≈ 6–8

fp16 开启

max_len = 256

4. 快速开始（以 zho-Laptop 为例）
4.1 训练（mDeBERTa）
bash
复制代码
python train.py \
  --config ./configs/zho-Laptop/data.yaml \
  --model_name ./mdeberta-v3-base \
  --output_dir ./output/runs/zho_laptop_mdeberta_cf1_v1 \
  --max_len 256 \
  --batch 6 \
  --epochs 80 \
  --lr 2e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.1 \
  --seed 42 \
  --fp16 \
  --label_pattern sentiment_dim \
  --neg_ratio 3.0 \
  --neg_shift 1 \
  --neg_max_per_sample 64 \
  --early_stop \
  --patience 12 \
  --min_epochs 10 \
  --min_delta 0.0005 \
  --select_by cf1
4.2 本地 valid 推理 + 评测
bash
复制代码
python predict.py \
  --config ./configs/zho-Laptop/data.yaml \
  --input ./data/zho-Laptop/local/valid.jsonl \
  --train_stats ./data/zho-Laptop/official/zho_laptop_train_alltasks.jsonl \
  --ckpt ./output/runs/zho_laptop_mdeberta_cf1_v1/best_model.pt \
  --model_name ./mdeberta-v3-base \
  --max_len 256 \
  --batch 8 \
  --label_pattern sentiment_dim \
  --thr_aux 0.05 \
  --topk_aux 80 \
  --max_span_len 12 \
  --thr_rel 0.06 \
  --topk_rel 220 \
  --max_pair_dist 80 \
  --max_quads 4 \
  --min_score 1.2 \
  --null_thr_o 0.08 \
  --va_stat median \
  --cat_case lower \
  --cat_source mix \
  --cat_head_min_conf 0.55 \
  --output ./output/runs/zho_laptop_mdeberta_cf1_v1/pred_valid.jsonl

python metrics_subtask_1_2_3.py \
  -t 3 \
  -g ./data/zho-Laptop/local/valid.jsonl \
  -p ./output/runs/zho_laptop_mdeberta_cf1_v1/pred_valid.jsonl
4.3 test 推理（可选本地评测）
bash
复制代码
python predict.py \
  --config ./configs/zho-Laptop/data.yaml \
  --input ./data/zho-Laptop/local/test.jsonl \
  --train_stats ./data/zho-Laptop/official/zho_laptop_train_alltasks.jsonl \
  --ckpt ./output/runs/zho_laptop_mdeberta_cf1_v1/best_model.pt \
  --model_name ./mdeberta-v3-base \
  --max_len 256 \
  --batch 8 \
  --label_pattern sentiment_dim \
  --output ./output/runs/zho_laptop_mdeberta_cf1_v1/pred_test.jsonl
5. 多语言实验规范（非常重要）
代码只维护一份

差异只存在于：

configs/<lang-domain>/

解码阈值

encoder 选择

推荐命名规范：

bash
复制代码
output/runs/
  zho_laptop_mdeberta_cf1_v1
  eng_restaurant_deberta_cf1_v2
6. 实验与复现
每一次 run 都会保存：

best_model.pt

train_config.json

best_metrics.json

pred_valid.jsonl

保证 任何一次结果都可回溯。

7. 备注
本仓库不包含官方数据文件

大文件（模型、输出、备份）均被 .gitignore 排除

提交文件请参考 sample_submission_files/

8. 致谢
One_ASQP baseline

SemEval-2026 Task 3 官方基线与评测脚本

DeBERTa / mDeBERTa 系列模型
