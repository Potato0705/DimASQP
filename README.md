# DimASQP

`main` 是当前仓库的整合主线。它保留了 `master` 的论文代码基础，并持续吸收 `Add-llm` 中已经验证过的 LLM 增强、数据扩增和实验工具改动。

## 分支说明

- `main`：当前维护中的主线分支
- `master`：冻结的旧论文代码线，对应 `paper/main.pdf`
- `Add-llm`：LLM / CCA / ISR 等实验分支，阶段性成果再合回 `main`

## 当前分支包含

- 基线训练与推理代码：`train.py`、`predict.py`
- 模型与损失实现：`models/`、`losses/`
- 传统数据目录与配置：`data/`、`configs/`
- v2 多语言数据与增强流程：`data/v2/`、`llm/`
- LLM 集成说明：`docs/llm_integration.md`
- 论文与附属材料：`paper/`
- 实验结果快照：`output/laptop_restaurant_all_2026-04-13/`

## 快速开始

```bash
pip install -r requirements.txt

python train.py \
  --task_domain eng_restaurant \
  --train_data data/v2/eng/eng_restaurant_train.txt \
  --valid_data data/v2/eng/eng_restaurant_dev.txt \
  --label_pattern category \
  --use_efficient_global_pointer \
  --model_name_or_path microsoft/deberta-v3-base
```

如果你想继续做 LLM 增强实验，请优先阅读 `docs/llm_integration.md`；如果你想复现旧论文结果，请切到 `master`。
