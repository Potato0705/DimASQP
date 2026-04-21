# DimASQP

`master` 是当前仓库保留下来的旧论文代码线，对应论文文件 `paper/main.pdf`。这条分支以复现和留档为主，不再继续承接新的 LLM 增强实验。

## 分支定位

- `master`：冻结的论文代码线
- `main`：当前维护中的整合主线
- `Add-llm`：LLM / CCA / ISR 等实验分支

## 本分支包含的核心内容

- 论文源码与 PDF：`paper/main.tex`、`paper/main.pdf`
- 训练入口：`train.py`
- 推理入口：`predict.py`
- 基线与分析工具：`tools/`
- 原论文阶段的数据与配置：`data/`、`configs/`

## 使用建议

如果你的目标是：

1. 复现旧论文中的实验或查看投稿材料，请使用 `master`
2. 查看当前整合后的最新代码，请使用 `main`
3. 继续做 LLM 增强或数据扩增实验，请使用 `Add-llm`

## 快速开始

```bash
pip install -r requirements.txt

python train.py \
  --task_domain Laptop-ACOS \
  --train_data data/Laptop-ACOS/train.txt \
  --valid_data data/Laptop-ACOS/dev.txt
```

更完整的实验背景可参考 `docs/project_plan_v2.md` 与 `paper/` 目录。
