# DimASQP

`Add-llm` 是本仓库的 LLM 增强实验分支，主要承载伪标注、CCA（Compositional Category Augmentation）、ISR-Recovery 和概率化 VA 头等探索性工作。已经验证有效的阶段性成果会再合并回 `main`，而 `master` 保持为旧论文代码线。

## 分支定位

- `Add-llm`：LLM 与数据增强实验线
- `main`：当前整合主线，吸收已验证的里程碑改动
- `master`：冻结的论文代码线，对应 `paper/main.pdf`

## 本分支重点内容

- 离线伪标注流程：`data/llm_pseudo_labeler.py`
- CCA 数据增强：`data/cca_generator.py`
- ISR 数据生成：`data/isr_generator.py`
- LLM 客户端与提示词：`llm/`
- LLM 集成说明：`docs/llm_integration.md`
- 已保留的实验结果快照：`output/laptop_restaurant_all_2026-04-13/`

## 典型使用流程

```bash
pip install -r requirements.txt
python tools/convert_v2_all_languages.py
```

准备好 `OPENROUTER_API_KEY` 后，可以按 `docs/llm_integration.md` 中的流程：

1. 生成伪标注或 CCA 增强数据
2. 用 `data/merge_pseudo_with_gold.py` 合并 gold 与增强样本
3. 把 `train.py` 的 `--train_data` 指向合并后的文件继续训练

## 说明

- 这个分支允许保留实验脚本、分析工具和阶段性结果产物。
- 只有经过验证的结果才建议继续同步回 `main`。
