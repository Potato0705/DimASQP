# LLM 集成 · 方案 3 伪标注数据增强

本文档描述如何用 OpenRouter 上的 LLM 为 DimASQP 训练集生成伪标注，并把它合入训练数据做消融实验。

实现遵循 `docs/deep-research-report.md` 推荐的「方案 3」：**离线伪标注 + 训练前合并**。训练循环本身完全不变——仅把 `--train_data` 指向合并后的文件即可。

---

## 1. 先决条件

1. 已经运行过数据预处理（把 DimABSA 2026 原始 JSONL 转为内部 TXT/sidecar/gold）：
   ```bash
   python tools/convert_v2_all_languages.py
   ```
   这会在 `data/v2/{lang}/` 下为 8 个 task_domain × 3 个 split 生成 24 组文件。

2. 在 `OPENROUTER_API_KEY` 环境变量里设置你的 key：
   ```bash
   cp .env.example .env      # 然后编辑 .env 把 key 填进去
   set -a && source .env && set +a
   # 或直接：
   export OPENROUTER_API_KEY=sk-or-v1-...
   ```
   **绝不要把真实 key 写进任何提交文件**。`.env` 已在 `.gitignore` 中忽略。

3. 确认已安装 `requests`（已包含在 `requirements.txt` 中，无需新增依赖）。

---

## 2. 生成伪标注

基本调用（以 `eng_restaurant` 为主指标）：

```bash
python data/llm_pseudo_labeler.py \
    --task_domain eng_restaurant \
    --source_file data/v2/eng/eng_restaurant_train.jsonl \
    --llm_model_name meta-llama/llama-3.1-70b-instruct \
    --out_prefix data/v2/eng/eng_restaurant_train_pseudo__llama31-70b
```

产物：
- `..._pseudo__llama31-70b.txt`         —— 与 `eng_restaurant_train.txt` 同格式，可直接喂给 `train.py`
- `..._pseudo__llama31-70b.jsonl`       —— gold-style JSONL，便于检查
- `..._pseudo__llama31-70b_sidecar.json`
- `..._pseudo__llama31-70b_stats.json`  —— 调用次数、token 开销、丢弃率等

脚本会把每次 LLM 响应按 `sha256(prompt_version || model || sentence)` 缓存在 `cache/llm_pseudo/`（已 gitignore），重跑时直接命中不会重复扣费。

**防呆检查**：`--source_file` 路径里一旦出现 `_dev` / `_test` / `/dev.txt` / `/test.txt`，脚本会直接拒绝运行（除非加 `--i_really_mean_it_not_a_leak`，该开关仅供测试，禁止用于训练合并）。

小样本烟雾测试：
```bash
python data/llm_pseudo_labeler.py \
    --task_domain eng_restaurant \
    --source_file data/v2/eng/eng_restaurant_train.jsonl \
    --llm_model_name meta-llama/llama-3.1-70b-instruct \
    --out_prefix /tmp/pseudo_smoke \
    --nrows 10 --verbose
```

---

## 3. 合并到训练集

```bash
python data/merge_pseudo_with_gold.py \
    --gold   data/v2/eng/eng_restaurant_train.txt \
    --pseudo data/v2/eng/eng_restaurant_train_pseudo__llama31-70b.txt \
    --ratio  1.0 --seed 42 \
    --out    data/v2/eng/eng_restaurant_train__gold+pseudo_llama31-70b_r1.0.txt
```

`--ratio` 语义：
- `0.5` —— 采样一半伪标注行
- `1.0` —— 使用全部伪标注行（默认）
- `2.0` —— 通过有放回采样把伪标注行翻倍

输出的合并 `.txt` 连带 `_sidecar.json` 和 `.jsonl` 都会按行号对齐写出。

---

## 4. 训练（与基线共用同一条命令，只换 `--train_data`）

基线 3-seed 训练（来自 `AGENTS.md` 的固定超参数，**严禁改动**）：
```bash
# 替换 TRAIN_DATA 后，其他参数保持不变
TRAIN_DATA=data/v2/eng/eng_restaurant_train__gold+pseudo_llama31-70b_r1.0.txt

for SEED in 42 66 123; do
  python train.py \
    --task_domain eng_restaurant \
    --train_data  $TRAIN_DATA \
    --valid_data  data/v2/eng/eng_restaurant_dev.txt \
    --label_pattern category --use_efficient_global_pointer \
    --model_name_or_path microsoft/deberta-v3-base \
    --max_seq_len 128 --head_size 256 --mode mul \
    --dropout_rate 0.1 --mask_rate 0.0 \
    --epoch 200 --early_stop 20 \
    --per_gpu_train_batch_size 4 --gradient_accumulation_steps 8 \
    --use_amp --with_adversarial_training \
    --encoder_learning_rate 1e-5 --task_learning_rate 3e-5 \
    --max_grad_norm 1.0 \
    --weight1 1.0 --weight2 0.5 --weight3 0.5 --weight4 0.5 \
    --va_mode opinion_guided --weight_va_prior 0.3 \
    --seed $SEED
done
```

评测（threshold sweep + ensemble）沿用 `AGENTS.md` 步骤 1 / 2，不再赘述。

---

## 5. 推荐的消融网格

**阶段 A — 用低成本模型拟合「伪标注质量 × 数据量」曲线**（本文档主实施阶段）：

| 模型 | `--ratio` | seeds | 备注 |
|---|---|---|---|
| `meta-llama/llama-3.1-70b-instruct` | 0.5 | 42/66/123 | 开源最便宜 |
| `meta-llama/llama-3.1-70b-instruct` | 1.0 | 42/66/123 | |
| `meta-llama/llama-3.1-70b-instruct` | 2.0 | 42/66/123 | 有放回采样翻倍 |
| `openai/gpt-4o-mini` | 0.5 | 42/66/123 | 性价比平衡 |
| `openai/gpt-4o-mini` | 1.0 | 42/66/123 | |
| `openai/gpt-4o-mini` | 2.0 | 42/66/123 | |

共 6 组 × 3 seed = 18 次训练，对照基线（`AGENTS.md` 记录的 Span-Pair VA 3-seed avg cF1 = 0.5180 / Ensemble = 0.5370）填入结果表。

**阶段 B — 冲分**：只在阶段 A 曲线显示「LLM 质量」是瓶颈时，再切到 `anthropic/claude-3.5-sonnet` 跑同样的 0.5/1.0/2.0 三档。

---

## 6. 记录模板

| 模型 | Ratio | Seed | Best cF1 | Opt Threshold | cPrec | cRecall | VA% | 备注 |
|---|---|---|---|---|---|---|---|---|
| llama31-70b | 1.0 | 42 | | | | | | |
| llama31-70b | 1.0 | 66 | | | | | | |
| llama31-70b | 1.0 | 123 | | | | | | |
| llama31-70b | 1.0 | ensemble | | | | | | |
| gpt-4o-mini | 1.0 | 42 | | | | | | |
| ... | | | | | | | | |

---

## 7. 多语言 / 跨域扩展

`data/llm_pseudo_labeler.py` 已支持所有 8 个 task_domain（`eng_restaurant` / `eng_laptop` / `zho_restaurant` / `zho_laptop` / `jpn_hotel` / `rus_restaurant` / `tat_restaurant` / `ukr_restaurant`）——把 `--task_domain` 和 `--source_file` 换成对应语言的 gold JSONL 即可。伪标注不会跨域泄漏：每个 task_domain 的允许 category 表是独立从它自己的 gold train JSONL 里动态抽取的。

---

## 8. 安全性 & 额度注意事项

- **不要**把 `.env` 或硬编码 key 的文件提交到 git（`.env` 已忽略；`.env.example` 是占位）。
- LLM 响应磁盘缓存开启后，重跑同一句子不会重复扣费；但更换 `--llm_model_name` 或提升 `PROMPT_VERSION` 会重新调用。
- `--dry_run` 可以在无 key / 离线环境下仅从缓存产出结果，便于自测。
- 默认温度 `0.2`，已足够保证 JSON 输出稳定；如果遇到格式错误较多，可先降到 `0.0` 再排查 prompt。
