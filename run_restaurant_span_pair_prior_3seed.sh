#!/bin/bash
set -euo pipefail

ROOT=/data-ai/sl20250133/Wuyaoshuo/DimASQP
ENV_PREFIX=/data-ai/sl20250133/Wuyaoshuo/.conda-envs/dimasqp

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PIP_CACHE_DIR=/data-ai/sl20250133/Wuyaoshuo/.cache/pip
export XDG_CACHE_HOME=/data-ai/sl20250133/Wuyaoshuo/.cache
export HF_HOME=/data-ai/sl20250133/Wuyaoshuo/.cache/huggingface
export TRANSFORMERS_CACHE=/data-ai/sl20250133/Wuyaoshuo/.cache/huggingface/transformers
export TORCH_HOME=/data-ai/sl20250133/Wuyaoshuo/.cache/torch
export TMPDIR=/data-ai/sl20250133/Wuyaoshuo/tmp

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX"
cd "$ROOT"

run_seed() {
  local seed="$1"
  echo "==== Restaurant | Span-Pair + Prior Loss | seed=${seed} ===="
  python train.py \
    --task_domain eng_restaurant \
    --train_data data/eng/eng_restaurant_train.txt \
    --valid_data data/eng/eng_restaurant_dev.txt \
    --label_pattern category \
    --use_efficient_global_pointer \
    --model_name_or_path microsoft/deberta-v3-base \
    --max_seq_len 128 \
    --head_size 256 \
    --mode mul \
    --dropout_rate 0.1 \
    --mask_rate 0.0 \
    --epoch 200 \
    --early_stop 20 \
    --per_gpu_train_batch_size 4 \
    --per_gpu_eval_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --num_workers 0 \
    --with_adversarial_training \
    --encoder_learning_rate 1e-5 \
    --task_learning_rate 3e-5 \
    --weight_decay 0.01 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --weight1 1.0 \
    --weight2 0.5 \
    --weight3 0.5 \
    --weight4 0.5 \
    --va_mode span_pair \
    --use_va_prior_aux \
    --weight_va_prior 0.3 \
    --seed "$seed"
}

run_seed 42
run_seed 66
run_seed 123
