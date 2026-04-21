#!/bin/bash
# Phase 1 baseline: eng_restaurant with label_pattern=sentiment
# GPU: RTX 4060 Laptop 8GB -> batch_size=8, grad_accum=4 (effective=32)

python train.py \
    --task_domain eng_restaurant \
    --train_data data/eng/eng_restaurant_train.txt \
    --valid_data data/eng/eng_restaurant_dev.txt \
    --max_seq_len 128 \
    --label_pattern sentiment \
    --use_efficient_global_pointer \
    --model_name_or_path microsoft/deberta-v3-base \
    --head_size 256 \
    --dropout_rate 0.1 \
    --mode mul \
    --mask_rate 0.3 \
    --epoch 100 \
    --weight1 1.0 \
    --weight2 0.5 \
    --weight3 0.5 \
    --weight4 0.0 \
    --early_stop 10 \
    --per_gpu_train_batch_size 8 \
    --per_gpu_eval_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --use_amp \
    --with_adversarial_training \
    --encoder_learning_rate 1e-5 \
    --task_learning_rate 3e-5 \
    --seed 42
