# =============================================================
# 全量实验串行脚本 — 共 12 runs
# eng_restaurant baseline(补跑) + eng_laptop 全条件
# =============================================================
# 用法: powershell -ExecutionPolicy Bypass -File scripts/run_all_experiments.ps1
# 预计总耗时: ~36-48 小时（每 run ~3-4h）

$common = @(
    "--label_pattern", "category",
    "--use_efficient_global_pointer",
    "--model_name_or_path", "microsoft/deberta-v3-base",
    "--max_seq_len", "128",
    "--head_size", "256",
    "--mode", "mul",
    "--dropout_rate", "0.1",
    "--mask_rate", "0.0",
    "--epoch", "200",
    "--early_stop", "20",
    "--per_gpu_train_batch_size", "4",
    "--gradient_accumulation_steps", "8",
    "--use_amp",
    "--with_adversarial_training",
    "--encoder_learning_rate", "1e-5",
    "--task_learning_rate", "3e-5",
    "--max_grad_norm", "1.0",
    "--weight1", "1.0",
    "--weight2", "0.5",
    "--weight3", "0.5",
    "--weight4", "0.5"
)

$experiments = @(
    # --- Group 1: eng_restaurant baseline (position VA) x3 ---
    @{ domain="eng_restaurant"; train="data/eng/eng_restaurant_train.txt"; valid="data/eng/eng_restaurant_dev.txt"; va_mode="position"; extra=@(); seed=42 },
    @{ domain="eng_restaurant"; train="data/eng/eng_restaurant_train.txt"; valid="data/eng/eng_restaurant_dev.txt"; va_mode="position"; extra=@(); seed=66 },
    @{ domain="eng_restaurant"; train="data/eng/eng_restaurant_train.txt"; valid="data/eng/eng_restaurant_dev.txt"; va_mode="position"; extra=@(); seed=123 },

    # --- Group 2: eng_laptop baseline (position VA) x3 ---
    @{ domain="eng_laptop"; train="data/eng/eng_laptop_train.txt"; valid="data/eng/eng_laptop_dev.txt"; va_mode="position"; extra=@(); seed=42 },
    @{ domain="eng_laptop"; train="data/eng/eng_laptop_train.txt"; valid="data/eng/eng_laptop_dev.txt"; va_mode="position"; extra=@(); seed=66 },
    @{ domain="eng_laptop"; train="data/eng/eng_laptop_train.txt"; valid="data/eng/eng_laptop_dev.txt"; va_mode="position"; extra=@(); seed=123 },

    # --- Group 3: eng_laptop span_pair VA x3 ---
    @{ domain="eng_laptop"; train="data/eng/eng_laptop_train.txt"; valid="data/eng/eng_laptop_dev.txt"; va_mode="span_pair"; extra=@(); seed=42 },
    @{ domain="eng_laptop"; train="data/eng/eng_laptop_train.txt"; valid="data/eng/eng_laptop_dev.txt"; va_mode="span_pair"; extra=@(); seed=66 },
    @{ domain="eng_laptop"; train="data/eng/eng_laptop_train.txt"; valid="data/eng/eng_laptop_dev.txt"; va_mode="span_pair"; extra=@(); seed=123 },

    # --- Group 4: eng_laptop opinion_guided VA x3 ---
    @{ domain="eng_laptop"; train="data/eng/eng_laptop_train.txt"; valid="data/eng/eng_laptop_dev.txt"; va_mode="opinion_guided"; extra=@("--weight_va_prior", "0.3"); seed=42 },
    @{ domain="eng_laptop"; train="data/eng/eng_laptop_train.txt"; valid="data/eng/eng_laptop_dev.txt"; va_mode="opinion_guided"; extra=@("--weight_va_prior", "0.3"); seed=66 },
    @{ domain="eng_laptop"; train="data/eng/eng_laptop_train.txt"; valid="data/eng/eng_laptop_dev.txt"; va_mode="opinion_guided"; extra=@("--weight_va_prior", "0.3"); seed=123 }
)

$total = $experiments.Count
for ($i = 0; $i -lt $total; $i++) {
    $exp = $experiments[$i]
    $n = $i + 1
    Write-Host "`n============================================================="
    Write-Host "[$n/$total] $($exp.domain) | va_mode=$($exp.va_mode) | seed=$($exp.seed)"
    Write-Host "=============================================================`n"

    $args_list = @(
        "--task_domain", $exp.domain,
        "--train_data", $exp.train,
        "--valid_data", $exp.valid
    ) + $common + @(
        "--va_mode", $exp.va_mode
    ) + $exp.extra + @(
        "--seed", $exp.seed
    )

    python train.py @args_list

    if ($LASTEXITCODE -ne 0) {
        Write-Host "`n[ERROR] Run $n failed (exit code $LASTEXITCODE). Stopping."
        exit 1
    }

    Write-Host "`n[OK] Run $n completed.`n"
}

Write-Host "`n============================================================="
Write-Host "All $total experiments completed!"
Write-Host "============================================================="
