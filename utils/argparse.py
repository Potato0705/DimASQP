import argparse


def get_argparse():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--task_domain", required=True, type=str)
    parser.add_argument("--train_data", required=True, type=str)
    parser.add_argument("--valid_data", required=True, type=str)
    parser.add_argument("--nrows", default=None, type=int)
    parser.add_argument("--max_seq_len", default=512, type=int)
    parser.add_argument("--split_word", default="####", type=str)

    # model parameter
    parser.add_argument("--label_pattern", default="sentiment_dim", type=str)
    parser.add_argument("--use_efficient_global_pointer", action="store_true", default=False)
    parser.add_argument("--model_name_or_path", default="microsoft/deberta-v3-base", type=str)
    parser.add_argument("--head_size", default=256, type=int)
    parser.add_argument("--dropout_rate", default=0.1, type=float)
    parser.add_argument("--mode", default="mul", type=str)

    # train
    parser.add_argument("--mask_rate", default=0.0, type=float)
    parser.add_argument("--epoch", default=400, type=int)
    parser.add_argument("--weight1", required=True, type=float)
    parser.add_argument("--weight2", required=True, type=float)
    parser.add_argument("--weight3", required=True, type=float)
    parser.add_argument("--weight4", required=True, type=float)
    parser.add_argument("--early_stop", default=4, type=int)
    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int)
    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--do_distri_train", action="store_true")
    parser.add_argument("--with_adversarial_training", action="store_true")
    parser.add_argument("--encoder_learning_rate", default=2e-5, type=float)
    parser.add_argument("--task_learning_rate", default=5e-5, type=float)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm for clipping (0 to disable)")
    parser.add_argument("--va_mode", default="span_pair", type=str,
                        choices=["position", "span_pair", "opinion_guided"],
                        help="VA prediction: position (per-token), span_pair (conditioned on pair), opinion_guided (prior + residual)")
    parser.add_argument("--weight_va_prior", default=0.3, type=float,
                        help="Weight for opinion VA prior auxiliary loss (only used with opinion_guided)")
    parser.add_argument("--use_va_contrastive", action="store_true", default=False,
                        help="Enable VA-aware contrastive learning on span-pair representations")
    parser.add_argument("--weight_va_cl", default=0.1, type=float,
                        help="Weight for VA-aware contrastive loss (only used with --use_va_contrastive)")
    parser.add_argument("--seed", type=int, default=66)

    # output / model path
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--model_path", default=0, type=str)

    return parser


def get_predict_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nrows", default=None, type=int)
    parser.add_argument("--model_path", default=0, type=str)
    parser.add_argument("--test_data", required=True, type=str)
    parser.add_argument("--per_gpu_test_batch_size", default=32, type=int)
    parser.add_argument("--threshold", default=0.0, type=float,
                        help="Matrix logit threshold for predictions (default: 0.0)")
    return parser
