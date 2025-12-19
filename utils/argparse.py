"""
@Time : 2022/12/1721:32
@Auth : zhoujx
@File ：finetuning_argparse.py
@DESCRIPTION:

"""
import argparse


def get_argparse():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--task_domain", required=True, type=str, help="")
    parser.add_argument("--train_data", required=True, type=str, help="")
    parser.add_argument("--valid_data", required=True, type=str, help="")
    parser.add_argument("--nrows", default=None, type=int, help="")
    parser.add_argument("--max_seq_len", default=512, type=int, help="最大长度")
    parser.add_argument("--split_word", default='####', type=str, help="切分数据集的方法")

    # model parameter
    parser.add_argument("--label_pattern", default="sentiment_dim", type=str, help="")
    parser.add_argument("--use_efficient_global_pointer", action="store_true", default=False, help="")
    parser.add_argument("--model_name_or_path",
                        default="/data/zhoujx/prev_trained_model/chinese_roberta_wwm_ext_pytorch", type=str,
                        help="预训练模型的路径")
    parser.add_argument("--head_size", default=256, type=int, help="任务层的维度")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="dropout rate")
    parser.add_argument("--mode", default='mul', type=str, help="乘性、加性、双仿射")

    # train
    parser.add_argument("--mask_rate", default=0, type=float, help="训练轮数")
    parser.add_argument("--epoch", default=400, type=int, help="训练的epoch")
    parser.add_argument("--weight1", required=True, type=float, help="主损失的权重")
    parser.add_argument("--weight2", required=True, type=float, help="辅助损失的权重")
    parser.add_argument("--weight3", required=True, type=float, help="维度序列的权重")
    parser.add_argument("--weight4", required=True, type=float, help="情感序列的权重")
    parser.add_argument("--early_stop", default=4, type=int, help="早停")
    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int, help="训练Batch size的大小")
    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int, help="验证Batch size的大小")
    parser.add_argument("--use_amp", action="store_true", help="")
    parser.add_argument("--do_distri_train", action="store_true", help="是否用两个卡并行训练")
    parser.add_argument("--with_adversarial_training", action="store_true", help="")
    parser.add_argument("--encoder_learning_rate", default=2e-5, type=float, help="transformer层学习率")
    parser.add_argument("--task_learning_rate", default=5e-5, type=float, help="linear层学习率")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--seed", type=int, default=66, help="random seed for initialization")

    # model
    parser.add_argument("--output_dir", default="./output", type=str, help="保存模型的路径")

    # predict
    parser.add_argument("--model_path", default=0, type=str, help="模型保存路径")

    return parser


def get_predict_argparse():
    parser = argparse.ArgumentParser()

    # predict
    parser.add_argument("--nrows", default=None, type=int, help="")
    parser.add_argument("--model_path", default=0, type=str, help="模型保存路径")
    parser.add_argument("--test_data", required=True, type=str, help="")
    parser.add_argument("--per_gpu_test_batch_size", default=32, type=int, help="验证Batch size的大小")

    return parser
