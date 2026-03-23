Paper Code

1. dataset is in './data' 

2. configs is in './congis

3. Running Example
```
python3  train.py
--task_domain=Laptop-ACOS
--mode=mul
--per_gpu_train_batch_size=32
--label_pattern=sentiment
--max_seq_len=128
--early_stop=5
--mask_rate=0.6
--head_size=256
--weight1=0.3
--weight2=0
--weight3=0.6
--weight4=0.3
--encoder_learning_rate=1e-05
--task_learning_rate=3e-05
--use_efficient_global_pointer
--train_data=data/Laptop-ACOS/train.txt
--valid_data=data/Laptop-ACOS/dev.txt
--use_amp
--output_dir=/data/junius/quadruple_sentiment/
--model_name_or_path=microsoft/deberta-v3-large
```