from dataset.dataset import AcqpDataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./deberta-v3-base")
ds = AcqpDataset("L", "./data/eng_laptop_train_alltasks.jsonl", 256, tokenizer, label_pattern="sentiment_dim")
print(ds.label_types)
