"""
@Time : 2022/12/1720:27
@Auth : zhoujx
@File ：models.py
@DESCRIPTION:

"""
import torch.nn
import torch.nn.functional as F
from torch.nn import Module
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from dataset.dataset import AcqpDataset, collate_fn
from models.layers import EfficientGlobalPointer, GlobalPointer


class QuadrupleModel(Module):
    """
    新版三元组模型(四元组)
    """

    def __init__(self,
                 num_label_types,
                 num_dimension_types,
                 max_seq_len,
                 pretrain_model_path,
                 with_adversarial_training,
                 use_efficient_global_pointer,
                 mode='multiply',
                 head_size=64,
                 matrix_hidden_size=400,
                 dimension_hidden_size=400,
                 dimension_sequence_hidden_size=400,
                 sentiment_sequence_hidden_size=400,
                 dropout_rate=0.1,
                 RoPE=True):
        super(QuadrupleModel, self).__init__()
        self.num_label_types = num_label_types
        self.num_dimension_types = num_dimension_types
        self.num_sentiment_types = 3
        self.head_size = head_size
        self.mode = mode
        self.matrix_hidden_size = matrix_hidden_size,
        self.dimension_hidden_size = dimension_hidden_size,
        self.dimension_sequence_hidden_size = dimension_sequence_hidden_size,
        self.sentiment_sequence_hidden_size = sentiment_sequence_hidden_size,
        self.dropout_rate = dropout_rate
        self.RoPE = RoPE
        self.mode = mode
        self.max_seq_len = max_seq_len
        self.pretrain_model_path = pretrain_model_path
        self.encoder = AutoModel.from_pretrained(self.pretrain_model_path)
        self.encoder_hidden_size = self.encoder.config.hidden_size
        self.with_adversarial_training = with_adversarial_training
        self.use_efficient_global_pointer = use_efficient_global_pointer  # True
        self.dropout_layer = torch.nn.Dropout(p=self.dropout_rate, )

        #
        self.matrix_linear = torch.nn.Linear(in_features=self.encoder_hidden_size, out_features=matrix_hidden_size)
        if self.mode == 'mul' and self.use_efficient_global_pointer:
            self.global_pointer_layer = EfficientGlobalPointer(hidden_size=matrix_hidden_size,
                                                               heads=self.num_label_types,
                                                               head_size=self.head_size,
                                                               RoPE=self.RoPE,
                                                               use_bias=True,
                                                               tril_mask=False,
                                                               max_length=self.max_seq_len)
        elif self.mode == 'mul' and not self.use_efficient_global_pointer:
            self.global_pointer_layer = GlobalPointer(hidden_size=matrix_hidden_size,
                                                      heads=self.num_label_types,
                                                      head_size=self.head_size,
                                                      RoPE=self.RoPE,
                                                      use_bias=True,
                                                      tril_mask=False,
                                                      max_length=self.max_seq_len)
        elif self.mode == 'add':
            self.matrix_linear = torch.nn.Linear(in_features=self.encoder_hidden_size * 2,
                                                 out_features=self.head_size)
            self.matrix_output = torch.nn.Linear(in_features=self.head_size,
                                                 out_features=self.num_label_types)
        elif self.mode == 'biaffine':
            self.linear_layer_start = torch.nn.Linear(in_features=self.encoder_hidden_size,
                                                      out_features=self.head_size)
            self.linear_layer_end = torch.nn.Linear(in_features=self.encoder_hidden_size,
                                                    out_features=self.head_size)
            self.U = torch.nn.Parameter(
                torch.nn.init.xavier_uniform_(torch.randn([self.head_size + 1, self.num_label_types, self.head_size + 1])),
                requires_grad=True)

        #
        self.dimension_linear = torch.nn.Linear(in_features=self.encoder_hidden_size,
                                                out_features=dimension_hidden_size)
        self.dimension_output = torch.nn.Linear(in_features=dimension_hidden_size,
                                                out_features=self.num_dimension_types)

        # 维度序列
        self.dimension_sequence_linear = torch.nn.Linear(in_features=self.encoder_hidden_size,
                                                         out_features=dimension_sequence_hidden_size)
        self.dimension_sequence_output = torch.nn.Linear(in_features=dimension_sequence_hidden_size,
                                                         out_features=self.num_dimension_types)

        # 情感序列
        self.sentiment_sequence_linear = torch.nn.Linear(in_features=self.encoder_hidden_size,
                                                         out_features=sentiment_sequence_hidden_size)
        self.sentiment_sequence_output = torch.nn.Linear(in_features=sentiment_sequence_hidden_size,
                                                         out_features=self.num_sentiment_types)

        # VA回归头: 每个token位置预测 [V, A], 通过 sigmoid*8+1 映射到 [1, 9]
        va_hidden_size = 256
        self.va_linear = torch.nn.Linear(in_features=self.encoder_hidden_size, out_features=va_hidden_size)
        self.va_output = torch.nn.Linear(in_features=va_hidden_size, out_features=2)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None):

        outputs = self.encoder(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids)
        sequence_output = outputs.last_hidden_state

        cls_output = sequence_output[:, 0, :]
        # cls_output = F.relu(cls_output)
        # cls_output = self.dropout_layer(cls_output)
        cls_dim_output = self.dimension_linear(cls_output)
        cls_dim_output = F.relu(cls_dim_output)
        cls_dim_output = self.dropout_layer(cls_dim_output)
        cls_dim_output = self.dimension_output(cls_dim_output)

        # multiply
        if self.mode == 'mul':
            matrix_output = self.matrix_linear(sequence_output)
            matrix_output = F.relu(matrix_output)
            matrix_output = self.dropout_layer(matrix_output)
            matrix_output = self.global_pointer_layer(matrix_output, attention_mask=attention_mask)  # [B, 3, L, L]
        # add
        elif self.mode == 'add':
            start_extend = torch.unsqueeze(sequence_output, 2)  # [B, L, 1, H]
            start_extend = start_extend.expand(-1, -1, self.max_seq_len, -1)
            end_extend = torch.unsqueeze(sequence_output, 1)  # [B, 1, L, H]
            end_extend = end_extend.expand(-1, self.max_seq_len, -1, -1)
            span_matrix = torch.cat([start_extend, end_extend], 3)  # [B, L, L, 2*H]
            matrix_output = self.matrix_linear(span_matrix)
            matrix_output = F.relu(matrix_output)
            matrix_output = self.dropout_layer(matrix_output)
            matrix_output = self.matrix_output(matrix_output)
            matrix_output = torch.transpose(matrix_output, 1, 3)
            matrix_output = torch.transpose(matrix_output, 2, 3)
        # biaffine
        elif self.mode == 'biaffine':
            start = self.linear_layer_start(sequence_output)
            end = self.linear_layer_end(sequence_output)
            start = torch.cat((start, torch.ones_like(start[..., :1])), dim=-1)  # [B, L, Hide_size+1]
            end = torch.cat((end, torch.ones_like(end[..., :1])), dim=-1)  # [B, L, Hide_size+1]
            matrix_output = torch.einsum('bxi,ioj,byj->bxyo', start, self.U, end)
            matrix_output = torch.transpose(matrix_output, 1, 3)
            matrix_output = torch.transpose(matrix_output, 2, 3)

        # 维度序列
        dim_seq_output = self.dimension_sequence_linear(sequence_output)
        dim_seq_output = F.relu(dim_seq_output)
        dim_seq_output = self.dropout_layer(dim_seq_output)
        dim_seq_output = self.dimension_sequence_output(dim_seq_output)
        dim_seq_output = dim_seq_output.transpose(1, 2)

        # 情感序列
        sen_seq_output = self.sentiment_sequence_linear(sequence_output)
        sen_seq_output = F.relu(sen_seq_output)
        sen_seq_output = self.dropout_layer(sen_seq_output)
        sen_seq_output = self.sentiment_sequence_output(sen_seq_output)
        sen_seq_output = sen_seq_output.transpose(1, 2)

        # VA回归: [B, L, 2] -> sigmoid*8+1 -> [1, 9]
        va_output = self.va_linear(sequence_output)
        va_output = F.relu(va_output)
        va_output = self.dropout_layer(va_output)
        va_output = self.va_output(va_output)
        va_output = torch.sigmoid(va_output) * 8.0 + 1.0  # map to [1, 9]

        return {"matrix": matrix_output,
                "dimension": cls_dim_output,
                "dimension_sequence": dim_seq_output,
                "sentiment_sequence": sen_seq_output,
                "va": va_output}


if __name__ == '__main__':
    model = QuadrupleModel(num_label_types=40,
                           num_dimension_types=40,
                           max_seq_len=128,
                           pretrain_model_path="microsoft/deberta-v3-base",
                           with_adversarial_training=True,
                           use_efficient_global_pointer=True, )
    tokenizer = AutoTokenizer.from_pretrained("/data/transformers_pretrain_models/transformers_tf_en_deberta-v3-base/")
    dataset = AcqpDataset(task_domain="天美游戏",
                          tokenizer=tokenizer,
                          data_path="../data/游戏三元组标注数据_20221206.csv",
                          max_seq_len=128,
                          nrows=2000)
    dataloader = DataLoader(dataset,
                            batch_size=32,
                            shuffle=True,
                            num_workers=16,
                            drop_last=False,
                            collate_fn=collate_fn)
    data = next(iter(dataloader))
    input_ids = data["all_input_ids"]
    token_type_ids = data["all_token_type_ids"]
    attention_mask = data["all_attention_mask"]
    model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
