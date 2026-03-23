"""
@Time : 2022/12/1717:18
@Auth : zhoujx
@File ：dataset.py
@DESCRIPTION:

"""
import json
import os
from typing import List

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

CURRENT_DIR = os.path.dirname(__file__)
from utils.utils import smart_read_csv, read_data_from_txt

try:
    import MySQLdb
except Exception:
    try:
        import pymysql as MySQLdb
    except Exception:
        MySQLdb = None  # MySQL not needed for TXT-based pipeline
try:
    import configparser
except Exception:
    configparser = None


class AcqpDataset_bak(Dataset):
    def __init__(self, task_domain, data_path, max_seq_len, tokenizer, label_pattern="sentiment_dim", **kwargs):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.task_domain = task_domain
        self.dimension_types = self.get_label_list(self.task_domain)
        self.num_dimension_types = len(self.dimension_types)
        self.entity_types = ['特征词', '情感词', '独立特征词']
        self.sentiment_types = ['正面', '负面', '中性']
        self.label_pattern = label_pattern
        self.label_types = self.get_label_types()
        self.num_label_types = len(self.label_types)
        self.df_raw = smart_read_csv(data_path, **kwargs)
        self.df = self.get_df()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        tokens = self.df.loc[index, "Tokens"]
        raw_label = self.df.loc[index, "Labels"]

        tokens = ['[SEP]'] + tokens
        encoding = self.tokenizer(tokens,
                                  add_special_tokens=True,
                                  max_length=self.max_seq_len,
                                  truncation=True,
                                  padding='max_length',
                                  return_token_type_ids=True,
                                  return_attention_mask=True,
                                  is_split_into_words=True)
        input_ids = encoding["input_ids"]
        token_type_ids = encoding["token_type_ids"]
        attention_mask = encoding["attention_mask"]

        matrix_ids = torch.zeros([self.num_label_types, self.max_seq_len, self.max_seq_len], dtype=torch.int32)
        dimension_ids = torch.zeros(self.num_dimension_types, dtype=torch.int32)
        for label in raw_label:
            if label["first_type"] == "情感词" and "second_type" in label and label["second_type"] == "特征词":
                BO = label["first_start"]
                EO = label["first_end"]
                BA = label["second_start"]
                EA = label["second_end"]
            elif label["first_type"] == "特征词" and "second_type" in label and label["second_type"] == "情感词":
                BO = label["second_start"]
                EO = label["second_end"]
                BA = label["first_start"]
                EA = label["first_end"]
            else:
                BO = None
                EO = None
                BA = label["first_start"]
                EA = label["first_end"]
            dim = label["dimension"]
            sentiment = label["sentiment"]
            dim_sentiment_label = f"{dim}_{sentiment}"
            if self.label_pattern == "raw":
                query = "BA-EO"
            elif self.label_pattern == "sentiment":
                query = sentiment
            elif self.label_pattern == "sentiment_dim":
                query = dim_sentiment_label

            dimension_ids[self.dimension_types.index(dim)] = 1

            if EO is not None:
                matrix_ids[self.label_types.index(query), BA + 2, EO + 1] = 1  # BA-EO
                matrix_ids[self.label_types.index("BA-BO"), BA + 2, BO + 2] = 1  # BA-BO
                matrix_ids[self.label_types.index("EA-EO"), EA + 1, EO + 1] = 1  # EA-EO
            else:
                matrix_ids[self.label_types.index(query), BA + 2, 1] = 1  # BA-EO
                matrix_ids[self.label_types.index("EA-EO"), EA + 1, 1] = 1  # EA-EO

        return {"input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask,
                "matrix_ids": matrix_ids,
                "dimension_ids": dimension_ids}

    @staticmethod
    def token_index_map_char_index(token_level_label_index: List[str]):
        try:
            labels = list(map(lambda x: list(map(int, x.split(','))), token_level_label_index))
            labels.append([labels[-1][-1] + 1])
            labels = {k: v for k, v in enumerate(labels)}
            return labels
        except:
            print(token_level_label_index)

    @staticmethod
    def replace_relations(json_dict):
        """
        映射新版的情感标注适配老版本
        """
        relations = json_dict.get('relations', None)
        if not relations:
            return json_dict
        new_relations = []
        for rdict in relations:
            if rdict.get('relation_type', None) == '情感':
                rdict['relation_type'] = '情感极性'
            if rdict.get('relation_value', None) == '中':
                rdict['relation_value'] = '中性'
            elif rdict.get('relation_value', None) == '正':
                rdict['relation_value'] = '正面'
            elif rdict.get('relation_value', None) == '负':
                rdict['relation_value'] = '负面'
            new_relations.append(rdict)
        json_dict['relations'] = new_relations
        return json_dict

    @staticmethod
    def match_index(char_index: int, token_char_index_dict: dict):
        """
        找出字符分词后对应的token index
        :param char_index:
        :param token_char_index_dict:
        :return:
        """
        for k, v in token_char_index_dict.items():
            if char_index in v:
                return k
        print(
            '[Error]: can not match char_index in token_char_index_dict!!!\n This may be due to the fact that the entity is segmented into two sub sentences.')
        print('[Error]: char_index-->%s\t\n token_char_index_dict-->%s\t\t\t\n' % (char_index, token_char_index_dict))
        return None

    def get_label_types(self):
        label_types = ["BA-BO", "EA-EO"]
        if self.label_pattern == "raw":
            label_types.extend(["BA-EO"])
        if self.label_pattern == "sentiment":
            label_types.extend(["中性", "正面", "负面"])
        if self.label_pattern == "sentiment_dim":
            for dimension_type in self.dimension_types:
                for sentiment_type in self.sentiment_types:
                    label_types.append(f"{dimension_type}_{sentiment_type}")
        return label_types

    def get_label_list(self, *args, **kwargs):
        config_info = configparser.ConfigParser()
        config_info.read(os.path.join(CURRENT_DIR, '../configs/mysql.ini'), encoding="utf-8")
        db_connect = MySQLdb.connect(passwd=config_info['DEFAULT']['passwd'],
                                     db=config_info['DEFAULT']['db'],
                                     host=config_info['DEFAULT']['host'],
                                     port=int(config_info['DEFAULT']['port']),
                                     user=config_info['DEFAULT']['user'],
                                     charset='utf8')
        db_cursor = db_connect.cursor()
        labels_list = self.get_output_labels(db_cursor=db_cursor,
                                             # task_name='天美游戏',
                                             )
        db_connect.close()
        return labels_list

    def get_triple_labels(self, db_cursor):
        """
        获取三元组标签, 不包含情感极性
        here "X" used to represent "##eer","##soo" and so on!
        "[PAD]" for padding
        :return:
        """
        labels = []
        sentiments = []
        db_cursor.execute("select label from t_triple_smart where domain='%s' order by map_id;" % self.task_domain)
        for i in db_cursor.fetchall():
            if len(i[0]) < 7:
                labels.append(i[0])
                continue
            split_label = i[0][1:].split('-')
            sentiments.append(split_label[1])
            label1 = split_label[0]
            label2 = split_label[0]
            label3 = split_label[0]
            if label1 not in labels:
                labels.append(label1)
            if label2 not in labels:
                labels.append(label2)
            if label3 not in labels:
                labels.append(label3)
        self.is_contains_neutral = True if '中' in sentiments or '中性' in sentiments else False
        unused_label = ['[PAD]', 'X', '[CLS]', '[SEP]', 'O', ]
        for label in unused_label:
            labels.remove(label)
        return labels

    def get_output_labels(self, db_cursor, task_name=None):
        """
        获取三元组标签, 如果 task_name==triple 则返回包含情感极性的标签
        """
        if task_name == 'triple':
            labels = []
            db_cursor.execute("select label from t_triple_smart where domain='%s' order by map_id;" % self.task_domain)
            for i in db_cursor.fetchall():
                labels.append(i[0])
            return labels
        else:
            return self.get_triple_labels(db_cursor)

    def get_df(self):
        encoding = self.tokenizer(self.df_raw["内容"].tolist(),
                                  add_special_tokens=False,
                                  return_offsets_mapping=True,
                                  return_overflowing_tokens=True,
                                  max_length=self.max_seq_len - 3)
        results = []
        for i in range(len(encoding["input_ids"])):
            num_tokens = len(encoding["input_ids"][i])
            dic_tmp = {}
            dic_tmp["Tokens"] = encoding.tokens(i)
            overflow_to_sample_mapping = encoding["overflow_to_sample_mapping"][i]
            dic_tmp["Text_Id"] = self.df_raw.iloc[overflow_to_sample_mapping, self.df_raw.columns.get_loc("Text_Id")]
            dic_tmp["Seq"] = self.tokenizer.decode(encoding["input_ids"][i])
            dic_tmp["Seq_Len"] = encoding[i].token_to_chars(num_tokens - 1)[1] - encoding[i].token_to_chars(0)[0]
            New_Labels = []
            last_end, _ = encoding[i].token_to_chars(0)
            for j in range(num_tokens):
                start, end = encoding[i].token_to_chars(j)
                # text:"How old are you …?"
                if last_end == end:
                    last_end = start
                # if last_end == end:
                #     end = last_end + 1
                new_label = list(range(last_end, end))
                new_label = list(map(str, new_label))
                new_label = ",".join(new_label)
                New_Labels.append(new_label)
                last_end = end
            dic_tmp["New_Labels"] = New_Labels
            results.append(dic_tmp)

        df = pd.DataFrame(results)
        df["Seq_Id"] = df.groupby("Text_Id").cumcount()

        print('[Info]: map token char...')
        use_raw_cols = ['内容', '答案', 'Text_Id']
        df = pd.merge(self.df_raw[use_raw_cols], df, on='Text_Id', how='right')
        df['Seq_Lens'] = df['Seq_Len'].apply(lambda x: [x])
        df['Seq_Lens'] = df.groupby('Text_Id')['Seq_Lens'].transform('sum')
        df['Seq_Start_Index'] = df.apply(lambda row: np.sum(
            np.tril(np.ones([len(row['Seq_Lens']) + 1, len(row['Seq_Lens']) + 1]))[:-1, 1:][row['Seq_Id']] * row[
                'Seq_Lens']), axis=1).astype(int)
        df['Token_Len'] = df['Tokens'].str.len()
        df['Token_Lens'] = df['Token_Len'].apply(lambda x: [x])
        df['Token_Lens'] = df.groupby('Text_Id')['Token_Lens'].transform('sum')
        df['Token_Start_Index'] = df.apply(lambda row: np.sum(
            np.tril(np.ones([len(row['Token_Lens']) + 1, len(row['Token_Lens']) + 1]))[:-1, 1:][row['Seq_Id']] * row[
                'Token_Lens']), axis=1).astype(int)
        df['Token_Index_Map_Char_Index'] = df['New_Labels'].apply(self.token_index_map_char_index)

        if '答案' in df.columns:
            print('[Info]: convert_char_index_to_token_index...')
            df['Labels'] = df.apply(lambda row: self.convert_raw_label(row["内容"], row["答案"]), axis=1)
            df['Labels'] = df.apply(lambda row: self.convert_char_index_to_token_index(row["内容"],
                                                                                       row['Labels'],
                                                                                       row['Seq_Start_Index'],
                                                                                       row['Seq_Len'],
                                                                                       row[
                                                                                           'Token_Index_Map_Char_Index']),
                                    axis=1)

        df = df.reset_index(drop=True)

        return df

    def match_dimension(self, last_dimension):
        """
        标注的标签有些只有最后一级维度,需要根据最后一级的维度匹配到原始的维度
        """
        dimension = None
        for dim in self.dimension_types:
            if last_dimension in dim.split('_')[-1]:
                dimension = dim
                break
        return dimension

    def concatenate_label(self, label):
        """
        拼接标签, 处理错误标签
        """
        if label:
            if ':' in label:
                dimensions = label.split(':')
            elif '-' in label or '_' in label:
                dimensions = label.replace('-', '_').split('_')
            else:
                dimensions = [label]
            matched_dimension = self.match_dimension(dimensions[-1])
            if label == "商业化:宝箱礼包":
                matched_dimension = "天美游戏_商业化_宝箱/礼包"
            if matched_dimension is None:
                raise RuntimeError('[Error]: 数据格式错误, 数据包含不存在于维度表中的标签!', "Label '%s' not in dimension_types!!!" % label)
            return matched_dimension
        return label

    def convert_raw_label(self, content, raw_label):
        """
        {"nodes": [{"end_index": 33, "id": 1, "num_index": 1, "start_index": 31, "text": "包装", "text_index": 1, "type": "特征词"},
                   {"end_index": 37, "id": 2, "num_index": 1, "start_index": 34, "text": "不细致", "text_index": 1, "type": "情感词"}],
         "relations": [{"node1": 1, "relation_type": "情感极性", "relation_value": "负面"},
                       {"node1": 2, "relation_type": "匹配第几个特征词", "relation_value": "1"},
                       {"node1": 1, "relation_type": "维度", "relation_value": "美妆日化-包装:包装密封性"}]}
        """
        # raw_label = self.replace_relations(raw_label)
        nodes = {node['id']: node for node in raw_label['nodes']}
        nodes = dict(sorted(nodes.items(), key=lambda x: x[0]))
        feature_words = {}
        idx = 1
        ################################
        for id_ in nodes:
            while nodes[id_]["text"].startswith(" "):
                nodes[id_]["text"] = nodes[id_]["text"][1:]
                nodes[id_]["start_index"] += 1
            while nodes[id_]["text"].endswith(" "):
                nodes[id_]["text"] = nodes[id_]["text"][:-1]
                nodes[id_]["end_index"] -= 1
        ################################
        for node in nodes.values():
            if '特征词' in node['type']:
                feature_words[idx] = node
                idx += 1
        relations = []
        used_nodes = []
        # 给节点打上维度和情感
        for relation in raw_label['relations']:
            if relation['relation_type'] == '情感极性':
                nodes[relation['node1']]['情感极性'] = relation['relation_value']
            elif relation['relation_type'] == '维度':
                dim = relation['relation_value'].replace('-', '_').replace(':', '_').split('_')
                dim = '_'.join([self.task_domain] + dim[1:])
                if dim not in self.dimension_types:
                    dim = self.concatenate_label(relation['relation_value'])
                    if dim == "天美游戏_商业化_宝箱/礼包" and self.task_domain == "天美游戏过滤维度":
                        dim = "天美游戏过滤维度_商业化_宝箱/礼包"
                    if dim not in self.dimension_types:
                        raise RuntimeError('[Error]: %s not in dimension_types!' % dim)
                        print('[Error]: %s not in dimension_types!' % dim)
                nodes[relation['node1']]['维度'] = dim
        #
        for relation in raw_label['relations']:
            # 旧版标注
            if relation['relation_type'] == '匹配第几个特征词':
                first = nodes[relation['node1']]
                feature_word_index = int(float(relation['relation_value']))
                if feature_words.__contains__(feature_word_index):
                    second = feature_words[feature_word_index]
                else:
                    print('[Error]: 标注错误, 特征词数量异常(缺失)!')
                    print('\t\t\t feature_word_index: %s' % feature_word_index)
                    print('\t\t\t relations: %s' % raw_label['relations'])
                    print('\t\t\t feature_words: %s \n' % feature_words)
                    continue
                if second['start_index'] < first['end_index']:
                    first, second = second, first
                sentiment = first.get('情感极性', None) if first.get('情感极性', None) else second.get('情感极性', None)
                if sentiment is None:
                    continue
                # sentiment = sentiment + '_正序' if first['type'] == '特征词' else sentiment + '_逆序'
                relations.append({'first_start': first['start_index'],
                                  'first_end': first['end_index'],
                                  'first_text': first['text'],
                                  'first_type': first['type'].replace('多情感特征词', '特征词').replace('多情感情感词', '情感词'),
                                  'second_start': second['start_index'],
                                  'second_end': second['end_index'],
                                  'second_text': second['text'],
                                  'second_type': second['type'].replace('多情感特征词', '特征词').replace('多情感情感词', '情感词'),
                                  'sentiment': sentiment,
                                  'dimension': first.get('维度') if first.get('维度') else second.get('维度'),
                                  })
                used_nodes.append(first['id'])
                used_nodes.append(second['id'])
            # 新版标注
            elif relation['relation_type'] == '相关':
                first = nodes[relation['node1']]
                second = nodes[relation['node2']]
                if first.get("type") == "对象" or second.get("type") == "对象":
                    continue
                if second['start_index'] < first['end_index']:
                    first, second = second, first
                sentiment = first.get('情感极性', None) if first.get('情感极性', None) else second.get('情感极性', None)
                if sentiment is None:
                    continue
                # sentiment = sentiment + '_正序' if first['type'] == '特征词' else sentiment + '_逆序'
                relations.append({'first_start': first['start_index'],
                                  'first_end': first['end_index'],
                                  'first_text': first['text'],
                                  'first_type': first['type'].replace('多情感特征词', '特征词').replace('多情感情感词', '情感词'),
                                  'second_start': second['start_index'],
                                  'second_end': second['end_index'],
                                  'second_text': second['text'],
                                  'second_type': second['type'].replace('多情感特征词', '特征词').replace('多情感情感词', '情感词'),
                                  'sentiment': sentiment,
                                  'dimension': first.get('维度') if first.get('维度') else second.get('维度'),
                                  })
                used_nodes.append(first['id'])
                used_nodes.append(second['id'])
        unused_nodes = set(nodes.keys()).difference(set(used_nodes))
        #
        for idx in unused_nodes:
            first = nodes[idx]
            if '特征词' in first['type']:
                try:
                    sentiment = first.get('情感极性', None)
                    if sentiment is None:
                        continue
                    relations.append({'first_start': first['start_index'],
                                      'first_end': first['end_index'],
                                      'first_text': first['text'],
                                      'first_type': '独立特征词',  # first['type'],
                                      'sentiment': sentiment,  # + '_正序',
                                      'dimension': first.get('维度'),
                                      })
                except Exception as e:
                    print('[Error]: %s' % e)
                    print('\t\t\t %s \n' % raw_label)
            else:
                if first.get("type") != "对象":
                    print(f"content：{content}")
                    print(f"没用到的的 node: {first}， 所在的内容：{content}")
            # elif first['type'] == '情感词':
            #     print('[Warning]: 情感词没有匹配的特征词! %s' % first)
            #     print('\t\t\t raw_label --> %s \n' % raw_label)
        return {'nodes': nodes.values(), 'relations': relations}

    def convert_char_index_to_token_index(self,
                                          content,
                                          answer: dict,
                                          seq_start_index: int,
                                          seq_len: int,
                                          token_char_index_dict: dict):
        """
        把分句后的子句子中按字符index标记的答案转换为按token index标记的答案
        :return:
        """
        label = []
        for node in answer['relations']:
            first_start, first_end = None, None
            if seq_start_index <= node['first_start'] <= node['first_end'] <= seq_start_index + seq_len:
                first_start = self.match_index(node['first_start'], token_char_index_dict)
                first_end = self.match_index(node['first_end'], token_char_index_dict)
                if first_end <= first_start:
                    first_end += 1
                    print('debug')
            if node.__contains__('second_start'):
                second_start, second_end = None, None
                if seq_start_index <= node['second_start'] <= node['second_end'] <= seq_start_index + seq_len:
                    second_start = self.match_index(node['second_start'], token_char_index_dict)
                    second_end = self.match_index(node['second_end'], token_char_index_dict)
                    if second_end <= second_start:
                        second_end += 1
                        print('debug')
                # 特征-情感对
                if None not in [first_start, first_end, second_start, second_end]:
                    label.append({'first_start': first_start,
                                  'first_end': first_end,
                                  'first_text': node['first_text'],
                                  'first_type': node['first_type'],
                                  'second_start': second_start,
                                  'second_end': second_end,
                                  'second_text': node['second_text'],
                                  'second_type': node['second_type'],
                                  'sentiment': node['sentiment'],
                                  'dimension': node['dimension'],
                                  })
                    continue
                # 独立特征词
                elif None not in [first_start, first_end]:
                    label.append({'first_start': first_start,
                                  'first_end': first_end,
                                  'first_text': node['first_text'],
                                  'first_type': node['first_type'],
                                  'sentiment': node['sentiment'],
                                  'dimension': node['dimension'],
                                  })
                    continue
            # 独立特征词
            if None not in [first_start, first_end]:
                label.append({'first_start': first_start,
                              'first_end': first_end,
                              'first_text': node['first_text'],
                              'first_type': node['first_type'],
                              'sentiment': node['sentiment'],
                              'dimension': node['dimension'],
                              })
        return label


class AcqpDataset(Dataset):
    def __init__(self, task_domain, data_path, max_seq_len, tokenizer, label_pattern="sentiment_dim", **kwargs):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.task_domain = task_domain
        self.dimension2id = self.get_label_list(task=self.task_domain)
        self.dimension_types = list(self.dimension2id.keys())
        self.id2dimension = {id: dimension for dimension, id in self.dimension2id.items()}
        self.num_dimension_types = len(self.dimension2id)
        self.sentiment2id = {"negative": 0, "neutral": 1, "positive": 2}
        self.sentiment_types = list(self.sentiment2id.keys())
        self.id2sentiment = {id: sentiment for sentiment, id in self.sentiment2id.items()}
        self.label_pattern = label_pattern
        self.label_types = self.get_label_types()
        self.num_label_types = len(self.label_types)
        self.df_raw = read_data_from_txt(data_path, **kwargs)
        #
        self.df = self.get_df()

    @staticmethod
    def get_label_list(task):
        json_path = f"./configs/{task}.json"
        if not os.path.exists(json_path):
            raise RuntimeError(f"Make sure {json_path} exists")
        with open(json_path, "r", encoding="utf-8") as f:
            dimension2id = json.load(f)
        return dimension2id

    def get_label_types(self):
        label_types = ["BA-BO", "EA-EO"]
        if self.label_pattern == "raw":
            label_types.extend(["BA-EO"])
        if self.label_pattern == "sentiment":
            tmp = ['BA-EO-' + x for x in  list(self.sentiment2id.keys())]
            label_types.extend(tmp)
        if self.label_pattern == "sentiment_dim":
            tmp = []
            for dimension_type in list(self.dimension2id.keys()):
                for sentiment_type in list(self.sentiment2id.keys()):
                    tmp.append(f'BA-EO-{dimension_type}-{sentiment_type}')
            label_types.extend(tmp)
        if self.label_pattern == "category":
            tmp = ['BA-EO-' + x for x in list(self.dimension2id.keys())]
            label_types.extend(tmp)
        return label_types

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        tokens = self.df.loc[index, "Tokens"]
        answers = self.df.loc[index, "new_answer"]

        tokens = ['[SEP]'] + tokens
        encoding = self.tokenizer(tokens,
                                  add_special_tokens=True,
                                  max_length=self.max_seq_len,
                                  truncation=True,
                                  padding='max_length',
                                  return_token_type_ids=True,
                                  return_attention_mask=True,
                                  is_split_into_words=True)
        input_ids = encoding["input_ids"]
        token_type_ids = encoding["token_type_ids"]
        attention_mask = encoding["attention_mask"]

        matrix_ids = torch.zeros([self.num_label_types, self.max_seq_len, self.max_seq_len], dtype=torch.int32)
        dimension_ids = torch.zeros(self.num_dimension_types, dtype=torch.float32)
        dimension_sequences = torch.zeros([self.num_dimension_types, self.max_seq_len], dtype=torch.int32)  # category序列
        sentiment_sequences = torch.zeros([3, self.max_seq_len], dtype=torch.int32)  # sentiment序列
        # VA regression targets: [L, 2] for (valence, arousal), va_mask: [L] indicates which positions have VA labels
        va_targets = torch.zeros([self.max_seq_len, 2], dtype=torch.float32)
        va_mask = torch.zeros([self.max_seq_len], dtype=torch.float32)

        for category, aspcet, opinion, sentiment_id in answers:

            BA, EA = aspcet.split(',')
            BA, EA = int(BA), int(EA)
            BO, EO = opinion.split(',')
            BO, EO = int(BO), int(EO)
            if self.label_pattern == "raw":
                query = "BA-EO"
            elif self.label_pattern == "sentiment":
                query = f'BA-EO-{self.id2sentiment[int(sentiment_id)]}'
            elif self.label_pattern == "sentiment_dim":
                query = f'BA-EO-{category}-{self.id2sentiment[int(sentiment_id)]}'
            elif self.label_pattern == "category":
                query = f'BA-EO-{category}'

            # Parse VA values from sentiment_id field (which stores "V#A" or discrete id)
            va_v, va_a = 5.0, 5.0  # default
            sid_str = str(sentiment_id)
            if '#' in sid_str:
                try:
                    va_v, va_a = float(sid_str.split('#')[0]), float(sid_str.split('#')[1])
                except ValueError:
                    pass
            else:
                # Discrete sentiment -> centroid VA (Phase 1 fallback)
                sid_int = int(sentiment_id)
                va_v = [2.5, 5.0, 7.5][sid_int] if 0 <= sid_int <= 2 else 5.0
                va_a = 5.5

            # EA & EO
            if BA != -1 and EA != -1 and BO != -1 and EO != -1:
                matrix_ids[self.label_types.index(query), BA + 2, EO + 1] = 1
                matrix_ids[self.label_types.index("BA-BO"), BA + 2, BO + 2] = 1
                matrix_ids[self.label_types.index("EA-EO"), EA + 1, EO + 1] = 1
                dimension_sequences[self.dimension2id[category], BA + 2: EA + 2] = 1
                dimension_sequences[self.dimension2id[category], BO + 2: EO + 2] = 1
                sentiment_sequences[int(float(sentiment_id)) if '#' not in str(sentiment_id) else 1, BA + 2: EA + 2] = 1
                sentiment_sequences[int(float(sentiment_id)) if '#' not in str(sentiment_id) else 1, BO + 2: EO + 2] = 1
                # VA targets on aspect span
                va_targets[BA + 2: EA + 2, 0] = va_v
                va_targets[BA + 2: EA + 2, 1] = va_a
                va_mask[BA + 2: EA + 2] = 1.0

            # IA & EO  (implicit aspect)
            elif BA == -1 and EA == -1 and BO != -1 and EO != -1:
                matrix_ids[self.label_types.index(query), 1, EO + 1] = 1
                matrix_ids[self.label_types.index("BA-BO"), 1, BO + 2] = 1
                matrix_ids[self.label_types.index("EA-EO"), 1, EO + 1] = 1
                dimension_sequences[self.dimension2id[category], BO + 2: EO + 2] = 1
                sentiment_sequences[int(float(sentiment_id)) if '#' not in str(sentiment_id) else 1, BO + 2: EO + 2] = 1
                # VA targets on [SEP] position (implicit aspect)
                va_targets[1, 0] = va_v
                va_targets[1, 1] = va_a
                va_mask[1] = 1.0

            # EA & IO (implicit opinion)
            elif BA != -1 and EA != -1 and BO == -1 and EO == -1:
                matrix_ids[self.label_types.index(query), BA + 2, 1] = 1
                matrix_ids[self.label_types.index("BA-BO"), BA + 2, 1] = 1
                matrix_ids[self.label_types.index("EA-EO"), EA + 1, 1] = 1
                dimension_sequences[self.dimension2id[category], BA + 2: EA + 2] = 1
                sentiment_sequences[int(float(sentiment_id)) if '#' not in str(sentiment_id) else 1, BA + 2: EA + 2] = 1
                # VA targets on aspect span
                va_targets[BA + 2: EA + 2, 0] = va_v
                va_targets[BA + 2: EA + 2, 1] = va_a
                va_mask[BA + 2: EA + 2] = 1.0

            dimension_ids[self.dimension2id[category]] = 1

        return {"input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask,
                "matrix_ids": matrix_ids,
                "dimension_ids": dimension_ids,
                "dimension_sequences": dimension_sequences,
                "sentiment_sequences": sentiment_sequences,
                "va_targets": va_targets,
                "va_mask": va_mask}

    @staticmethod
    def token_index_map_char_index(token_level_label_index: List[str]):
        try:
            labels = list(map(lambda x: list(map(int, x.split(','))), token_level_label_index))
            labels.append([labels[-1][-1] + 1])
            labels = {k: v for k, v in enumerate(labels)}
            return labels
        except:
            print(token_level_label_index)

    @staticmethod
    def match_index(char_index: int, token_char_index_dict: dict):
        """
        找出字符分词后对应的token index
        :param char_index:
        :param token_char_index_dict:
        :return:
        """
        for k, v in token_char_index_dict.items():
            if char_index in v:
                return k
        print(
            '[Error]: can not match char_index in token_char_index_dict!!!\n This may be due to the fact that the entity is segmented into two sub sentences.')
        print('[Error]: char_index-->%s\t\n token_char_index_dict-->%s\t\t\t\n' % (char_index, token_char_index_dict))
        return None

    def get_df(self):

        encoding = self.tokenizer(self.df_raw["text"].tolist(),
                                  add_special_tokens=False,
                                  return_offsets_mapping=True,
                                  return_overflowing_tokens=True,
                                  max_length=self.max_seq_len - 3)
        results = []
        for i in range(len(encoding["input_ids"])):
            num_tokens = len(encoding["input_ids"][i])
            dic_tmp = {}
            dic_tmp["Tokens"] = encoding.tokens(i)
            overflow_to_sample_mapping = encoding["overflow_to_sample_mapping"][i]
            dic_tmp["Text_Id"] = self.df_raw.iloc[overflow_to_sample_mapping, self.df_raw.columns.get_loc("Text_Id")]
            dic_tmp["Seq"] = self.tokenizer.decode(encoding["input_ids"][i])
            dic_tmp["Seq_Len"] = encoding[i].token_to_chars(num_tokens - 1)[1] - encoding[i].token_to_chars(0)[0]
            New_Labels = []
            last_end, _ = encoding[i].token_to_chars(0)
            for j in range(num_tokens):
                start, end = encoding[i].token_to_chars(j)
                # text:"How old are you …?"
                if last_end == end:
                    last_end = start
                # if last_end == end:
                #     end = last_end + 1
                new_label = list(range(last_end, end))
                new_label = list(map(str, new_label))
                new_label = ",".join(new_label)
                New_Labels.append(new_label)
                last_end = end
            dic_tmp["New_Labels"] = New_Labels
            results.append(dic_tmp)

        df = pd.DataFrame(results)

        #####去掉长度过长的
        print(f'去掉长度前{df.shape}')
        df = df.drop_duplicates(subset='Text_Id', keep=False)
        print(f'去掉长度后{df.shape}')
        #####去掉长度过长的

        df["Seq_Id"] = df.groupby("Text_Id").cumcount()

        print('[Info]: map token char...')
        use_raw_cols = ['text', 'answer', 'Text_Id']
        df = pd.merge(self.df_raw[use_raw_cols], df, on='Text_Id', how='right')
        df['Seq_Lens'] = df['Seq_Len'].apply(lambda x: [x])
        df['Seq_Lens'] = df.groupby('Text_Id')['Seq_Lens'].transform('sum')
        df['Seq_Start_Index'] = df.apply(lambda row: np.sum(
            np.tril(np.ones([len(row['Seq_Lens']) + 1, len(row['Seq_Lens']) + 1]))[:-1, 1:][row['Seq_Id']] * row[
                'Seq_Lens']), axis=1).astype(int)
        df['Token_Len'] = df['Tokens'].str.len()
        df['Token_Lens'] = df['Token_Len'].apply(lambda x: [x])
        df['Token_Lens'] = df.groupby('Text_Id')['Token_Lens'].transform('sum')
        df['Token_Start_Index'] = df.apply(lambda row: np.sum(
            np.tril(np.ones([len(row['Token_Lens']) + 1, len(row['Token_Lens']) + 1]))[:-1, 1:][row['Seq_Id']] * row[
                'Token_Lens']), axis=1).astype(int)
        df['Token_Index_Map_Char_Index'] = df['New_Labels'].apply(self.token_index_map_char_index)

        if 'answer' in df.columns:
            print('[Info]: convert_char_index_to_token_index...')
            df['new_answer'] = df.apply(lambda row: self.convert_char_index_to_token_index(row["text"],
                                                                                           row['answer'],
                                                                                           row['Seq_Start_Index'],
                                                                                           row['Seq_Len'],
                                                                                           row[
                                                                                               'Token_Index_Map_Char_Index']),
                                        axis=1)

        df = df.reset_index(drop=True)

        return df

    def convert_char_index_to_token_index(self,
                                          text,
                                          answer: dict,
                                          seq_start_index: int,
                                          seq_len: int,
                                          token_char_index_dict: dict):
        """把 char index 映射到 token index
        """
        new_labels = []
        for category, aspect, opinion, sentiment in answer:
            aspect_char_start_index, aspect_char_end_index = aspect.split(",")
            aspect_char_start_index, aspect_char_end_index = int(aspect_char_start_index), int(aspect_char_end_index)
            opinion_char_start_index, opinion_char_end_index = opinion.split(",")
            opinion_char_start_index, opinion_char_end_index = int(opinion_char_start_index), int(
                opinion_char_end_index)

            aspect_token_start_index, aspect_token_end_index = None, None
            if seq_start_index <= aspect_char_start_index <= aspect_char_end_index <= seq_start_index + seq_len:
                aspect_token_start_index = self.match_index(aspect_char_start_index, token_char_index_dict)
                aspect_token_end_index = self.match_index(aspect_char_end_index, token_char_index_dict)
                if aspect_token_end_index <= aspect_token_start_index:
                    aspect_token_end_index = aspect_token_start_index + 1
                    logger.info(f"[Debug]: {text}, {answer}")

            if aspect_token_start_index is not None and aspect_token_end_index is not None:
                new_aspect = ",".join([str(aspect_token_start_index), str(aspect_token_end_index)])
            else:
                new_aspect = '-1,-1'

            opinion_token_start_index, opinion_token_end_index = None, None
            if seq_start_index <= opinion_char_start_index <= opinion_char_end_index <= seq_start_index + seq_len:
                opinion_token_start_index = self.match_index(opinion_char_start_index, token_char_index_dict)
                opinion_token_end_index = self.match_index(opinion_char_end_index, token_char_index_dict)
                if opinion_token_end_index <= opinion_token_start_index:
                    opinion_token_end_index = opinion_token_start_index + 1
                    logger.info(f"[Debug]: {text}, {answer}")
            if opinion_token_start_index is not None and opinion_token_end_index is not None:
                new_opinion = ','.join([str(opinion_token_start_index), str(opinion_token_end_index)])
            else:
                new_opinion = '-1,-1'

            new_labels.append([category, new_aspect, new_opinion, sentiment])

        return new_labels


def collate_fn(batch):
    input_ids = torch.tensor([x['input_ids'] for x in batch])
    token_type_ids = torch.tensor([x['token_type_ids'] for x in batch])
    attention_mask = torch.tensor([x['attention_mask'] for x in batch])
    matrix_ids = torch.stack([x["matrix_ids"] for x in batch])
    dimension_ids = torch.stack([x["dimension_ids"] for x in batch])
    dimension_sequences = torch.stack([x["dimension_sequences"] for x in batch])
    sentiment_sequences = torch.stack([x["sentiment_sequences"] for x in batch])
    va_targets = torch.stack([x["va_targets"] for x in batch])
    va_mask = torch.stack([x["va_mask"] for x in batch])

    return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_mask,
        "matrix_ids": matrix_ids,
        "dimension_ids": dimension_ids,
        "dimension_sequences": dimension_sequences,
        "sentiment_sequences": sentiment_sequences,
        "va_targets": va_targets,
        "va_mask": va_mask,
    }


if __name__ == '__main__':
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
    for x in dataloader:
        a = 4
