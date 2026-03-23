"""
@Time : 2022/12/1717:11
@Auth : zhoujx
@File ：predict.py
@DESCRIPTION:

"""
import json
import os
import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from dataset.dataset import AcqpDataset, collate_fn
from utils.argparse import get_predict_argparse
from utils.utils import load_train_model, load_train_args

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Use {device} device")


def predict(args):
    model = load_train_model(args.model_path)
    model = model.to(device)

    # get training data
    training_args_dic = load_train_args(args.model_path)
    label_pattern = training_args_dic['label_pattern']

    print(json.dumps(training_args_dic, indent=4))

    # dataloader
    tokenizer = AutoTokenizer.from_pretrained(training_args_dic['model_name_or_path'])
    test_dataset = AcqpDataset(task_domain=training_args_dic['task_domain'],
                               tokenizer=tokenizer,
                               data_path=args.test_data,
                               max_seq_len=training_args_dic['max_seq_len'],
                               label_pattern=training_args_dic['label_pattern'],
                               nrows=args.nrows)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=args.per_gpu_test_batch_size,
                                 shuffle=False,
                                 num_workers=4,
                                 pin_memory=True,
                                 drop_last=False,
                                 collate_fn=collate_fn)
    model.eval()
    total_step = len(test_dataloader)
    loop = tqdm(enumerate(test_dataloader), total=total_step)
    mat_preds = []
    dim_seq_preds = []
    sen_seq_preds = []
    dim_seq_trues = []
    with torch.no_grad():
        for step, data in loop:
            for key in data:
                data[key] = data[key].to(device)

            pred = model(input_ids=data["input_ids"],
                         token_type_ids=data["token_type_ids"],
                         attention_mask=data["attention_mask"])
            pred_matrix = pred["matrix"].cpu().numpy()
            pred_dim_seq = pred["dimension_sequence"].cpu().numpy()
            pred_sen_seq = pred["sentiment_sequence"].cpu().numpy()
            for matrix_ in pred_matrix:
                mat_preds.append(np.argwhere(matrix_ > 0).tolist())
            for dim_seq_ in pred_dim_seq:
                dim_seq_preds.append(np.argwhere(dim_seq_ > 0).tolist())
            for sen_seq_ in pred_sen_seq:
                sen_seq_preds.append(np.argwhere(sen_seq_ > 0).tolist())
            for true_dimension_sequences in data['dimension_sequences'].cpu().numpy():
                dim_seq_trues.append(np.argwhere(true_dimension_sequences > 0).tolist())

    df_test = test_dataset.df
    df_test["pred_matrix"] = mat_preds
    df_test['pred_dim_seq'] = dim_seq_preds
    df_test['true_dim_seq'] = dim_seq_trues
    df_test['pred_sen_seq'] = sen_seq_preds
    label_types = test_dataset.label_types
    sentiment2id = test_dataset.sentiment2id
    id2sentiment = test_dataset.id2sentiment
    dimension_types = test_dataset.dimension_types

    # decode
    df_test, reports = decode(df_test, label_pattern, label_types, dimension_types, sentiment2id, id2sentiment)

    predict_file_name = os.path.split(args.test_data)[1].replace('.txt', '_predict.csv')
    predict_file_path = os.path.join(args.model_path, predict_file_name)
    df_test.to_csv(predict_file_path, index=False, encoding="utf-8-sig")
    logger.info(f'predict_csv save in {predict_file_path}')
    metric_file_path = os.path.join(args.model_path, os.path.split(args.test_data)[1].replace('.txt', '_metrics.json'))
    with open(metric_file_path, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)
    logger.info(f'metrics_json save in {metric_file_path}')
    preds_file_path = os.path.join(args.model_path, os.path.split(args.test_data)[1].replace('.txt', '_predictions.json'))
    keep_cols = ['Text_Id', 'text', 'answer', 'pred_answer', 'true_quadruple', 'pred_quadruple', 'true_triplet', 'pred_triplet']
    with open(preds_file_path, "w", encoding="utf-8") as f:
        json.dump(df_test[keep_cols].to_dict(orient="records"), f, indent=2, ensure_ascii=False, default=list)
    logger.info(f'predictions_json save in {preds_file_path}')


def decode(df, label_pattern, label_types, dimension_types, sentiment2id, id2sentiment):
    df['pred_answer'] = df.apply(lambda row: create_pred_answer(row['pred_matrix'],
                                                                row['pred_dim_seq'],
                                                                row['pred_sen_seq'],
                                                                row['Token_Index_Map_Char_Index'],
                                                                label_pattern,
                                                                label_types,
                                                                dimension_types,
                                                                sentiment2id
                                                                ), axis=1)
    df['pred_triplet_answer'] = df.apply(lambda row: create_triplet_pred_answer(row['pred_matrix'],
                                                                row['pred_dim_seq'],
                                                                row['pred_sen_seq'],
                                                                row['Token_Index_Map_Char_Index'],
                                                                label_pattern,
                                                                label_types,
                                                                dimension_types,
                                                                sentiment2id
                                                                ), axis=1)

    df["true_quadruple"] = df.apply(lambda row: create_quadruple(row['text'], row['answer'], id2sentiment), axis=1)
    df["pred_quadruple"] = df.apply(lambda row: create_quadruple(row['text'], row['pred_answer'], id2sentiment), axis=1)

    df["true_triplet"] = df.apply(lambda row: create_triplet(row['text'], row['answer'], id2sentiment), axis=1)
    df["pred_triplet"] = df.apply(lambda row: create_triplet(row['text'], row['pred_answer'], id2sentiment), axis=1)

    df['is_equal'] = df.apply(lambda row: is_equal(row["true_quadruple"], row["pred_quadruple"]), axis=1)
    category_seq_report = get_category_seq_report(df)
    triplet_report = get_triplet_report(df)
    quad_report = get_report(df)

    return df, {
        "category_seq_report": category_seq_report,
        "triplet_report": triplet_report,
        "quad_report": quad_report,
    }


def is_equal(true_quadruple, pred_quadruple):
    true_quadruple = {x for x in true_quadruple if not (x[1] is None and x[2] is None)}
    if (true_quadruple - pred_quadruple) or (pred_quadruple - true_quadruple):
        return False
    return True


def create_pred_answer(pred_matrix,
                       pred_dim_seq,
                       pred_sen_seq,
                       token_index_map_char_index,
                       label_pattern,
                       label_types,
                       dimension_types,
                       sentiment2id):
    BA_BO = [x for x in pred_matrix if x[0] == 0]
    EA_EO = [x for x in pred_matrix if x[0] == 1]
    BA_EO = [x for x in pred_matrix if x[0] > 1]
    triples = []
    results = []
    for label_type_id, aspect_token_start_index, opinion_token_end_index in BA_EO:
        cand_BA_BO = [x for x in BA_BO if x[1] == aspect_token_start_index and x[2] <= opinion_token_end_index]
        cand_EA_EO = [x for x in EA_EO if x[2] == opinion_token_end_index and x[1] >= aspect_token_start_index]
        aspect_token_end_index = cand_EA_EO[0][1] if cand_EA_EO else aspect_token_start_index
        opinion_token_start_index = cand_BA_BO[-1][2] if cand_BA_BO else opinion_token_end_index

        # EA & EO
        if aspect_token_start_index != 1 and aspect_token_end_index != 1 and opinion_token_start_index != 1 and opinion_token_end_index != 1:
            aspect_char_start_index = token_index_map_char_index.get(aspect_token_start_index - 2, [-1])[0]
            aspect_char_end_index = token_index_map_char_index.get(aspect_token_end_index - 2, [-2])[-1] + 1
            opinion_char_start_index = token_index_map_char_index.get(opinion_token_start_index - 2, [-1])[0]
            opinion_char_end_index = token_index_map_char_index.get(opinion_token_end_index - 2, [-2])[-1] + 1
        # IA & EO
        elif aspect_token_start_index == 1 and aspect_token_end_index == 1 and opinion_token_start_index != 1 and opinion_token_end_index != 1:
            aspect_char_start_index = -1
            aspect_char_end_index = -1
            opinion_char_start_index = token_index_map_char_index.get(opinion_token_start_index - 2, [-1])[0]
            opinion_char_end_index = token_index_map_char_index.get(opinion_token_end_index - 2, [-2])[-1] + 1
        # EA & IO
        elif aspect_token_start_index != 1 and aspect_token_end_index != 1 and opinion_token_start_index == 1 and opinion_token_end_index == 1:
            aspect_char_start_index = token_index_map_char_index.get(aspect_token_start_index - 2, [-1])[0]
            aspect_char_end_index = token_index_map_char_index.get(aspect_token_end_index - 2, [-2])[-1] + 1
            opinion_char_start_index = -1
            opinion_char_end_index = -1
        else:
            continue

        aspect_char_index = ','.join([str(aspect_char_start_index), str(aspect_char_end_index)])
        opinion_char_index = ','.join([str(opinion_char_start_index), str(opinion_char_end_index)])

        if label_pattern == 'sentiment_dim':
            _, _, *category, sentiment = label_types[label_type_id].split('-')
            category = "-".join(category)
            sentiment_id = sentiment2id[sentiment]

            results.append([category, aspect_char_index, opinion_char_index, str(sentiment_id)])
        elif label_pattern == 'sentiment':
            sentiment = label_types[label_type_id].split('-')[-1]
            sentiment_id = sentiment2id[sentiment]

            category_ids = find_category_by_dim_seq(aspect_token_start_index,
                                                    opinion_token_start_index,
                                                    pred_dim_seq)

            for category_id in category_ids:
                category = dimension_types[category_id]
                results.append([category, aspect_char_index, opinion_char_index, str(sentiment_id)])
        elif label_pattern == 'raw':
            category_ids, sentiment_ids = find_category_sentiment_by_dim_seq(aspect_token_start_index,
                                                                             opinion_token_start_index,
                                                                             pred_dim_seq,
                                                                             pred_sen_seq)
            for category_id in category_ids:
                category = dimension_types[category_id]
                for sentiment_id in sentiment_ids:
                    results.append([category, aspect_char_index, opinion_char_index, str(sentiment_id)])

    # fix
    EA_EO = [x for x in results if x[1] != "-1,-1" and x[2] != "-1,-1"]
    x0_x1_x3 = [(x[0], x[1], x[3]) for x in EA_EO]
    x0_x2_x3 = [(x[0], x[2], x[3]) for x in EA_EO]

    #
    new_results = []
    for x in results:
        if x[1] == "-1,-1":
            if (x[0], x[2], x[3]) in x0_x2_x3:
                continue
        if x[2] == '-1,-1':
            if (x[0], x[1], x[3]) in x0_x1_x3:
                continue
        new_results.append(x)
    results = new_results

    return results

def create_triplet_pred_answer(pred_matrix,
                       pred_dim_seq,
                       pred_sen_seq,
                       token_index_map_char_index,
                       label_pattern,
                       label_types,
                       dimension_types,
                       sentiment2id):
    BA_BO = [x for x in pred_matrix if x[0] == 0]
    EA_EO = [x for x in pred_matrix if x[0] == 1]
    BA_EO = [x for x in pred_matrix if x[0] > 1]
    triples = []
    results = []
    for label_type_id, aspect_token_start_index, opinion_token_end_index in BA_EO:
        cand_BA_BO = [x for x in BA_BO if x[1] == aspect_token_start_index and x[2] <= opinion_token_end_index]
        cand_EA_EO = [x for x in EA_EO if x[2] == opinion_token_end_index and x[1] >= aspect_token_start_index]
        aspect_token_end_index = cand_EA_EO[0][1] if cand_EA_EO else aspect_token_start_index
        opinion_token_start_index = cand_BA_BO[-1][2] if cand_BA_BO else opinion_token_end_index

        # EA & EO
        if aspect_token_start_index != 1 and aspect_token_end_index != 1 and opinion_token_start_index != 1 and opinion_token_end_index != 1:
            aspect_char_start_index = token_index_map_char_index.get(aspect_token_start_index - 2, [-1])[0]
            aspect_char_end_index = token_index_map_char_index.get(aspect_token_end_index - 2, [-2])[-1] + 1
            opinion_char_start_index = token_index_map_char_index.get(opinion_token_start_index - 2, [-1])[0]
            opinion_char_end_index = token_index_map_char_index.get(opinion_token_end_index - 2, [-2])[-1] + 1
        # IA & EO
        elif aspect_token_start_index == 1 and aspect_token_end_index == 1 and opinion_token_start_index != 1 and opinion_token_end_index != 1:
            aspect_char_start_index = -1
            aspect_char_end_index = -1
            opinion_char_start_index = token_index_map_char_index.get(opinion_token_start_index - 2, [-1])[0]
            opinion_char_end_index = token_index_map_char_index.get(opinion_token_end_index - 2, [-2])[-1] + 1
        # EA & IO
        elif aspect_token_start_index != 1 and aspect_token_end_index != 1 and opinion_token_start_index == 1 and opinion_token_end_index == 1:
            aspect_char_start_index = token_index_map_char_index.get(aspect_token_start_index - 2, [-1])[0]
            aspect_char_end_index = token_index_map_char_index.get(aspect_token_end_index - 2, [-2])[-1] + 1
            opinion_char_start_index = -1
            opinion_char_end_index = -1
        # IA & IO
        # elif aspect_token_start_index == 1 and aspect_token_end_index == 1 and opinion_token_start_index == 1 and opinion_token_end_index == 1:
        #     aspect_char_start_index = -1
        #     aspect_char_end_index = -1
        #     opinion_char_start_index = -1
        #     opinion_char_end_index = -1
        else:
            continue

        aspect_char_index = ','.join([str(aspect_char_start_index), str(aspect_char_end_index)])
        opinion_char_index = ','.join([str(opinion_char_start_index), str(opinion_char_end_index)])

        sentiment = label_types[label_type_id].split('-')[-1]
        sentiment_id = sentiment2id[sentiment]

        results.append([aspect_char_index, opinion_char_index, str(sentiment_id)])


    # fix
    EA_EO = [x for x in results if x[0] != "-1,-1" and x[0] != "-1,-1"]
    # x0_x1_x3 = [(x[0], x[1], x[3]) for x in EA_EO]
    # x0_x2_x3 = [(x[0], x[2], x[3]) for x in EA_EO]

    #
    # new_results = []
    # for x in results:
    #     if x[1] == "-1,-1":
    #         if (x[0], x[2], x[3]) in x0_x2_x3:
    #             continue
    #     if x[2] == '-1,-1':
    #         if (x[0], x[1], x[3]) in x0_x1_x3:
    #             continue
    # new_results.append(x)
    # results = new_results

    return results

def find_category_by_dim_seq(aspect_token_start_index, opinion_token_start_index, pred_dim_seq):
    # 显示特征词 & 隐含维度词
    if aspect_token_start_index != 1 and opinion_token_start_index == 1:
        category_ids = {x[0] for x in pred_dim_seq if x[1] == aspect_token_start_index}
    # 隐式特征词 & 显式维度词
    elif aspect_token_start_index == 1 and opinion_token_start_index != 1:
        category_ids = {x[0] for x in pred_dim_seq if x[1] == opinion_token_start_index}
    # elif aspect_token_start_index == 1 and opinion_token_start_index == 1:
    #     aspect_category_ids = {x[0] for x in pred_dim_seq if x[1] == aspect_token_start_index}
    #     opinion_category_ids = {x[0] for x in pred_dim_seq if x[1] == opinion_token_start_index}
    #     category_ids = aspect_category_ids & opinion_category_ids
    # 显示特征词 & 显示维度词
    elif aspect_token_start_index != 1 and opinion_token_start_index != 1:
        aspect_category_ids = {x[0] for x in pred_dim_seq if x[1] == aspect_token_start_index}
        opinion_category_ids = {x[0] for x in pred_dim_seq if x[1] == opinion_token_start_index}
        category_ids = aspect_category_ids & opinion_category_ids
    return category_ids


def find_category_sentiment_by_dim_seq(aspect_token_start_index, opinion_token_start_index, pred_dim_seq, pred_sen_seq):
    # 显示特征词 & 隐含维度词
    if aspect_token_start_index != 1 and opinion_token_start_index == 1:
        category_ids = {x[0] for x in pred_dim_seq if x[1] == aspect_token_start_index}
        sentiment_ids = {x[0] for x in pred_sen_seq if x[1] == aspect_token_start_index}
    # 隐式特征词 & 显式维度词
    elif aspect_token_start_index == 1 and opinion_token_start_index != 1:
        category_ids = {x[0] for x in pred_dim_seq if x[1] == opinion_token_start_index}
        sentiment_ids = {x[0] for x in pred_sen_seq if x[1] == opinion_token_start_index}
    # 显示特征词 & 显示维度词
    elif aspect_token_start_index != 1 and opinion_token_start_index != 1:
        aspect_category_ids = {x[0] for x in pred_dim_seq if x[1] == aspect_token_start_index}
        opinion_category_ids = {x[0] for x in pred_dim_seq if x[1] == opinion_token_start_index}
        category_ids = aspect_category_ids & opinion_category_ids
        aspect_sentiment_ids = {x[0] for x in pred_sen_seq if x[1] == aspect_token_start_index}
        opinion_sentiment_ids = {x[0] for x in pred_sen_seq if x[1] == opinion_token_start_index}
        sentiment_ids = aspect_sentiment_ids & opinion_sentiment_ids

    return category_ids, sentiment_ids


def create_quadruple(text, answer, id2sentiment):
    results = set()
    for category, aspect_index, opinion_index, sentimentid in answer:

        # 修bug
        if aspect_index == '-1, -1' and opinion_index == '-1, -1':
            continue

        # aspect_index
        aspect_char_start_index, aspect_char_end_index = aspect_index.split(',')
        aspect_char_start_index, aspect_char_end_index = int(aspect_char_start_index), int(aspect_char_end_index)
        if aspect_char_start_index == -1:
            aspect = None
        else:
            aspect = text[aspect_char_start_index: aspect_char_end_index].strip()
        # category
        category = category
        # sentiment
        sentiment = id2sentiment[int(sentimentid)]
        # opinion
        opinion_char_start_index, opinion_char_end_index = opinion_index.split(',')
        opinion_char_start_index, opinion_char_end_index = int(opinion_char_start_index), int(opinion_char_end_index)
        if opinion_char_start_index == -1:
            opinion = None
        else:
            opinion = text[opinion_char_start_index: opinion_char_end_index].strip()
        results.add((category, aspect, opinion, sentiment))
    return results

def create_triplet(text, answer, id2sentiment):
    results = set()

    if answer and len(answer[0]) == 4:
        for cateogyr, aspect_index, opinion_index, sentimentid in answer:
            if aspect_index == '-1,-1' and opinion_index == '-1,-1':
                continue
            # aspect_index
            aspect_char_start_index, aspect_char_end_index = aspect_index.split(',')
            aspect_char_start_index, aspect_char_end_index = int(aspect_char_start_index), int(
                aspect_char_end_index)
            if aspect_char_start_index == -1:
                aspect = None
            else:
                aspect = text[aspect_char_start_index: aspect_char_end_index].strip()
            # category
            # sentiment
            sentiment = id2sentiment[int(sentimentid)]
            # opinion
            opinion_char_start_index, opinion_char_end_index = opinion_index.split(',')
            opinion_char_start_index, opinion_char_end_index = int(opinion_char_start_index), int(
                opinion_char_end_index)
            if opinion_char_start_index == -1:
                opinion = None
            else:
                opinion = text[opinion_char_start_index: opinion_char_end_index].strip()
            results.add((aspect, opinion, sentiment))
    if answer and len(answer[0]) == 3:
        for aspect_index, opinion_index, sentimentid in answer:
            if aspect_index == '-1,-1' and opinion_index == '-1,-1':
                continue
            # aspect_index
            aspect_char_start_index, aspect_char_end_index = aspect_index.split(',')
            aspect_char_start_index, aspect_char_end_index = int(aspect_char_start_index), int(aspect_char_end_index)
            if aspect_char_start_index == -1:
                aspect = None
            else:
                aspect = text[aspect_char_start_index: aspect_char_end_index].strip()
            # category
            # sentiment
            sentiment = id2sentiment[int(sentimentid)]
            # opinion
            opinion_char_start_index, opinion_char_end_index = opinion_index.split(',')
            opinion_char_start_index, opinion_char_end_index = int(opinion_char_start_index), int(opinion_char_end_index)
            if opinion_char_start_index == -1:
                opinion = None
            else:
                opinion = text[opinion_char_start_index: opinion_char_end_index].strip()
            results.add((aspect, opinion, sentiment))
    return results


def get_report(df):
    true_total = 0
    pred_total = 0
    true_positive = 0
    for true_quadruple, pred_quadruple in zip(df['true_quadruple'].tolist(), df['pred_quadruple'].tolist()):
        # TODO : 忽略 IA & IO
        true_quadruple = {x for x in true_quadruple if not (x[1] is None and x[2] is None)}
        true_total += len(true_quadruple)
        pred_total += len(pred_quadruple)
        true_positive += len(true_quadruple & pred_quadruple)
    if true_total == 0:
        return None
    precision = true_positive / pred_total if pred_total > 0 else 0
    recall = true_positive / true_total
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    logger.info(f'true_total: {true_total}, true_positive: {true_positive}')
    logger.info(f'Precison: {precision}, Recall: {recall}, F1: {f1}')
    return {
        "true_total": true_total,
        "pred_total": pred_total,
        "true_positive": true_positive,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def get_category_seq_report(df):
    true_total = 0
    pred_total = 0
    true_positive = 0

    fist_token_true_dim_label = []
    for x1 in df['true_dim_seq'].tolist():
        new_tmp = []
        for x2 in x1:
            if [x2[0], x2[1]-1] in x1:
                continue
            new_tmp.append(x2)
        fist_token_true_dim_label.append(new_tmp)

    fist_token_pred_dim_label = []
    for x1 in df['pred_dim_seq'].tolist():
        new_tmp = []
        for x2 in x1:
            if [x2[0], x2[1]-1] in x1:
                continue
            new_tmp.append(x2)
        fist_token_pred_dim_label.append(new_tmp)

    # for true_dim_seq, pred_dim_seq in zip(df['true_dim_seq'].tolist(), df['pred_dim_seq'].tolist()):
    for true_dim_seq, pred_dim_seq in zip(fist_token_true_dim_label, fist_token_pred_dim_label):
        true_total += len(true_dim_seq)
        pred_total += len(pred_dim_seq)
        for x in pred_dim_seq:
            if x in true_dim_seq:
                true_positive += 1
    if true_total == 0:
        return None
    precision = true_positive / pred_total if pred_total > 0 else 0
    recall = true_positive / true_total
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    logger.info(f'Category Sequence Precison: {precision}, Recall: {recall}, F1: {f1}')
    return {
        "true_total": true_total,
        "pred_total": pred_total,
        "true_positive": true_positive,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def get_triplet_report(df):
    true_total = 0
    pred_total = 0
    true_positive = 0
    for true_quadruple, pred_quadruple in zip(df['true_triplet'].tolist(), df['pred_triplet'].tolist()):
        # TODO : 忽略 IA & IO
        true_quadruple = {x for x in true_quadruple if not (x[0] is None and x[1] is None)}
        true_total += len(true_quadruple)
        pred_total += len(pred_quadruple)
        true_positive += len(true_quadruple & pred_quadruple)
    if true_total == 0:
        return None
    precision = true_positive / pred_total if pred_total > 0 else 0
    recall = true_positive / true_total
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    logger.info(f'Triplet Precison: {precision}, Recall: {recall}, F1: {f1}')
    return {
        "true_total": true_total,
        "pred_total": pred_total,
        "true_positive": true_positive,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }



if __name__ == '__main__':
    args = get_predict_argparse().parse_args()
    predict(args)
