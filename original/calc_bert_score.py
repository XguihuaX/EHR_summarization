from evaluate import load
import csv
import pandas as pd
from bert_score import score
import numpy as np
import math
import copy
from sklearn.metrics import cohen_kappa_score

# Initialize an empty list to store rows
csv_list = []
tool ='spacy'
column_names = ['text', 'true', 'pred', 'rule_llm_annotation', 'evaluation reasoning', 'extraction reasoning']


semantic_types_list = ['tmco', 'spco', 'qlco', 'fndg', 'sosy', 'acty', 'phpr']
entity = input(f"input the entity you want to annotate（such as {', '.join(semantic_types_list)}）: ")

path_to_dir = f'/ihome/hdaqing/jul230/program/result/tools/'
files = [f'{tool}_evaluated_annotations_extracted_{entity}.csv']

for p in range(len(files)):

    df = pd.read_csv(path_to_dir + files[p], names=column_names)

    deberta_score = []
    roberta_score = []

    rule_llm_annotation = []
    reasoning_list = []

    results_deberta = []
    results_roberta = []

    preds = []
    trues = []

    for i in range(len(list(df['pred']))):
        if isinstance(list(df['pred'])[i], float) or isinstance(list(df['true'])[i], float):  # if the string is 'nan' replace with ''
            preds.append('')
            trues.append('')
        else:
            preds.append(list(df['pred'])[i])
            trues.append(list(df['true'])[i])

    P, R, results_deberta = score(preds, trues, lang="en", verbose=True, model_type='khalidalt/DeBERTa-v3-large-mnli')
    P, R, results_roberta = score(preds, trues, lang="en", verbose=True, model_type='roberta-large')

    threshold_078_deberta = []
    threshold_093_roberta = []

    for i in range(len(preds)):
        if not isinstance(list(df['evaluation reasoning'])[i], float):
            if float(results_deberta[i]) < 0.78:
                threshold_078_deberta.append('Dissimilar')
            else:
                threshold_078_deberta.append('Similar')

            if float(results_roberta[i]) < 0.93:
                threshold_093_roberta.append('Dissimilar')
            else:
                threshold_093_roberta.append('Similar')
        else:
            threshold_078_deberta.append(list(df['rule_llm_annotation'])[i])
            threshold_093_roberta.append(list(df['rule_llm_annotation'])[i])

    df = pd.DataFrame({
        'text': list(df['text']),
        'true': trues,
        'pred': preds,
        'rule_llm_annotation': list(df['rule_llm_annotation']),
        'deberta_score': results_deberta.tolist(),
        'roberta_score': results_roberta.tolist(),
        'threshold_078_deberta': threshold_078_deberta,
        'threshold_093_roberta': threshold_093_roberta,
        'evaluation reasoning': list(df['evaluation reasoning']),
        'extraction reasoning': list(df['extraction reasoning'])
    })

    path = '/ihome/hdaqing/jul230/program/result/tools/'  # 设置保存结果的路径
    filename = path + f'{tool}_new_roberta_deberta_score_{files[p]}'

    df.to_csv(filename, index=False)

    print(f"CSV file '{filename}' has been created successfully.")
