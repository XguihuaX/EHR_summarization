import pandas as pd
import re
from tqdm import tqdm
import os

# 要处理的文件列表
files = ['metamap_extracted_results_tmco.csv']

# 输入文件所在目录
input_path = '/ihome/hdaqing/jul230/program/result/tools/'


# 输出文件保存路径
output_file_path =f'/ihome/hdaqing/jul230/program/result/tools/{tool}_rule_annotations_extracted_{entity}.csv'

for f in files:
    df = pd.read_csv(input_path + f)
    df.columns = ['text', 'true', 'pred', 'extraction reasoning']
    true = list(df['true'])
    pred = list(df['pred'])

    n = len(true)
    rule_annotation = ['' for i in range(n)]

    # 使用 tqdm 包装循环，添加进度条
    for i in tqdm(range(n), desc=f"Processing {f}"):
        if true[i] == ' ' or str(true[i]) == 'nan':
            true_compare = ''  # 替换为空或nan
        else:
            true_compare = str(true[i])

        if pred[i] == ' ' or str(pred[i]) == 'nan':
            pred_compare = ''
        else:
            pred_compare = str(pred[i])

        pattern = r"\b((n|N)o\s|(n|N)ot\s|(n|N)othing\b|N/A|n/a|(w|W)ithout\s|(d|D)enies\s|(n|N)on[\w-]*)\b"

        no_in_true = bool(re.search(pattern, true_compare))
        no_in_pred =  bool(re.search(pattern, pred_compare))

        if no_in_true:
            true_compare = ''  # 如果字符串包含模式中的任何单词，替换为空字符串
        if no_in_pred:
            pred_compare = ''

        if true_compare == '' and pred_compare != '':
            rule_annotation[i] = 'Spurious'
        elif pred_compare == '' and true_compare != '':
            rule_annotation[i] = 'Missing'
        elif true_compare == '' and pred_compare == '':
            rule_annotation[i] = 'Similar'
        elif true_compare == pred_compare:
            rule_annotation[i] = 'Similar'
        else:
            rule_annotation[i] = 'to evaluate'  # 需要由LLM评估

    df['rule_annotation'] = rule_annotation
    ordered_columns = ['text', 'true', 'pred', 'rule_annotation', 'extraction reasoning']

    # 保存结果到输出文件
    df.to_csv(output_file_path, columns=ordered_columns, index=False)
