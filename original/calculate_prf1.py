import pandas as pd

column_names = ['text', 'true', 'pred', 'rule_llm_annotation', 'evaluation reasoning', 'extraction reasoning']

semantic_types_list = ['tmco', 'spco', 'qlco', 'fndg', 'sosy', 'acty', 'phpr']
entity = input(f"input the entity you want to annotate（such as {', '.join(semantic_types_list)}）: ")

path_to_dir = '/ihome/hdaqing/jul230/program/result/tools/'  # 替换为你的文件目录

files = [f'{tool}_evaluated_annotations_extractd_{entity}.csv']  # 文件名列表

for file_name in files:
    df = pd.read_csv(path_to_dir + file_name, names=column_names, header=0)  # header=0 表示文件中第一行是列名

    list1 = list(df['rule_llm_annotation'])

    # 处理annotation标签
    refined_list1 = [str(i).strip() for i in list1]

    def convert(l):  # 将 Similar/Dissimilar 转换为 Correct 和 Incorrect
        for i in range(len(l)):
            if l[i] == 'Similar':
                l[i] = 'Correct'
            if l[i] == 'Dissimilar':
                l[i] = 'Incorrect'
        return l

    refined_list1 = convert(refined_list1)

    # 计算不同标签的数量
    correct = sum([1 for item in refined_list1 if item == 'Correct'])
    incorrect = sum([1 for item in refined_list1 if item == 'Incorrect'])
    spurious = sum([1 for item in refined_list1 if item == 'Spurious'])
    missing = sum([1 for item in refined_list1 if item == 'Missing'])

    possible = correct + incorrect + missing
    actual = correct + incorrect + spurious

    # 计算 precision, recall 和 f1 值
    precision = correct / actual if actual > 0 else 0
    recall = correct / possible if possible > 0 else 0
    f1 = (2 * (precision * recall)) / (precision + recall) if (precision + recall) > 0 else 0

    print(f'File: {file_name}')
    print('precision', round(precision, 3))
    print('recall', round(recall, 3))
    print('f1', round(f1, 3))
    print('---------')
