import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

tool='metamap'
entity='spco'
input_file_path = f'/ihome/hdaqing/jul230/program/result/tools/{tool}_extracted_results_{entity}.csv'
df = pd.read_csv(input_file_path)

# 初始化标签列表
true_labels = []
pred_labels = []

# 将 true 和 pred 列的内容读取到列表中
true = list(df['true'])
pred = list(df['pred'])

# 遍历每一行数据
for i in range(len(true)):
    true_labels.append(set(true[i]))
    pred_labels.append(set(pred[i]))

# 计算所有唯一标签
all_labels = sorted(set.union(*true_labels, *pred_labels))

# 将标签转换为向量
true_vectors = []
pred_vectors = []

for true_set, pred_set in zip(true_labels, pred_labels):
    true_vector = [1 if label in true_set else 0 for label in all_labels]
    pred_vector = [1 if label in pred_set else 0 for label in all_labels]
    true_vectors.append(true_vector)
    pred_vectors.append(pred_vector)

# 将向量展平成一维列表
true_labels_flat = [item for sublist in true_vectors for item in sublist]
pred_labels_flat = [item for sublist in pred_vectors for item in sublist]

# 确保 true_labels_flat 和 pred_labels_flat 的长度相同
assert len(true_labels_flat) == len(pred_labels_flat), "Lengths of true_labels_flat and pred_labels_flat do not match!"

# 计算评估指标
precision = precision_score(true_labels_flat, pred_labels_flat, average='macro', zero_division=0)
recall = recall_score(true_labels_flat, pred_labels_flat, average='macro', zero_division=0)
f1 = f1_score(true_labels_flat, pred_labels_flat, average='macro', zero_division=0)

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1:.4f}')
