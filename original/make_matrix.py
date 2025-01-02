import csv
import pandas as pd

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import cohen_kappa_score

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


column_names = ['text','true','pred', 'annotation', 'rule_llm_annotation', 'evaluation reasoning', 'extraction reasoning'] # can change with custom column names



path_to_dir = ''
files = []


def convert(l):
    for i in range(len(l)):
        if l[i] == 'Similar':
            l[i] = 'Correct'
        if l[i] == 'Dissimilar':
            l[i] = 'Incorrect'
    return l


def generate_heatmap(pred, true, categories):
    matrix = np.zeros((4, 4))

    for t, p in zip(true, pred):
        if t in categories and p in categories:
            matrix[categories.index(t), categories.index(p)] += 1
    print('matrix:', matrix)
    return matrix


def plot_multiple_heatmaps(data, save_path):
    num_files = len(data)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10)) 
    categories = ['Correct', 'Incorrect', 'Missing', 'Spurious']

    # Ensure axes is always a 2D array
    if num_files == 1:
        axes = np.array([[axes]])

    for i, (pred, true, title) in enumerate(data):
        row = i // 3
        col = i % 3
        matrix = generate_heatmap(pred, true, categories)  
        ax = axes[row, col]

        sns.heatmap(matrix, annot=True, annot_kws={"color": "black", "weight": "bold", "size": 16}, xticklabels=categories, yticklabels=categories, cmap="YlOrRd", ax=ax)
        ax.set_xlabel('Predicted Category')
        ax.set_ylabel('True Category')
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()



data = []
for i in range(len(files)):
    df = pd.read_csv(path_to_dir + files[i], names = column_names)

    true = list(df['annotation']) # human annotation
    pred = list(df['rule_llm_annotation'])
  
    refined_true = [str(i).strip() for i in true]
    refined_pred = [str(i).strip() for i in pred]

    refined_true = convert(refined_true)
    refined_pred = convert(refined_pred)
    data.append((refined_pred, refined_true, files[i].split('_')[1]))

save_path = 'my_plot.png'  # Replace with your desired path
plot_multiple_heatmaps(data, save_path)

	





