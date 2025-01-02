import os
import json
import csv
import spacy
import scispacy
from tqdm import tqdm
import re
from scispacy.abbreviation import AbbreviationDetector

# 加载SciSpacy模型
nlp = spacy.load("en_core_sci_sm")
nlp.add_pipe("abbreviation_detector")

# 确保输出目录存在
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# 定义实体提取函数
def extract_entities(text, label):
    doc = nlp(text)
    entities = []

    if label == "Region":
        pattern = re.compile(r'\b(?:left|right|mid|lower|upper|side|center|sub|retro)\s?(?:sternal|chest|region|side|part|area|cp|epigastric)\b|\b(?:epigastric|retrosternal|midsternal|substernal|abdomen|head|arm|leg|back|neck|shoulder|foot|hand|ankle|elbow|knee|hip|wrist|forearm|thigh|calf|lumbar|cervical|thoracic)\b', re.IGNORECASE)
    elif label == "Radiation":
        pattern = re.compile(r'\b(?:radiating|radiates|radiated|extends|extended|extending|spreads|spread|spreading)\s?(?:to|into|through|towards|from|down|up|across)?\s?(?:\w+\s){0,3}?(?:chest|arm|leg|back|neck|shoulder|foot|hand|abdomen|head|elbow|knee|hip|wrist|forearm|thigh|calf|lumbar|cervical|thoracic)\b', re.IGNORECASE)
    elif label == "Severity":
        pattern = re.compile(r'\b(\d{1,2})\s?\/\s?10|(\d{1,2})\s?out\s?of\s?10|(\d{1,2})\s?over\s?10\b', re.IGNORECASE)
    elif label == "Onset":
        pattern = re.compile(r'(\b(?:onset|start|begin|started|began|initiated)\b.{0,30}?\b(?:\d{1,2}:\d{2}\s?(?:am|pm)|\d{1,2}\s?(?:hours?|hrs?|days?|weeks?|months?|years?)|today|this\s?\w+|yesterday|last\s?\w+|several\s?\w+|few\s?\w+|about\s?\d{1,2}|\d{1,2}\s?or\s?\d{1,2}))', re.IGNORECASE)
    elif label == "Provocation":
        pattern = re.compile(r'\b(?:worse|better|exacerbated|relieved|aggravated|improved)\s?(?:with|by|on|during)?\s?(?:\w+\s){0,3}?(?:exertion|movement|inspiration|breathing|eating|ambulation|rest|activity)\b', re.IGNORECASE)
    elif label == "Quality":
        pattern = re.compile(r'\b(?:sharp|dull|pressure|tightness|burning|achy|crushing|stabbing|throbbing|radiating|pounding|piercing|cramping|gnawing|nagging|soreness|tenderness|shooting|twinge)\b', re.IGNORECASE)
    else:
        pattern = None

    if pattern:
        matches = pattern.findall(text)
        for match in matches:
            if isinstance(match, tuple):
                entities.extend([m for m in match if m])
            else:
                entities.append(match)

    return entities

# 输入文件路径
input_file_path = '/ihome/hdaqing/abg96/llm/EHR_notes_hpi_annotation_updated_1.28.json'
output_directory = '/ihome/hdaqing/jul230/program/result/tools'

# 确保输出目录存在
ensure_directory_exists(output_directory)

# 读取JSON文件
with open(input_file_path, 'r') as json_file:
    data = json.load(json_file)

# 定义工具名称和输出标签
tool = "scispacy"
labels = ['Onset', 'Region', 'Radiation', 'Quality', 'Severity', 'Provocation']

for label in labels:
    output_file_path = f'{output_directory}/{tool}_extracted_results_{label}_pattern.csv'

    # 初始化CSV文件
    column_names = ['text', 'true', 'pred', 'extraction reasoning']
    with open(output_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(column_names)

        # 遍历数据并提取实体
        for i in tqdm(data):
            hpi = i['hpi']
            annotations = i['annotation']['HPI']
            entity_texts = []

            # 遍历所有注释，找到所有匹配的注释
            for annotation in annotations:
                if label.lower() in (lbl.lower() for lbl in annotation['labels']):  # 匹配指定的原始标签
                    entity_texts.append(annotation['text'])

            # 处理找到的所有匹配的注释
            for entity_text in entity_texts:
                try:
                    results = extract_entities(hpi, label)
                    pred_sentence = ', '.join(results)  # 将实体列表转换为完整的句子

                    if pred_sentence:
                        writer.writerow(
                            [hpi, entity_text, pred_sentence, f'SciSpacy extracted these entities ({label})'])
                    else:
                        writer.writerow([hpi, entity_text, '', 'No relevant entities extracted'])
                    except Exception as e:
                    writer.writerow([hpi, entity_text, '', f'Error: {str(e)}'])
