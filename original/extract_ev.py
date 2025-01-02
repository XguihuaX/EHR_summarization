import time
import json
import csv
from skr_web_api import Submission
from tqdm import tqdm
from datetime import datetime
import os

# MetaMap API 账号信息
email = 'muxuechi'
apikey = '354ae295-85aa-4d8c-a217-c094c1a32feb'

# 定义MetaMap调用函数
def extract_entities_with_metamap(text, email, apikey):
    inst = Submission(email, apikey)
    inst.init_mm_interactive(text)
    response = inst.submit()

    if response.status_code != 200:
        raise Exception(f"MetaMap API request failed with status code {response.status_code}")

    content = response.content.decode()
    return parse_metamap_output(content)

def parse_metamap_output(content):
    entities = []
    for line in content.splitlines():
        if line.startswith("USER|MMI|"):
            parts = line.split('|')
            entity = {
                'preferred_name': parts[3],
                'semantic_types': parts[5].strip('[]').split(',')  # 去除方括号并转换为列表
            }
            entities.append(entity)
    return entities

# 确保输出目录存在
def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# 输入文件路径
input_file_path = '/ihome/hdaqing/abg96/llm/EHR_notes_hpi_annotation_updated_1.28.json'
output_directory = '/ihome/hdaqing/jul230/program/result/X'

# 确保输出目录存在
ensure_directory_exists(output_directory)

# 定义MetaMap工具名称和输出标签
tool = "metamap"
labels = ['Onset', 'Region', 'Quality', 'Severity', 'Time of symptom', 'Provocation']
entity_cat = [
    ['tmco', 'time'],  # onset
    ['spco', 'blor'],  # region/radiation
    ['qlco', 'sosy'],  # quality
    ['fndg', 'qnco','inpr'],  # severity
    ['tmco', 'freq', 'dura'],  # time of symptom
    ['acty', 'fndg','phpr']  # provocation
]

for label, entity_types in zip(labels, entity_cat):
    output_file_path = f'{output_directory}/{tool}_extracted_results_{label}.csv'

    with open(input_file_path, 'r') as json_file:
        data = json.load(json_file)

    # 初始化必要参数
    column_names = ['text', 'true', 'pred', 'extraction reasoning']

    with open(output_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(column_names)

        for i in tqdm(data):
            hpi = i['hpi']
            annotations = i['annotation']['HPI']
            entity_texts = []

            # 遍历所有注释，找到所有匹配的注释
            for annotation in annotations:
                if label in annotation['labels']:  # 匹配指定的原始标签
                    entity_texts.append(annotation['text'])

            # 处理找到的所有匹配的注释
            for entity_text in entity_texts:
                try:
                    results = extract_entities_with_metamap(hpi, email, apikey)
                    matched_entities = [res['preferred_name'] for res in results if any(sem_type in res['semantic_types'] for sem_type in entity_types)]

                    # 将每个匹配的实体单独写入CSV
                    for entity in matched_entities:
                        writer.writerow([hpi, entity_text, entity, f'MetaMap extracted this entity ({label})'])
                except Exception as e:
                    writer.writerow([hpi, entity_text, '', f'Error: {str(e)}'])
