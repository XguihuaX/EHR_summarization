import time
import json
import csv
from skr_web_api import Submission
from tqdm import tqdm
from datetime import datetime

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

# 让用户输入工具名称和实体名称
tool = input("请输入工具名称（如metamap、spacy等）: ")


# 定义要匹配的MetaMap语义类型列表
semantic_types_list = ['tmco', 'spco', 'qlco', 'fndg', 'sosy', 'acty', 'phpr']
semantic_type_to_extract = input(f"input the entity you want to annotate（such as {', '.join(semantic_types_list)}）: ")

input_file_path = f'/ihome/hdaqing/jul230/program/code/original/EHR_notes_hpi_annotation_meta.json'
output_file_path = f'/ihome/hdaqing/jul230/program/result/tools/{tool}_extracted_results_{semantic_type_to_extract}.csv'

with open(input_file_path, 'r') as json_file:
    data = json.load(json_file)

# 初始化必要参数
column_names = ['text', 'true', 'pred', 'extraction reasoning']

# 获取当前日期时间
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

start_time = time.time()

with open(output_file_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(column_names)
    for i in tqdm(data):
        hpi = i['hpi']
        annotations = i['annotation']['HPI']
        entity_texts = []

        # 遍历所有注释，找到所有匹配的注释
        for annotation in annotations:
            if semantic_type_to_extract in annotation['labels']:  # 匹配指定的MetaMap语义类型
                entity_texts.append(annotation['text'])

        # 处理找到的所有匹配的注释
        for entity_text in entity_texts:
            try:
                results = extract_entities_with_metamap(hpi, email, apikey)
                pred_entities = []
                all_results = [res['preferred_name'] for res in results]  # 未过滤的所有结果

                for res in results:
                    if semantic_type_to_extract in res['semantic_types']:
                        pred_entities.append(res['preferred_name'])

                pred_sentence = ', '.join(pred_entities)  # 将实体列表转换为完整的句子

                if pred_sentence:
                    writer.writerow([hpi, entity_text, pred_sentence, f'MetaMap extracted these entities ({semantic_type_to_extract})'])
                else:
                    writer.writerow([hpi, entity_text, '', 'No relevant entities extracted'])
            except Exception as e:
                writer.writerow([hpi, entity_text, '', f'Error: {str(e)}'])

end_time = time.time()
execution_time = end_time - start_time
print(f"\nExecution time: {execution_time:.2f} seconds")
