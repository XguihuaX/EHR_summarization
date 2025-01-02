import jpype
import jpype.imports
from jpype.types import *
import json
import csv
import os
from tqdm import tqdm
from datetime import datetime
import glob

java_home = os.environ.get('JAVA_HOME')
if not java_home:
    raise EnvironmentError('JAVA_HOME environment variable is not set.')

# cTAKES 安装目录
ctakes_home = '/ihome/hdaqing/jul230/program/tool/ctakes_api/apache-ctakes-4.0.0'

# 设置 classpath
classpath = [
    f"{ctakes_home}/lib/*",  # 包含所有的JAR文件
    f"{ctakes_home}/resources"  # 包含资源文件
]

# 启动 JVM
jpype.startJVM(classpath=classpath)

# 导入需要的 cTAKES 类
from org.apache.ctakes.typesystem import TypeSystemDescription
from org.apache.ctakes.clinicalpipeline import ClinicalPipelineFactory
from org.apache.uima.jcas import JCas
from org.apache.ctakes.typesystem.type.textsem import IdentifiedAnnotation

# 定义 cTAKES 处理函数
def extract_entities_with_ctakes(text):
    # 创建 JCas 对象
    jcas = JCas()

    # 将文本放入 JCas
    jcas.setDocumentText(text)

    # 加载 cTAKES 管道
    pipeline = ClinicalPipelineFactory.getDefaultPipeline()

    # 处理文本
    pipeline.process(jcas)

    # 从 JCas 中提取实体
    entities = []
    for annotation in jcas.getAnnotationIndex():
        if isinstance(annotation, IdentifiedAnnotation):
            entities.append({
                'preferred_name': annotation.getCoveredText(),
                'semantic_types': [concept.getCode() for concept in annotation.getOntologyConceptArr()]
            })

    return entities

# 读取 JSON 数据
with open('/ihome/hdaqing/abg96/llm/EHR_notes_hpi_annotation_updated_1.28.json', 'r') as json_file:
    data = json.load(json_file)

# 初始化必要参数
column_names = ['text', 'true', 'pred', 'reasoning']
entity = 'Onset'
tool = 'ctakes'

# 获取当前日期时间
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# 处理数据并生成输出
output_directory = '/ihome/hdaqing/jul230/program/result/tools'
filename = f'{output_directory}/extracted_{current_time}_{tool}_{entity}.csv'
start_time = time.time()

with open(filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(column_names)
    for i in tqdm(data):
        hpi = i['hpi']
        annotations = i['annotation']['HPI']
        entity_texts = []

        flag = 0
        for annotation in annotations:
            if entity in annotation['labels']:
                flag = 1
                entity_texts.append(annotation['text'])

        if flag == 0:
            entity_texts.append('')

        for entity_text in entity_texts:
            try:
                results = extract_entities_with_ctakes(hpi)
                pred_entities = [res['preferred_name'] for res in results if entity in res['semantic_types']]
                if pred_entities:
                    writer.writerow([hpi, entity_text, pred_entities, 'cTAKES extracted these entities'])
                else:
                    writer.writerow([hpi, entity_text, '', 'No relevant entities extracted'])
            except Exception as e:
                writer.writerow([hpi, entity_text, '', f'Error: {str(e)}'])

end_time = time.time()
execution_time = end_time - start_time
print(f"\nExecution time: {execution_time:.2f} seconds")

# 关闭 JVM
jpype.shutdownJVM()
