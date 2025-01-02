import torch
import transformers
import re
import json
import textwrap
import langchain
import torch
import csv
from tqdm import tqdm
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaTokenizerFast
from accelerate import infer_auto_device_map, init_empty_weights
from accelerate import load_checkpoint_and_dispatch
from transformers import LlamaModel, LlamaConfig
from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import pipeline
from datetime import datetime

access_token = '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'  # 替换成自己的huggin face API的令牌

cache_dir = '/zfs1/hdaqing/abg96/cache'
# 加载模型
# 使用了加速器模块来分配模型层到不同的GPU上以提高计算效率，同时初始化了模型和分词器

config = LlamaConfig.from_pretrained("meta-llama/Llama-2-13b-chat-hf", cache_dir=cache_dir, torch_dtype=torch.float16,
                                     token=access_token)
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)

model.tie_weights()
device_map = infer_auto_device_map(model, no_split_module_classes=["LlamaDecoderLayer"])

# print(device_map, '\n') # print device map to view which layers are on which gpu.

model = transformers.LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-13b-chat-hf", torch_dtype=torch.float16,
                                                      cache_dir=cache_dir, device_map=device_map,
                                                      offload_folder="offload", offload_state_dict=True,
                                                      token=access_token)

device_map = infer_auto_device_map(model)

tokenizer = LlamaTokenizerFast.from_pretrained("meta-llama/Llama-2-13b-chat-hf", unk_token="<unk>", bos_token="<s>",
                                               eos_token="</s>", cache_dir=cache_dir, token=access_token)


# 定义了分词器
class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = [stop.to("cuda") for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False


stop_words = ["</s>"]
stop_words_ids = [tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze() for stop_word in stop_words]

stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(
    stops=[torch.Tensor([669])])])  # replace with index of the </s> token for your tokeniser.，要替换成tokeniser中的</s>

# 定义文本生成管道
pipe = pipeline("text-generation",
                model=model,
                tokenizer=tokenizer,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                # automatically infers and distributes the layers of the model on multiple GPUs according to availability
                max_new_tokens=2000,  # controls how many tokens to generate as the output
                do_sample=True,
                top_k=30,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria
                )

# 定义提示模版change entity type depending on the prompt
entity = 'Onset'
sys_prom = f'''

You are a chatbot who knows the task of 'Named Entity recognition'. You are provided with some clinical data related to 
a patients visit to the emergency department (ED) of a hospital. The target task is to extract the the words that belong 
to the entity {entity}. For that we will first extract words related to the entity 'Chief Complaint' and then based on 
that we will extract the words related '{entity}'. I will now give you the definition of both entities.
'Chief complaint' is the main issue/problem that the patient presents to the ED with.
'{entity}'  means the beginning or initiation of a symptom, sign, or medical condition i.e the chief complaint (extracted before).



You have to make sure that you follow the definition I provide and not interpret the meaning of '{entity}' in a general way. 
So I will be providing with you some examples on how to extract the entities 'Chief Complaint' and '{entity}. Basically
we will be trying to mimic a physicians train of thought on how to extract these entities. 
We will be first looking for words that usually mark the starting of phrases related to 'Chief Complaint'. 
For example, 'presents with', 'complains of', 'comes to/in', etc and mostly the actual chief complaint follows 
these words. 
For {entity}  we will be looking temporal markers. For example, 'several days/hours ago',  'today/yesterday', etc. 


Putting this down into steps:

1) Identify the verb tense that indicates the chief complaint.
2) Identify the specific phrase that describes the chief problem.
3) Identify the adverbial phrase for time that indicates the start of the chief complaint.
4) Based on step 3, should you give an output? Answer yes or no 
5) Provide the final phrase. 

'''

few_shot = f'''

Now I will provide you with an example on how to extract the entity 'Chief Complaint' and then the entity  {entity}

Example1: This is a 73-year-old male comes in for evaluation of chest pain. he states the chest pain started about 7 or 7:30 this evening was off and on got progressively worse. he took some aspirin is morphine tablet and nitroglycerin at home with some relief of the chest pain but was still having its recall the ambulance committed for evaluation. he also had some mild shortness of breath. denied any nausea or vomiting. denied any diaphoresis. no lightheadedness or dizziness. denies any syncope or presyncope. no other complaints at this time. symptoms are moderate in severity. nothing is made this worse the nitroglycerin aspirated did seem to help somewhat

Reasoning steps:
1) Identify the verb tense that indicates the chief complaint.- comes in for evaluation of chest pain 
2) Identify the specific phrase that describes the chief problem. - chest pain 
3) Identify the adverbial phrase for time that indicates the start of the chief complaint. -  chest pain started about 7 or 7:30 this evening  
4) Based on step 3, should you give an output? Answer yes or no  - yes
5) Provide the final phrase.  - @about 7 or 7:30 this evening@

Now I will provide you with an example where the Chief Complaint is  present and but the entity {entity}  is not present. 

Example2: 39-year-old white female presents with chest tightness. she relates that she was initially seen at a medics present was concern for blood clot so they sent her here. she relates she has a history of pulmonary embolus in her family. she has pain in her back at times. she denies any shortness of breath. she's had no syncope. she's had no recent illnesses.


1) Identify the verb tense that indicates the chief complaint.- presents with chest tightness
2) Identify the specific phrase that describes the chief problem. - chest tightness
3) Identify the adverbial phrase for time that indicates the start of the chief complaint. - not provided
4) Based on step 3, should you give an output? Answer yes or no  - no
5) Provide the final phrase.  - @ @


Now I will provide you with an example where the Chief Complaint is not present and as a result the  entity  {entity} is not present as well. 

Example3: Medications: per nursing records .

1)Identify the verb phrase that indicates the chief complaint. -  not provided
2) Provide the final phrase - @ @

'''

instruct = '''

Now for the following text using the previous conversation as context can you extract the entity 'Onset' in a 
similar manner i.e by following the same reasoning steps described above, Don't forget to add the '@' tag before the final answer and the '@' tag after the final answer 
to indicate the final answer.
Text: {text} 

'''

# 定义提示模版的必要令牌
B_SEN, E_SEN = '<s>', '</s>'
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n"


# 构建提示模版函数
def get_prompt():
    SYSTEM_PROMPT = B_SYS + sys_prom + E_SYS + few_shot + '\n'
    prompt_template = B_SEN + B_INST + SYSTEM_PROMPT + instruct + E_INST
    return prompt_template


# 读取json数据
with open('/ihome/hdaqing/abg96/llm/EHR_notes_hpi_annotation_updated_1.28.json', 'r') as json_file:
    data = json.load(json_file)

# 初始化必要参数
template = get_prompt()

column_names = ['text', 'true', 'pred', 'reasoning']
temp_value = 1.0
top_p_value = 0.95


# 定义辅助函数
def remove_phrase(text, phrase):
    return text.replace(phrase, '').strip()


start_time = time.time()


def parse_output(text):
    pattern = r'@[^@]*@|@\.|@\"|@\''
    matches = re.findall(pattern, text)
    return matches if matches else 'Cannot parse string'


# 设置huggingface管道

llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature': temp_value, 'top_p': top_p_value})

# 处理数据并生成输出
output_directory = '/ihome/hdaqing/jul230/program/result/test'
filename = f'extracted_{entity}.csv'
with open(filename, 'w', newline='') as file:  # would recommend using batches for inferece to speed up experimentation.
    writer = csv.writer(file)
    writer.writerow(column_names)
    for i in data:
        hpi = i['hpi']

        annotations = i['annotation']['HPI']

        target_label = entity

        entity_texts = []

        flag = 0
        for annotation in annotations:
            if entity in annotation['labels']:
                flag = 1
                entity_texts.append(annotation['text'])

        if flag == 0:
            entity_texts.append('')

        for entity_text in entity_texts:
            question = hpi

            prompt = PromptTemplate(template=template, input_variables=["text"])
            llm_chain = LLMChain(prompt=prompt, llm=llm)

            output = llm_chain.run(text=question)

            results = parse_output(output)

            if results != 'Cannot parse string':  # found matches for outputs.
                writer.writerow([hpi, entity_text, results[-1].replace('@', ''), output])

            else:
                times = 0
                while (times < 5):  # check 5 times if we can get an output.
                    output = llm_chain.run(text=question)
                    results = parse_output(output)

                    if results != 'Cannot parse string':
                        writer.writerow([hpi, entity_text, results[-1].replace('@', ''), output])
                        break

                    times = times + 1

                else:
                    writer.writerow([hpi, entity_text, '', output])

# 计算时间

end_time = time.time()
execution_time = end_time - start_time
print(f"\nExecution time: {execution_time:.2f} seconds")