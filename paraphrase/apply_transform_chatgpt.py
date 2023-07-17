import json
from tqdm import tqdm
from copy import deepcopy
from transformation import WordTransformationWithLogits
import argparse
import requests

def send_request(prompt):
    dict = {'model':"gpt-3.5-turbo",
        'messages' : [{'role': 'user', 'content':prompt}],
        'top_p' : 0.9,
        'max_tokens':3000,
        'frequency_penalty':0,
        'presence_penalty':0,}

    headers = {
        'Authorization': 'Bearer sb-ed07016f987c6bb701b74fcf399c56d067b5a5c8bc3ad177',
        'Content-Type': 'application/json',
    }

    data = json.dumps(dict)
    while True:
        try:
            response = requests.post('https://api.openai-sb.com/v1/chat/completions', headers=headers, data=data)
        except Exception:
            continue
        else:
            break
    return response

gpt_prompt = "Given the instruction below:\n\
```{}```\n\
Please help me generate 20 paraphrases and do not change its original form and keep its function name and parameters and the test case using the format: \n\
sentence: \n\
sentence: \n\
... \n\
sentence: \n\
Remember do not change the example case and the function name and keep its original form"

parser = argparse.ArgumentParser()
parser.add_argument("--start_index", type=int, default=0)
parser.add_argument("--end_index", type=int, default=0)
parser.add_argument("--thread_id", type=int, default=0)
args = parser.parse_args()

def get_boundary_index(text):
    start_word = ['\"\"\"', '\'\'\'']
    stop_word = ["\n    Example", '\n    For example', '\n    for example', '\n    Examples', '\n    Note','\n    >>>',  "\n    example", "\n     Example"] 
    strip_text = text.strip()
    function_name = strip_text[4: strip_text.find('(')]
    function_name = '\n    ' + function_name
    start_index = -1
    end_index = -1
    for s_word in start_word:
        if s_word in text:
            start_index = text.find(s_word)+3
            break
    for e_word in stop_word:
        if e_word in text:
            end_index = text.find(e_word)
            break
    if end_index==-1:
        end_index = text.find(function_name, 25)
    return (start_index, end_index)

def augment_prompt(prompt, count):
    start_index, end_index = get_boundary_index(prompt)
    content = prompt[start_index:end_index]
    gpt_prompt_instance = gpt_prompt.format(content)
    cur_num = 0
    all_prompt = []
    while cur_num<count:
        response = send_request(prompt=gpt_prompt_instance)
        while True:
            if response.status_code==200:
                result = json.loads(response.text)
                if 'choices' not in result.keys():
                    response = send_request(prompt = gpt_prompt_instance)
                    continue
                else:
                    predict = result['choices'][0]['message']['content']
                    break
            else:
                response = send_request(prompt=gpt_prompt_instance)
        
        all_content = response.json()['choices'][0]['message']['content'].split('\n')
        augmented_prompts = []
        for content in all_content:
            if content.startswith('sentence:'):
                augmented_prompt = prompt[:start_index]+content[content.find('sentence')+10:]+prompt[end_index:]
                augmented_prompts.append(augmented_prompt)
            elif content.find(". ")<10:
                augmented_prompt = prompt[:start_index]+content[content.find(".")+2:]+prompt[end_index:]
                augmented_prompts.append(augmented_prompt)
        to_collect_num = min(count-cur_num, len(augmented_prompts))
        all_prompt.extend(augmented_prompts[:to_collect_num])
        cur_num += to_collect_num
           
    return all_prompt

datas = open("/home/weimin/CodeT5/CodeT5+/paraphrase/human-eval-v2-20210705.jsonl", 'r').readlines()
data = [json.loads(item) for item in datas]

map_tokens = json.load(open("/home/weimin/CodeT5/CodeT5+/paraphrase/tokens_map.json", 'r'))

all_transform = []
for key in map_tokens:
    target_tokens = map_tokens[key]
    for t in target_tokens:
        all_transform.append((key, t))
all_transform_num  = len(all_transform)

final_list = []

map_tokens_path = "/home/weimin/CodeT5/CodeT5+/paraphrase/tokens_map.json"
transformation = WordTransformationWithLogits(map_path=map_tokens_path)
augmented_list = []

augmented_list = json.load(open('/home/weimin/CodeT5/CodeT5+/paraphrase/augmented_prompt_original.json', 'r'))

augmented_keys = set()
for item in augmented_list:
    augmented_keys.add(item['task_id'])


for item in tqdm(data):
    prompt = item['prompt']
    if item['task_id'] in augmented_keys:
        continue
    cur_dict = {}
    cur_dict['task_id'] = item['task_id']
    cur_dict['original_prompt'] = prompt
    cur_dict['augmented_prompt'] = []
    
    if prompt.count("def ")<2:
        augmented_prompt = augment_prompt(prompt, 20)
    else:
        def_index = prompt.find("def ", 10)
        former_prompt = prompt[:def_index]
        augmented_prompt = augment_prompt(prompt[def_index:], 20)
        augmented_prompt = [former_prompt+item for item in augmented_prompt]
    cur_dict['augmented_prompt'] = augmented_prompt
    augmented_list.append(cur_dict)
    json.dump(augmented_list, open("/home/weimin/CodeT5/CodeT5+/paraphrase/augmented_prompt.json", 'w'))
        
json.dump(augmented_list, open("/home/weimin/CodeT5/CodeT5+/paraphrase/augmented_prompt.json", 'w'))