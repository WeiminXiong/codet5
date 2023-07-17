from textattack.transformations import WordSwapEmbedding
from textattack.shared.utils import words_from_text
from textattack.shared.attacked_text import AttackedText
import json
from tqdm import tqdm
from copy import deepcopy

embedding = WordSwapEmbedding()

datas = open("/home/weimin/CodeT5/CodeT5+/human-eval-v2-20210705.jsonl", 'r').readlines()
map_tokens = json.load(open("/home/weimin/CodeT5/CodeT5+/new_list_map.json", 'r'))


final_list = []

start_word = ['\"\"\"', '\'\'\'']
stop_word = ["\n    Example", '\n    For example', '\n    for example', '\n    Examples', '\n    Note','\n    >>>',  "\n    example"]
# result_tokens = set()

replaced_prompt = []

cnt = 0
data = [json.loads(item) for item in datas]


for key, value in tqdm(map_tokens.items()):
    for t in value:
        cur_item = {}
        cur_item['count'] = cnt
        cnt+=1
        temp = (key, t)
        cur_item['transform'] = temp
        cur_item['data'] = []
        
        for item in data:
            cur_dict = {}
            prompt = item['prompt']
            strip_prompt = prompt.strip()
            function_name = strip_prompt[4: strip_prompt.find('(')]
            start_index = None
            for s_word in start_word:
                if s_word in prompt:
                    start_index = prompt.find(s_word)+3
                    break
            end_index = -1
            for s_word in stop_word:
                if s_word in prompt:
                    end_index = prompt.find(s_word)
                    break
            if end_index==-1:
                end_index = prompt.find(function_name, 15)
                if end_index<start_index:
                    end_index = -1
            content = prompt[start_index: end_index]
            # print(content)
            # print("-------------------------------------------------------------------------")
            text = AttackedText(content)
            tokens = text.words
            cur_set = set()
            
            for token in tokens:
                if token == temp[0]:
                    cur_set.add(token)
            
            new_text = deepcopy(text)
            
            # for to_replace_token in cur_set:
            for idx, token in enumerate(tokens):
                if token ==temp[0]:
                    new_text = new_text.replace_word_at_index(idx, temp[1])
            
            cur_dict['task_id'] = item['task_id']
            cur_dict['original_prompt'] = prompt
            if prompt.count("def")>=2:
                cur_dict['repalced_tokens'] = []
                cur_dict['augmented_prompt'] = [prompt]
            else:
                cur_dict['repalced_tokens'] = list(cur_set)
                cur_dict['augmented_prompt'] = [prompt[:start_index]+new_text.text+prompt[end_index:]]
            # replaced_prompt.append(cur_dict)
            cur_item['data'].append(cur_dict)
    
        final_list.append(cur_item)
    # result_tokens.update(tokens)
    
json.dump(final_list, open('all_prompt.json','w'))