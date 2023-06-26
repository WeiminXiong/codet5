from textattack.transformations import WordSwapEmbedding
from textattack.shared.utils import words_from_text
from textattack.shared.attacked_text import AttackedText
import json
from tqdm import tqdm
from copy import deepcopy

embedding = WordSwapEmbedding()

datas = open("/home/weimin/CodeT5/CodeT5+/human-eval-v2-20210705.jsonl", 'r').readlines()
map_tokens = json.load(open("/home/weimin/CodeT5/CodeT5+/new_map.json", 'r'))


start_word = ['\"\"\"', '\'\'\'']
stop_word = ["\n    Example", '\n    For example', '\n    for example', '\n    Examples', '\n    Note','\n    >>>',  "\n    example"]
# result_tokens = set()

replaced_prompt = []

data = [json.loads(item) for item in datas]
for item in tqdm(data):
    cur_dict = {}
    prompt = item['prompt']
    strip_prompt = prompt.strip()
    function_name = strip_prompt[4: strip_prompt.find('(')]
    start_index = None
    for s_word in start_word:
        if s_word in prompt:
            start_index = prompt.find(s_word)+3
            break
    end_index = None
    for s_word in stop_word:
        if s_word in prompt:
            end_index = prompt.find(s_word)
            break
    if end_index==None:
        end_index = prompt.find(function_name)
        if end_index<start_index:
            end_index = None
    content = prompt[start_index: end_index]
    # print(content)
    # print("-------------------------------------------------------------------------")
    text = AttackedText(content)
    tokens = text.words
    cur_set = set()
    
    for token in tokens:
        if token in map_tokens:
            cur_set.add(token)
    
    new_text = deepcopy(text)
    
    for to_replace_token in cur_set:
        for idx, token in enumerate(tokens):
            if token ==to_replace_token:
                new_text = new_text.replace_word_at_index(idx, map_tokens[token])
    
    cur_dict['task_id'] = item['task_id']
    cur_dict['original_prompt'] = prompt
    if prompt.count("def")>=2:
        cur_dict['repalced_tokens'] = []
        cur_dict['augmented_prompt'] = [prompt]
    else:
        cur_dict['repalced_tokens'] = list(cur_set)
        cur_dict['augmented_prompt'] = [prompt[:start_index]+new_text.text+prompt[end_index:]]
    replaced_prompt.append(cur_dict)
    
    # result_tokens.update(tokens)
    
json.dump(replaced_prompt, open('best_prompt.json','w'))