import glob
import json
from textattack.shared.attacked_text import AttackedText


start_word = ['\"\"\"', '\'\'\'']
stop_word = ["\n    Example", '\n    For example', '\n    for example', '\n    Examples', '\n    Note','\n    >>>',  "\n    example"]

result_dict = {}

all_files = glob.glob("/home/weimin/CodeT5/CodeT5+/humaneval/preds/codet5p-770m-py_T0.2_N20_all/*.json")
for i, file in enumerate(all_files):
    data  = json.load(open(file, 'r'))
    transform = data['transform']
    start = transform[0]
    programs = data['data']
    cnt = 0
    for program in programs:
        prompt = program['original_prompt']
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
        cnt+=tokens.count(start)  
    result_dict[i] = cnt

json.dump(result_dict, open("key_transform.json",'w'))      