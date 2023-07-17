dir_path = "/home/weimin/CodeT5/CodeT5+/humaneval/preds/codet5p-770m-py_T0.2_N100_logits_replace"
import glob
import json
from human_eval.data import read_problems, write_jsonl, stream_jsonl
from collections import Counter
output = []

all_files = glob.glob(dir_path + "/*.json")
for file in all_files:
    counter = Counter()
    data  = json.load(open(file, 'r'))
    for item in data['oracle_completion']:
        item['prompt_id'] = 0
        output.append(item)
    for i, li in enumerate(data['completion']):
        for item in li:
            item['prompt_id'] = i+1
            counter[i+1]+=1
            if counter[i+1] > 100:
                break
            output.append(item)
        

write_jsonl("all_prompt_logits_replace.jsonl", output)