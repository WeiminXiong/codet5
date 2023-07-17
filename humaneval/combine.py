dir_path = "/home/weimin/CodeT5/CodeT5+/humaneval/preds/codet5p-770m-py_T0.2_N20_all"
import glob
import json
from human_eval.data import read_problems, write_jsonl, stream_jsonl
output = []

all_files = glob.glob(dir_path + "/*.json")
for file in all_files:
    data  = json.load(open(file, 'r'))
    count = data['count']
    program = data['data']
    for item in program:
        if item['completion'] != None:
            completion = item['completion']
            for temp in completion:
                temp['prompt_id'] = count
                temp['transform'] = item['transform']
                # print(item)
                temp['replaced_count'] = item['repalced_count']
            output.extend(completion)

write_jsonl( "all_prompt.jsonl", output)