from collections import Counter
import glob
import os
import json

path = "/home/weimin/CodeT5/CodeT5+/humaneval/preds/gpt3.5-turbo-without_thought"

files = glob.glob(path+"/*")
for file in files:
    data = open(file).readlines()
    item = json.loads(data[0])
    task_id = item['task_id']
    task_id = task_id.replace('/', '-')
    os.rename(file, path+f'/{task_id}.jsonl')