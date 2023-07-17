import json
# from transformation import WordTransformationWithLogits
import numpy as np
from human_eval.data import write_jsonl, read_problems, stream_jsonl

# word_transformation = WordTransformationWithLogits(map_path = "/home/weimin/CodeT5/CodeT5+/paraphrase/tokens_map.json")

# data = json.load(open("/home/weimin/CodeT5/CodeT5+/paraphrase/augmented_prompt.json", 'r'))

# final_dict = {}

# for item in data:
#     probability = word_transformation._calculate_probability(item['augmented_prompt'])
#     final_dict[item['task_id']] = probability
    
# json.dump(final_dict, open("augmented_prompt_logits.json", 'w'))

def cal_pass_rate(li, ks = [1, 5, 10, 50, 100, 200]):
    n = len(li)
    c = 0
    for item in li:
        if item['passed']:
            c+=1
    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    pass_rate = {}
    for k in ks:
        pass_rate[f'pass@{k}'] = estimator(n, c, k)
    
    return pass_rate

logits = json.load(open("augmented_prompt_logits.json", 'r'))

data_path = "/home/weimin/CodeT5/CodeT5+/all_completion.jsonl_results.jsonl"
data = open(data_path, 'r').readlines()
data = [json.loads(item) for item in data]

sorts_key = {}
for key in logits:
    sorted_ = np.argsort(-np.array(logits[key]))
    sorts_key[key] = {}
    sorts_key[key]['top'] = sorted_[0].item()+1
    sorts_key[key]['bottom'] = sorted_[-1].item()+1

# all_completion = []
# for item in data:
#     task_id = item['task_id']
#     prompt_id = item['prompt_id']
#     if prompt_id==sorts_key[task_id]['top'] or prompt_id==sorts_key[task_id]['bottom']:
#         all_completion.append(item)

# print(sorts_key)
# json.dump(sorts_key, open("sorts_key.json", 'w'))
# write_jsonl("all_completion.jsonl", all_completion)

task_results = {}
for key in logits:
    task_results[key] = {}
    task_results[key]['top'] = []
    task_results[key]['bottom'] = []
    
for item in data:
    task_id = item['task_id']
    prompt_id = item['prompt_id']
    if prompt_id==sorts_key[task_id]['top']:
        task_results[task_id]['top'].append(item)
    elif prompt_id==sorts_key[task_id]['bottom']:
        task_results[task_id]['bottom'].append(item)
    
for key in logits:
    task_results[key]['top'] = {"pass_rate": cal_pass_rate(task_results[key]['top']), "prob":logits[key][sorts_key[key]['top']-1]}
    task_results[key]['bottom'] = {"pass_rate": cal_pass_rate(task_results[key]['bottom']), "prob":logits[key][sorts_key[key]['bottom']-1]}

json.dump(task_results, open("task_results.json", 'w'))