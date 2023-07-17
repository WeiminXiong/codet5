import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm

device = "cuda:0"

tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5p-770m-py')
model = AutoModelForSeq2SeqLM.from_pretrained('Salesforce/codet5p-770m-py').to(device)


lines = open("/home/weimin/CodeT5/CodeT5+/humaneval/all_prompt_logits_gpt_slim.jsonl_results.jsonl", 'r').readlines()
data = [json.loads(item) for item in lines]

lines = open("/home/weimin/CodeT5/CodeT5+/humaneval/all_prompt_logits_gpt.jsonl_results.jsonl", 'r').readlines()
secret_data = [json.loads(item) for item in lines]

public_pass_secret_pass = 0
public_pass_secret_not_pass = 0
secret_pass_public_not_pass = 0

# for item, secret_item in zip(data, secret_data):
#     if item['passed'] and secret_item['passed']:
#         public_pass_secret_pass+=1
#     if item['passed'] and not secret_item['passed']:
#         public_pass_secret_not_pass+=1
#     if not item['passed'] and secret_item['passed']:
#         secret_pass_public_not_pass+=1

# print("public_pass_secret_pass", public_pass_secret_pass)
# print("public_pass_secret_not_pass", public_pass_secret_not_pass)
# print("secret_pass_public_not_pass", secret_pass_public_not_pass)

probility_data = json.load(open("/home/weimin/CodeT5/CodeT5+/humaneval/all_prompt_logits_results_gpt.json", 'r'))
# probility_data = secret_data


task_ids = set()
for item in data:
    task_ids.add(item['task_id'])
    
results = {}
completion_ids = list(range(1, 21))
for task_id in task_ids:
    results[task_id] = {}
    results[task_id]['oracle_completion'] = []
    results[task_id]['completion'] = {}
    for completion_id in completion_ids:
        results[task_id]['completion'][completion_id] = []


def cal_probility(probility, li):
    n = 0
    p = 0
    all_probility = 0
    for a, b in zip(probility, li):
        # if b['passed']:
        #     n+=1
        #     all_probility+=a
        # n+=1
        # all_probility+=a
        n+=1
        if b['passed']:
            p+=1
    if n==0:
        return -1000
    else:
        return p/n
            

# for item in data:
#     if item['prompt_id'] == 0:
#         results[item['task_id']]['oracle_completion'].append(item)
#     if item['prompt_id'] == 1:
#         results[item['task_id']]['completion'].append(item)
for item in data:
    if item['prompt_id'] == 0:
        results[item['task_id']]['oracle_completion'].append(item)
    elif item['prompt_id'] in completion_ids:
        results[item['task_id']]['completion'][item['prompt_id']].append(item)

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

final_results = {}
# for task_id in results:
#     final_results[task_id] = {}
#     final_results[task_id]['original_pass_rate'] = cal_pass_rate(results[task_id]['oracle_completion'])
#     final_results[task_id]['pass_rate'] = cal_pass_rate(results[task_id]['completion'])
    
# original_pass_rate_1 = np.mean([final_results[task_id]['original_pass_rate']['pass@1'] for task_id in final_results])
# augmented_pass_rate_1 = np.mean([final_results[task_id]['pass_rate']['pass@1'] for task_id in final_results])

# print("original_pass_rate_1", original_pass_rate_1)
# print("augmented_pass_rate_1", augmented_pass_rate_1)

for task_id in tqdm(results):
    final_results[task_id] = {}
    final_results[task_id]['original_pass_rate'] = probility_data[task_id]['original_pass_rate']
    final_results[task_id]['oracle_probility'] = cal_probility(probility_data[task_id]['oracle_probility'], results[task_id]['oracle_completion'])
    final_results[task_id]['pass_rate'] = {}
    final_results[task_id]['probility'] = {}
    for completion_id in completion_ids:
        final_results[task_id]['pass_rate'][completion_id] = probility_data[task_id]['pass_rate'][str(completion_id)]
        final_results[task_id]['probility'][completion_id] = cal_probility(probility_data[task_id]['probility'][str(completion_id)], results[task_id]['completion'][completion_id])

original_pass_rate_1 = np.mean([final_results[task_id]['original_pass_rate']['pass@1'] for task_id in final_results])
augmented_pass_rate_1 = []

for task_id in final_results:
    max_prob = -1000
    pass_1 = 0
    for completion_id in final_results[task_id]['probility']:
        if final_results[task_id]['probility'][completion_id] > max_prob:
            max_prob = final_results[task_id]['probility'][completion_id]
            pass_1 = final_results[task_id]['pass_rate'][completion_id]['pass@1']
    if final_results[task_id]['oracle_probility'] > max_prob:
        max_prob = final_results[task_id]['oracle_probility']
        pass_1 = final_results[task_id]['original_pass_rate']['pass@1']
    augmented_pass_rate_1.append(pass_1)
augmented_pass_rate_1 = np.mean(augmented_pass_rate_1)

print("original_pass_rate_1", original_pass_rate_1)
print("augmented_pass_rate_1", augmented_pass_rate_1)

json.dump(final_results, open("/home/weimin/CodeT5/CodeT5+/humaneval/all_prompt_logits_results_sorted.json", 'w'))