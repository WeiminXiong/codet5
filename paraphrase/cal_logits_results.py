import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm
import argparse
import fcntl

device = "cuda:0"

tokenizer = AutoTokenizer.from_pretrained('Salesforce/codet5p-770m-py')
model = AutoModelForSeq2SeqLM.from_pretrained('Salesforce/codet5p-770m-py').to(device)

parser = argparse.ArgumentParser()

parser.add_argument("--start_index", type=int, default=0)
parser.add_argument("--end_index", type=int, default=0)
parser.add_argument("--thread_id", type=int, default=0)
args = parser.parse_args()
@torch.no_grad()
def _calculate_probability(prompt, completions):
        batch_size = len(completions)
        start_tokens = [prompt]*batch_size
        # all_code = [prompt + completion for completion in completions]
        encodings = tokenizer(start_tokens, return_tensors='pt')
        encodings = {k: v.to(device) for k, v in encodings.items()}
        labels = tokenizer(completions, truncation=True, max_length=512, padding=True, return_tensors='pt')['input_ids'].to(device)
        
        outputs = model(**encodings, labels=labels)
        logits = outputs['logits'].detach()
        logits_softmax = torch.softmax(logits, dim=-1)
        
        labels_token_prob_list = [logits_softmax[i, range(labels.shape[-1]), labels[i, :]] for i in range(batch_size)]
        labels_token_prob_list = torch.stack(labels_token_prob_list)
        
        # labels[labels==2]=0
        # labels[labels==1]=0
        
        labels_token_prob_list[labels==0]=1
        labels_token_prob_list = torch.log(labels_token_prob_list)
        labels_token_prob_list = torch.sum(labels_token_prob_list, dim=-1)/torch.sum(labels!=0, dim=-1)
        
        return labels_token_prob_list.tolist()


lines = open("/home/weimin/CodeT5/CodeT5+/humaneval/all_prompt_logits_replace.jsonl_results.jsonl", 'r').readlines()
# for item in lines:
#     try:
#         json.loads(item)
#     except Exception:
#         print(item)
data = [json.loads(item) for item in lines]

prompts_ = json.load(open("/home/weimin/CodeT5/CodeT5+/paraphrase/augmented_prompt.json", 'r'))
prompts = {}
for item in prompts_:
    task_id = item['task_id']
    prompts[task_id] = item

task_ids = set()
for item in data:
    task_ids.add(item['task_id'])
    
results = {}
completion_ids = list(range(1, 21))
for task_id in task_ids:
    results[task_id] = {}
    results[task_id]['oracle_completion'] = []
    results[task_id]['original_prompt'] = prompts[task_id]['original_prompt']
    results[task_id]['completion'] = {}
    results[task_id]['completion_text'] = {}
    results[task_id]['oracle_completion_text'] = []
    results[task_id]['prompt'] = {}
    for completion_id in completion_ids:
        results[task_id]['completion'][completion_id] = []
        results[task_id]['completion_text'][completion_id] = []

for task_id in task_ids:
    for i, augmented_prompt in enumerate(prompts[task_id]['augmented_prompt']):
        results[task_id]['prompt'][i+1] = augmented_prompt

# for item in data:
#     if item['prompt_id'] == 0:
#         results[item['task_id']]['oracle_completion'].append(item)
#     if item['prompt_id'] == 1:
#         results[item['task_id']]['completion'].append(item)
for item in data:
    if item['prompt_id'] == 0:
        results[item['task_id']]['oracle_completion'].append(item)
        results[item['task_id']]['oracle_completion_text'].append(item['completion'])
    elif item['prompt_id'] in completion_ids:
        results[item['task_id']]['completion'][item['prompt_id']].append(item)
        results[item['task_id']]['completion_text'][item['prompt_id']].append(item['completion'])


def cal_pass_rate(li, ks = [1, 5, 10, 50, 100]):
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

for idx, task_id in enumerate(tqdm(results)):
    while True:
        try:
            all_results = json.load(open("/home/weimin/CodeT5/CodeT5+/humaneval/all_prompt_logits_replace_results.json", 'r'))
            break
        except Exception:
            continue
    if task_id in all_results:
        continue
    if idx < args.start_index or idx >= args.end_index:
        continue
    final_results[task_id] = {}
    final_results[task_id]['original_pass_rate'] = cal_pass_rate(results[task_id]['oracle_completion'])
    final_results[task_id]['pass_rate'] = {}
    final_results[task_id]['probility'] = {}
    final_results[task_id]['oracle_probility'] = _calculate_probability(results[task_id]['original_prompt'], results[task_id]['oracle_completion_text'])
    # final_results[task_id]['oracle_probility'] = [0] * len(results[task_id]['oracle_completion_text'])
    for completion_id in tqdm(completion_ids):
        if len(results[task_id]['completion'][completion_id]) == 0:
            continue
        final_results[task_id]['pass_rate'][completion_id] = cal_pass_rate(results[task_id]['completion'][completion_id])
        final_results[task_id]['probility'][completion_id] = _calculate_probability(results[task_id]['prompt'][completion_id], results[task_id]['completion_text'][completion_id])
        # final_results[task_id]['probility'][completion_id] = [0] * len(results[task_id]['completion_text'][completion_id])
    while True:
        try:
            with open ("/home/weimin/CodeT5/CodeT5+/humaneval/all_prompt_logits_replace_results.json", 'r') as fp:
                fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
                all_results = json.load(open("/home/weimin/CodeT5/CodeT5+/humaneval/all_prompt_logits_replace_results.json", 'r'))
                all_results[task_id] = final_results[task_id]
                json.dump(all_results, open("/home/weimin/CodeT5/CodeT5+/humaneval/all_prompt_logits_replace_results.json", 'w'))
                break
        except Exception:
            continue
        
# json.dump(final_results, open("/home/weimin/CodeT5/CodeT5+/humaneval/all_prompt_logits_results_without_logits.json", 'w'))