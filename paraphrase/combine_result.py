import json
import glob
import numpy as np

dir_path = "/home/weimin/CodeT5/CodeT5+/humaneval/preds/codet5p-770m-py_T0.2_N20_all/*.json"
all_files = glob.glob(dir_path)

task_id_result_path = "/home/weimin/CodeT5/CodeT5+/paraphrase/task_id_result.json"
task_id_result = json.load(open(task_id_result_path, 'r'))

test_result_path = "/home/weimin/CodeT5/CodeT5+/humaneval/all_prompt.jsonl_results.jsonl"
test_result = []
lines = open(test_result_path, 'r').readlines()
for line in lines:
    test_result.append(json.loads(line))

all_task_ids = []
cur_data = json.load(open(all_files[0], 'r'))
for item in cur_data['data']:
    all_task_ids.append(item['task_id'])


transform_results = {}
final_results = {}
id_to_transform = {}
for path in all_files:
    data = json.load(open(path, 'r'))
    id_to_transform[data['count']] = data['transform']

for id in id_to_transform:
    transform_results[id] = {}
    final_results[id] = {}
    for task_id in all_task_ids:
        transform_results[id][task_id] = []

for item in test_result:
    transform_results[item['prompt_id']][item['task_id']].append(item)

def cal_pass_rate(li, ks = [1, 5, 10, 20]):
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

for id in id_to_transform:
    final_results[id]['replaced_count'] = 0
    final_results[id]['transform'] = id_to_transform[id]
    final_results[id]['pass_rate'] = []
    for task_id in all_task_ids:
        li = transform_results[id][task_id]
        if len(li)>0:
            pass_rate = cal_pass_rate(li)
            final_results[id]['pass_rate'].append({'task_id': task_id, 'pass_rate': pass_rate})
            final_results[id]['replaced_count'] += li[0]['replaced_count']
        else:
            pass_rate = task_id_result[task_id]
            final_results[id]['pass_rate'].append({'task_id': task_id, 'pass_rate': pass_rate})

json.dump(final_results, open('final_results.json', 'w'))

cal_results = {}
for id in id_to_transform:
    cal_results[id] = {}
    cal_results[id]['replaced_count'] = final_results[id]['replaced_count']
    cal_results[id]['transform'] = final_results[id]['transform']
    cal_results[id]['pass_rate'] = {}
    cal_results[id]['pass_rate']['pass@1'] = np.mean([item['pass_rate']['pass@1'] for item in final_results[id]['pass_rate']])
    cal_results[id]['pass_rate']['pass@5'] = np.mean([item['pass_rate']['pass@5'] for item in final_results[id]['pass_rate']])
    cal_results[id]['pass_rate']['pass@10'] = np.mean([item['pass_rate']['pass@10'] for item in final_results[id]['pass_rate']])
    cal_results[id]['pass_rate']['pass@20'] = np.mean([item['pass_rate']['pass@20'] for item in final_results[id]['pass_rate']])
    
cal_results = dict(sorted(cal_results.items(), key=lambda item: item[1]['pass_rate']['pass@1'], reverse=True))

json.dump(cal_results, open('cal_results.json', 'w'))