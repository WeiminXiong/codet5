import json
import numpy as np

def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

test_results_path = "/home/weimin/CodeT5/CodeT5+/before/all_prompt.jsonl_results.jsonl"

prompt_id_sets = set()

test_results = open(test_results_path, 'r').readlines()
test_results = [json.loads(line) for line in test_results]
task_ids = list(set([result['task_id'] for result in test_results]))
final_result = {}

for item in test_results:
    prompt_id_sets.add(item['prompt_id'])

for task_id in task_ids:
    for count in prompt_id_sets:
        final_result[str(count)+"_"+task_id] = []
# thought_ids = list(range(2))

for item in test_results:
    count = item['prompt_id']
    task_id = item['task_id']
    final_result[str(count)+"_"+task_id].append(item)
    
new_dict = {}
for task_id in task_ids:
    for count in prompt_id_sets:
        key =str(count)+"_"+task_id
        if len(final_result[key]) >0 :
            li = final_result[key]
            cnt = 0
            for item in li:
                if item['passed']==True:
                    cnt+=1
            
            pass_rate_1 = estimator(20, cnt, 1)
            pass_rate_5 = estimator(20, cnt, 5)
            pass_rate_10 = estimator(20, cnt, 10)
            pass_rate_20 = estimator(20, cnt, 20)
            new_dict[key] = {
                "pass@1": pass_rate_1,
                "pass@5": pass_rate_5,
                "pass@10": pass_rate_10,
                "pass@20": pass_rate_20
            }
            
json.dump(new_dict, open('new_dict.json', 'w'))
            
# results= {}

# for task_id in task_ids:
#     results[task_id] = {}
#     for i in thought_ids:
#         results[task_id][i] = []

# for item in test_results:
#     task_id = item['task_id']
#     thought_id = item['prompt_id']
#     results[task_id][thought_id].append(item)
    
# json.dump(results, open('codet5p-770m-py_T0.2_N20_bestprompt.jsonl_results_result.json', 'w'))

# to_cal_result = {}
# for task_id in task_ids:
#     to_cal_result[task_id] = []
#     for i in thought_ids:
#         cnt=0
#         for item in results[task_id][i]:
#             if item['passed']:
#                 cnt+=1
#         to_cal_result[task_id].append(cnt)
        
    
# ks = [1, 5, 10, 20]

# pass_rate_1 = []
# pass_rate_5 = []
# pass_rate_10 = []
# pass_rate_20 = []

# final_result = {}
# for task_id in task_ids:
#     final_result[task_id] = {}
#     for i in thought_ids:
#         final_result[task_id][i] = {}
#         for k in ks:
#             final_result[task_id][i][f"pass@{k}"] = estimator(20, to_cal_result[task_id][i], k)
#             if i ==0:
#                 if k == 1:
#                     pass_rate_1.append(estimator(20, to_cal_result[task_id][i], k))
#                 elif k == 5:
#                     pass_rate_5.append(estimator(20, to_cal_result[task_id][i], k))
#                 elif k == 10:
#                     pass_rate_10.append(estimator(20, to_cal_result[task_id][i], k))
#                 elif k == 20:
#                     pass_rate_20.append(estimator(20, to_cal_result[task_id][i], k))

# print("pass@1: {}".format(np.mean(pass_rate_1)))
# print("pass@5: {}".format(np.mean(pass_rate_5)))
# print("pass@10: {}".format(np.mean(pass_rate_10)))
# print("pass@20: {}".format(np.mean(pass_rate_20)))      
            
json.dump(final_result, open('codet5p-770m-py_T0.2_N20_bestprompt.jsonl_results_final_result.json', 'w'))