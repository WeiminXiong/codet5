from textattack.augmentation import EmbeddingAugmenter
import json
from copy import deepcopy
from tqdm import tqdm

datas = open("/home/weimin/CodeT5/CodeT5+/human-eval-v2-20210705.jsonl", 'r').readlines()


start_word = '\"\"\"'
stop_word = ['\n    For example', '\n    for example', '\n    Examples', '\n    Note','\n    >>>']

# test = "\ndef generate_integers(a, b):\n    \"\"\"\n    Given two positive integers a and b, return the even digits between a\n    and b, in ascending order.\n\n    For example:\n    generate_integers(2, 8) => [2, 4, 6, 8]\n    generate_integers(8, 2) => [2, 4, 6, 8]\n    generate_integers(10, 14) => []\n    \"\"\"\n"

# start_index = test.find(start_word)+3
# end_index = None
# for s_word in stop_word:
#     if s_word in test:
#         end_index = test.find(s_word)
#         break
# print(test[:start_index])
# print("------------------------------------------------------------------")
# print(test[start_index:end_index])
# print("------------------------------------------------------------------")
# print(test[end_index:])

data = [json.loads(item) for item in datas]

augmented_data = []

augmenter = EmbeddingAugmenter(transformations_per_example=10, fast_augment=True)

for item in tqdm(data):
    task_id = item['task_id']
    prompt = item['prompt']
    new_item = deepcopy(item)
    new_item['original_prompt'] = prompt
    new_item['augmented_prompt'] = []
    start_index = prompt.find(start_word)+3
    end_index = None
    for s_word in stop_word:
        if s_word in prompt:
            end_index = prompt.find(s_word)
            break
    content = prompt[start_index:end_index]
    augmented_content = augmenter.augment(content)
    for temp in augmented_content:
        new_item['augmented_prompt'].append(prompt[:start_index]+temp+prompt[end_index:])
    augmented_data.append(new_item)

with open('human_eval_augmented.jsonl', 'w') as fw:
    fw.write('\n'.join([json.dumps(item) for item in augmented_data]))