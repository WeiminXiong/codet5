from textattack.transformations import WordSwapEmbedding
from textattack.shared.utils import words_from_text
from textattack.shared.attacked_text import AttackedText
import json
from tqdm import tqdm
from copy import deepcopy
import textattack
from constraints import PosConstraint
from textattack.augmentation import Augmenter
from constraints import PosConstraint
from preconstraints import ProgramConstraint, PosPreConstraint
from textattack.constraints.semantics import WordEmbeddingDistance
from transformation import WordTransformation, WordTransformationWithLogits
from textattack.constraints.pre_transformation import (RepeatModification, StopwordModification)
import argparse
from myAugmenter import myAugmenter


parser = argparse.ArgumentParser()
parser.add_argument("--start_index", type=int, default=0)
parser.add_argument("--end_index", type=int, default=0)
parser.add_argument("--thread_id", type=int, default=0)
args = parser.parse_args()

def get_boundary_index(text):
    start_word = ['\"\"\"', '\'\'\'']
    stop_word = ["\n    Example", '\n    For example', '\n    for example', '\n    Examples', '\n    Note','\n    >>>',  "\n    example"] 
    strip_text = text.strip()
    function_name = strip_text[4: strip_text.find('(')]
    start_index = -1
    end_index = -1
    for s_word in start_word:
        if s_word in prompt:
            start_index = prompt.find(s_word)+3
            break
    for e_word in stop_word:
        if e_word in prompt:
            end_index = prompt.find(e_word)
            break
    if end_index==-1:
        end_index = prompt.find(function_name, 15)
    return (start_index, end_index)


DEFAULT_CONSTRAINTS = [RepeatModification(), StopwordModification(), ProgramConstraint(), PosConstraint(), PosPreConstraint()]

embedding = WordSwapEmbedding()

datas = open("/home/weimin/CodeT5/CodeT5+/paraphrase/human-eval-v2-20210705.jsonl", 'r').readlines()
data = [json.loads(item) for item in datas]

map_tokens_path = "/home/weimin/CodeT5/CodeT5+/paraphrase/tokens_map.json"
map_tokens = json.load(open(map_tokens_path, 'r'))

augmented_list = []

constraints = DEFAULT_CONSTRAINTS + [WordEmbeddingDistance(min_cos_sim=0.8), PosConstraint()]

transformation = WordTransformationWithLogits(map_path=map_tokens_path)
for item in tqdm(data):
    prompt = item['prompt']
    start_index, end_index = get_boundary_index(prompt)
    cur_dict = {}
    cur_dict['task_id'] = item['task_id']
    cur_dict['original_prompt'] = prompt
    if prompt.count("def")>=2:
        cur_dict['augmented_prompt'] = [prompt]
        cur_dict['logits'] = [0]
    else:
        content = prompt[start_index:end_index]
        augmenter = myAugmenter(transformation, constraints=constraints, pct_words_to_swap=0, transformations_per_example=100)
        augmented_texts = augmenter.augment(content)
        augmented_prompts = []
        for augmented_text in augmented_texts:
            augmented_prompt = prompt[:start_index]+augmented_text+prompt[end_index:]
            augmented_prompts.append(augmented_prompt)
        if len(augmented_prompts)==0:
            cur_dict['augmented_prompt'] = [prompt]
            cur_dict['logits'] = [0]
        else:
            cur_dict['augmented_prompt'] = augmented_prompts
            cur_dict['logits'] = transformation._calculate_probability(augmented_texts).tolist()
    augmented_list.append(cur_dict)
        

# json.dump(valid_transform, open(f"/home/weimin/CodeT5/CodeT5+/paraphrase/augmented_prompt/valid_transform_{args.thread_id}.json", 'w'))
# json.dump(final_list, open(f"/home/weimin/CodeT5/CodeT5+/paraphrase/augmented_prompt/final_list_{args.thread_id}.json", 'w'))
json.dump(augmented_list, open(f"/home/weimin/CodeT5/CodeT5+/paraphrase/augmented_prompt_logits.json", 'w'))