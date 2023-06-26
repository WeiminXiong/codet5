from human_eval.data import read_problems, write_jsonl, stream_jsonl
import glob 
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()

# Inputs
parser.add_argument(
    '--path',
    type=str,
    help="",
    default="/home/weimin/CodeT5/CodeT5+/humaneval/preds/codet5p-770m-py_T0.2_N20_allprompt")
parser.add_argument(
    '--out_path',
    type=str,
    help="",
    default="/home/weimin/CodeT5/CodeT5+/humaneval/preds/codet5p-770m-py_T0.2_N20_allprompt.jsonl")
parser.add_argument(
    '--add_prompt',
    action='store_true',
    help='',
    )

args = parser.parse_args()

output = []

problems = sorted(glob.glob(args.path+'/*'))
for problem in problems:
    # files = sorted(glob.glob(problem+'/*.jsonl'))
    # files = problem
    # for thought_id, file in enumerate(files):
    #     codes = [c for c in stream_jsonl(file)]
    #     for code in codes:
    #         item = dict()
    #         item['task_id'] = code['task_id']
    #         item['all_code'] = code['program']
    #         item['thought_id'] = thought_id
    #         output.append(item)
    codes = stream_jsonl(problem)
    for code in codes:
        item = dict()
        item['task_id'] = code['task_id']
        item['all_code'] = code['all_code']
        item['completion'] = code['completion']
        item['prompt_id'] = code['prompt_id']
        output.append(item)
    
print("save to {}".format(args.out_path))
write_jsonl(args.out_path, output)