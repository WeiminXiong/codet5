import argparse
import pprint
import os
import re
from tqdm import tqdm
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from human_eval.data import write_jsonl, read_problems, stream_jsonl
import requests
import json


def send_request(prompt, args):
    dict = {'model':"gpt-3.5-turbo",
        'messages' : [{'role': 'user', 'content':prompt}],
        'temperature':args.temperature,
        'top_p' : 0.9,
        'max_tokens': args.max_len,
        'frequency_penalty':0,
        'presence_penalty':0,}

    headers = {
        'Authorization': 'Bearer sb-ed07016f987c6bb701b74fcf399c56d067b5a5c8bc3ad177',
        'Content-Type': 'application/json',
    }

    data = json.dumps(dict)
    while True:
        try:
            response = requests.post('https://api.openai-sb.com/v1/chat/completions', headers=headers, data=data)
        except Exception:
            continue
        else:
            break
    return response

test = """
    def has_close_elements(numbers: List[float], threshold: float) -> bool:
    \""" Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    \"""
"""

thought_prompt = """For the above question, could you briefly teach me how to solve it step by step in natural language? Don’ t write the code in this step."""

# response = send_request()

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='Salesforce/instructcodet5p-16b', help="")
    parser.add_argument('--output_path', type=str, help="")
    parser.add_argument('--start_index', type=int, default=0, help="")
    parser.add_argument('--end_index', type=int, default=164, help="")
    parser.add_argument('--temperature', type=float, default=0.8, help="")
    parser.add_argument('--N', type=int, default=200, help="")
    parser.add_argument('--max_len', type=int, default=600, help="")
    parser.add_argument('--decoding_style', type=str, default='sampling', help="")
    parser.add_argument('--num_seqs_per_iter', type=int, default=50, help='')
    parser.add_argument('--overwrite', action='store_true', help='')

    args = parser.parse_args()
    response = send_request(prompt=f"\'''{test}\'''\n"+thought_prompt, args=args)
    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    STOP_SEQS = ['\nclass', '\ndef', '\nif', '\nprint']
    problems = read_problems()

    task_ids = sorted(problems.keys())[args.start_index: args.end_index]
    prompts = [problems[task_id]['prompt'] for task_id in task_ids]
    num_samples = len(prompts)
    print("Number of samples: {}".format(num_samples))

    print(f"Loaded {args.model}.")
    for i in tqdm(range(num_samples), ncols=0, total=num_samples):
        output_file = args.output_path + '/{}.jsonl'.format(args.start_index + i)

        if os.path.exists(output_file) and not args.overwrite:
            print(f'Skip {output_file} as it already exists')
            continue

        prompt = prompts[i].replace('    ', '\t')

        ids_batch = [task_ids[i]]
        
        completion_seqs = []

        for _ in tqdm(range(args.N), total=args.N, leave=True, ncols=0):

            response = send_request(prompt=prompt, args=args)
            while True:
                if response.status_code==200:
                    result = json.loads(response.text)
                    if 'choices' not in result.keys():
                        response = send_request(prompt=prompt, args=args)
                        continue
                    else:
                        predict = result['choices'][0]['message']['content']
                        break
                else:
                    response = send_request(prompt=prompt, args=args)
            
            assert len(ids_batch) == 1
            task_id = ids_batch[0]

            completion_seq = predict
            for stop_seq in STOP_SEQS:
                index = completion_seq.find(stop_seq)
                if index != -1:
                    completion_seq = completion_seq[:index]
            completion_seq = completion_seq.replace('\t', '    ')
            completion_seq = '    '+completion_seq
            all_code = prompt.replace('\t', '    ') + completion_seq

            completion_seqs.append(
                {
                    'task_id': task_id,
                    'completion': completion_seq,
                    'all_code': all_code  # final code for evaluation with unit tests
                }
            )

        print("Saving results to {}".format(output_file))
        write_jsonl(output_file, completion_seqs)


if __name__ == '__main__':
    main()
