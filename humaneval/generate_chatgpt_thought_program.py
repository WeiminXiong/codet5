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
        'temperature':2,
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

test = "def fizz_buzz(n: int):\n    \"\"\"Return the number of times the digit 7 appears in integers less than n which are divisible by 11 or 13.\n    >>> fizz_buzz(50)\n    0\n    >>> fizz_buzz(78)\n    2\n    >>> fizz_buzz(79)\n    3\n    \"\"\""

# thought_prompt = """For the above question, could you briefly teach me how to solve it step by step in natural language? \nDon’ t write the code in this step.\n"""

thought_ = """Sure! Here are the steps to solve this problem:\n\n1. Define a function named `fizz_buzz` that takes an integer argument `n`.\n\n2. Initialize a variable `count` to 0. This variable will be used to keep track of the number of times the digit 7 appears in integers less than `n` which are divisible by 11 or 13.\n\n3. Loop through all integers less than `n`. For each integer, check if it is divisible by either 11 or 13.\n\n4. If the integer is divisible by either 11 or 13, convert it to a string and check if the digit 7 appears in it.\n\n5. If the digit 7 appears in the string, increment the `count` variable.\n\n6. After looping through all integers less than `n`, return the `count` variable as the result.\n\n7. Write test cases to make sure the function works correctly."""

program_prompt = """Based on the above idea, help me complete the function. \nBe attention, you should only output the codes without any explanation and natural language. Wrap your code with \"'''\""""

thought_path = "/home/weimin/CodeT5/CodeT5+/humaneval/preds/gpt3.5-turbo_T0.8_N20_thought"

# response = send_request()

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='Salesforce/instructcodet5p-16b', help="")
    parser.add_argument('--output_path', type=str, default="test",help="")
    parser.add_argument('--start_index', type=int, default=0, help="")
    parser.add_argument('--end_index', type=int, default=164, help="")
    parser.add_argument('--temperature', type=float, default=2, help="")
    parser.add_argument('--N', type=int, default=200, help="")
    parser.add_argument('--max_len', type=int, default=600, help="")
    parser.add_argument('--decoding_style', type=str, default='sampling', help="")
    parser.add_argument('--num_seqs_per_iter', type=int, default=50, help='')
    parser.add_argument('--overwrite', action='store_true', help='')

    args = parser.parse_args()
    response = send_request(prompt=f"{test}\n", args=args)
    print(response.json()['choices'][0]['message']['content'])
    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    # STOP_SEQS = ['\nclass', '\ndef', '\nif', '\nprint']
    # problems = read_problems()
    prompts = []
    thoughts = []
    task_ids = []
    ids = list(range(args.start_index, args.end_index))
    for id in ids:
        if id>=164:
            break
        thought_file_path = thought_path + '/{}.jsonl'.format(id)
        thought_content = open(thought_file_path, 'r').readlines()
        cur_thoughts = []
        for i, line in enumerate(thought_content):
            data = json.loads(line)
            if i==0:
                prompts.append(data['problem'].strip())
                task_ids.append(data['task_id'])
            cur_thoughts.append(data['thought'].strip())
        thoughts.append(cur_thoughts[:10])
    
    num_samples = len(prompts)
    print("Number of samples: {}".format(num_samples))

    print(f"Loaded {args.model}.")
    for i in tqdm(range(num_samples), ncols=0, total=num_samples):
        prompt = prompts[i]
        task_id = task_ids[i]
        cur_thoughts = thoughts[i]
        os.makedirs(args.output_path + f'/{args.start_index + i}', exist_ok=True)
        for j in tqdm(range(len(thoughts[i])), ncols=0, total=len(thoughts[i])):
            output_file = args.output_path + f'/{args.start_index + i}/{j}.jsonl'
            

            if os.path.exists(output_file) and not args.overwrite:
                print(f'Skip {output_file} as it already exists')
                continue

            thought = cur_thoughts[j]
            

            # ids_batch = [task_ids[i]]
            
            completion_seqs = []

            for _ in tqdm(range(args.N), total=args.N, leave=True, ncols=0):
                description = f"\'''{prompt}\'''\n"+thought+'\n'+program_prompt
                response = send_request(prompt=description, args=args)
                while True:
                    if response.status_code==200:
                        result = json.loads(response.text)
                        if 'choices' not in result.keys():
                            response = send_request(prompt=description, args=args)
                            continue
                        else:
                            predict = result['choices'][0]['message']['content']
                            # if predict.count("\'''") != 2:
                            #     response = send_request(prompt=description, args=args)
                            #     continue
                            break
                    else:
                        response = send_request(prompt=description, args=args)
                
                if predict.count("\'''") == 2:
                    program = predict
                    s_index = program.index("\'''")
                    e_index = program.rindex("\'''")

                    completion_seqs.append(
                        {
                            'task_id': task_id,
                            'problem': prompt,
                            'output': predict,
                            'program': program[s_index+3:e_index],
                            'thought': thought
                        }
                    )
                else:
                    program = predict
                    completion_seqs.append(
                        {
                            'task_id': task_id,
                            'problem': prompt,
                            'output': predict,
                            'program': program,
                            'thought': thought
                        }
                    )
                
                
            print("Saving results to {}".format(output_file))
            write_jsonl(output_file, completion_seqs)


if __name__ == '__main__':
    main()
