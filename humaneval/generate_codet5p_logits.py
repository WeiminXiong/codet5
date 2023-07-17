import argparse
import pprint
import os
import re
from tqdm import tqdm
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from human_eval.data import write_jsonl, read_problems, stream_jsonl
import json
from copy import deepcopy

STOP_SEQS = ['\nclass', '\ndef', '\n#', '\nif', '\nprint']

def read_datas():
    datas = open("/home/weimin/CodeT5/CodeT5+/paraphrase/augmented_prompt_logits_old.json", 'r')
    # datas = [json.loads(line) for line in datas]
    datas = json.load(datas)
    # return_dict = {}
    # for item in datas:
    #     return_dict[item['task_id']] = item
    # return return_dict
    return datas


def write_code(augmented_prompt, task_ids, model, tokenizer, device, args):
    augmented_prompt = augmented_prompt.replace('    ', '\t')
    prompt_batch = [augmented_prompt]
    prompt_batch_decoder = [augmented_prompt]

    ids_batch = [task_ids]

    encoding = tokenizer(prompt_batch, return_tensors="pt", truncation=True, max_length=args.max_len).to(device)
    encoding_decoder = tokenizer(prompt_batch_decoder, return_tensors="pt", truncation=True,
                                max_length=args.max_len).to(device)
    
    if args.decoding_style == 'sampling':
        loops = int(args.N / args.num_seqs_per_iter)
    else:
        loops = 1
        
    completion_seqs = []

    for _ in tqdm(range(loops), total=loops, ncols=0):

        with torch.no_grad():
            if args.decoding_style == 'sampling':
                gen_tokens = model.generate(**encoding,
                                            do_sample=True,
                                            temperature=args.temperature,
                                            max_length=args.max_len,
                                            num_return_sequences=args.num_seqs_per_iter,
                                            eos_token_id=tokenizer.eos_token_id,
                                            top_p=0.95)

        if gen_tokens is not None:
            gen_seqs = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        else:
            gen_seqs = None

        if gen_seqs is not None:
            assert len(ids_batch) == 1
            task_id = ids_batch[0]

            for seq_idx, gen_seq in enumerate(gen_seqs):
                completion_seq = gen_seq
                for stop_seq in STOP_SEQS:
                    index = completion_seq.find(stop_seq)
                    if index != -1:
                        completion_seq = completion_seq[:index]
                completion_seq = completion_seq.replace('\t', '    ')
                all_code = augmented_prompt.replace('\t', '    ') + completion_seq

                completion_seqs.append(
                    {'task_id': task_id,
                    'completion': completion_seq,
                    'all_code': all_code,  # final code for evaluation with unit tests,
                    'prompt_id': 1
                    }
                )
    return completion_seqs

to_cal_task_ids = ['HumanEval/61',
 'HumanEval/40',
 'HumanEval/18',
 'HumanEval/152',
 'HumanEval/14',
 'HumanEval/55',
 'HumanEval/3',
 'HumanEval/58',
 'HumanEval/0',
 'HumanEval/54',
 'HumanEval/16',
 'HumanEval/60',
 'HumanEval/66',
 'HumanEval/8'] + \
['HumanEval/13',
 'HumanEval/45',
 'HumanEval/76',
 'HumanEval/162',
 'HumanEval/11',
 'HumanEval/52',
 'HumanEval/5']

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

    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    problems = read_datas()
    
    args.end_index = min(args.end_index, len(problems))
    cur_problems = problems[args.start_index: args.end_index]

    num_samples = len(cur_problems)
    print("Number of samples: {}".format(num_samples))

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model,
                                                  trust_remote_code=True,  # False for 220m and 770m models
                                                  torch_dtype=torch.float16,
                                                  low_cpu_mem_usage=True)
    model.eval()
    model.to(device)

    # for larger LLMs such as 2B, 6B, and 16B, we need to pass the text prompt to the decoder
    print(f"Loaded {args.model}.")
    for i in tqdm(range(num_samples), ncols=0, total=num_samples):
        
        # cur_item = {}
        output_file = args.output_path + '/{}.json'.format(args.start_index + i)

        if os.path.exists(output_file) and not args.overwrite:
            print(f'Skip {output_file} as it already exists')
            continue
        
        cur_problem = cur_problems[i]
        if not cur_problem['task_id'] in to_cal_task_ids:
            continue
        results = {}
        results = deepcopy(cur_problem)
        results['completion'] = []
        for prompt in cur_problem['augmented_prompt']:
            completion = write_code(prompt, cur_problem['task_id'], model, tokenizer, device, args)
            results['completion'].append(completion)
        
        results['oracle_completion'] = write_code(cur_problem['original_prompt'], cur_problem['task_id'], model, tokenizer, device, args)
        
        print("Saving results to {}".format(output_file))
        # write_jsonl(output_file, completion_seqs)
        json.dump(results, open(output_file, 'w'))


if __name__ == '__main__':
    main()
