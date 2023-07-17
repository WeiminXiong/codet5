import argparse
import torch
from transformers import AutoModelForSeq2SeqLM
import json

def read_datas():
    

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='Salesforce/codet5p-770m-py')
parser.add_argument('--output_path', type=str, help="")
parser.add_argument('--start_index', type=int, default=0, help="")
parser.add_argument('--end_index', type=int, default=164, help="")
parser.add_argument('--temperature', type=float, default=0.8, help="")
parser.add_argument('--N', type=int, default=200, help="")
parser.add_argument('--max_len', type=int, default=600, help="")
parser.add_argument('--decoding_style', type=str, default='sampling', help="")
parser.add_argument('--num_seqs_per_iter', type=int, default=50, help='')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")