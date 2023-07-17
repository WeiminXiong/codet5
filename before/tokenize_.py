from textattack.transformations import WordSwapEmbedding
from textattack.shared.utils import words_from_text
from textattack.shared.attacked_text import AttackedText
import json
from tqdm import tqdm
from copy import deepcopy
from nltk import WordNetLemmatizer
import nltk
lemmatizer =WordNetLemmatizer()
from nltk import word_tokenize
from nltk import pos_tag
# embedding = WordSwapEmbedding()

datas = open("/home/weimin/CodeT5/CodeT5+/human-eval-v2-20210705.jsonl", 'r').readlines()
# map_tokens = json.load(open("/home/weimin/CodeT5/CodeT5+/new_map.json", 'r'))


start_word = ['\"\"\"', '\'\'\'']
stop_word = ["\n    Example", '\n    For example', '\n    for example', '\n    Examples', '\n    Note','\n    >>>',  "\n    example"]
# result_tokens = set()

# replaced_prompt = []
tokens = dict()

data = [json.loads(item) for item in datas]
for item in tqdm(data):
    cur_dict = {}
    prompt = item['prompt']
    if prompt.count("def")>=2:
        continue
    strip_prompt = prompt.strip()
    function_name = strip_prompt[4: strip_prompt.find('(')]
    start_index = None
    for s_word in start_word:
        if s_word in prompt:
            start_index = prompt.find(s_word)+3
            break
    end_index = None
    for s_word in stop_word:
        if s_word in prompt:
            end_index = prompt.find(s_word)
            break
    if end_index==None:
        end_index = prompt.find(function_name)
        if end_index<start_index:
            end_index = None
    content = prompt[start_index: end_index]
    # nltk.word_tokenize
    for word, tag in pos_tag(word_tokenize(content)):
        if not word.isalpha():
            continue
        if tag.startswith('NN'):
            lem = lemmatizer.lemmatize(word, pos='n')
        elif tag.startswith('VB'):
            lem = lemmatizer.lemmatize(word, pos='v')
        elif tag.startswith('JJ'):
            lem = lemmatizer.lemmatize(word, pos='a')
        else:
            lem = lemmatizer.lemmatize(word, pos='n')
        if word not in tokens:
            tokens[word] = [1, lem]
        else:
            tokens[word][0] += 1
        
tokens = sorted(tokens.items(), key = lambda kv:(kv[1][0], kv[0]), reverse=True)
json.dump(dict(tokens), open("/home/weimin/CodeT5/CodeT5+/tokens.json", 'w'))