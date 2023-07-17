from textattack.transformations import WordSwapEmbedding
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.shared.utils import words_from_text
from textattack.shared.attacked_text import AttackedText
import json
from tqdm import tqdm
from copy import deepcopy
from preconstraints import ProgramConstraint, PosPreConstraint
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from lemminflect import getInflection
from constraints import PosConstraint

noun = ["NN", "NNS", "NNP", "NNPS"]
adjective = ["JJ", "JJR", "JJS"]
adverb = ["RB", "RBR", "RBS"]
verb = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]

wnl = WordNetLemmatizer()

def transform_tag(tag):
    if tag in noun:
        return 'n'
    if tag in adjective:
        return 'a'
    if tag in adverb:
        return 'r'
    if tag in verb:
        return 'v'

embedding = WordSwapEmbedding(max_candidates=50)
embeding_distance = WordEmbeddingDistance(min_cos_sim=0.8)

datas = open("/home/weimin/CodeT5/CodeT5+/paraphrase/human-eval-v2-20210705.jsonl", 'r').readlines()
# map_tokens = json.load(open("/home/weimin/CodeT5/CodeT5+/new_list_map.json", 'r'))

programconstraint = ProgramConstraint()
pospreconstraint = PosPreConstraint()
posconstraint = PosConstraint()

final_list = []

start_word = ['\"\"\"', '\'\'\'']
stop_word = ["\n    Example", '\n    For example', '\n    for example', '\n    Examples', '\n    Note','\n    >>>',  "\n    example"]
# result_tokens = set()

replaced_prompt = []

cnt = 0
data = [json.loads(item) for item in datas]
all_tokens = set()

 
for item in tqdm(data):
    cur_dict = {}
    prompt = item['prompt']
    strip_prompt = prompt.strip()
    function_name = strip_prompt[4: strip_prompt.find('(')]
    start_index = -1
    for s_word in start_word:
        if s_word in prompt:
            start_index = prompt.find(s_word)+3
            break
    end_index = -1
    for s_word in stop_word:
        if s_word in prompt:
            end_index = prompt.find(s_word)
            break
    if end_index==-1:
        end_index = prompt.find(function_name, 15)
        if end_index<start_index:
            end_index = -1
    if not start_index<end_index:
        continue
    content = prompt[start_index: end_index]
    # print(content)
    # print("-------------------------------------------------------------------------")
    text = AttackedText(content)
    tokens = text.words
    editable_index_1 = programconstraint._get_modifiable_indices(text)
    editable_index_2 = pospreconstraint._get_modifiable_indices(text)
    editable_index = editable_index_1 & editable_index_2
    if prompt.count("def") ==2:
        continue
    pos_tags = pos_tag(tokens)
    for i, token in enumerate(tokens):
        if i in editable_index:
            if token.isalpha() and len(token)>1 and not token.lower() in programconstraint.program_constraint:
                all_tokens.add((token.lower(), pos_tags[i][1]))
    
json.dump(list(all_tokens), open("/home/weimin/CodeT5/CodeT5+/paraphrase/all_tokens.json", 'w'))

def similar_words(token, pos_tag):
    lem = wnl.lemmatize(token, transform_tag(pos_tag))
    synonyms = []
    for syn in wordnet.synsets(lem):
        for l in syn.lemmas():
            synonyms.append(l.name().lower())
    synonyms = set(synonyms).difference([token])
    target_synonyms = []
    for syn in synonyms:
        try:
            target_synonyms.append(getInflection(syn, tag=pos_tag)[0])
        except Exception:
            continue
    target_synonyms = set(target_synonyms).difference([token])
    return target_synonyms


tokens_map = {}
# for token in all_tokens:
#     referenced_text = AttackedText(token)
#     candidates = embedding._get_replacement_words(token)
#     similar_list = []
#     for candidate in candidates:
#         transformed_text = AttackedText(candidate)
#         transformed_text.attack_attrs["newly_modified_indices"] = [0]
#         if embeding_distance._check_constraint(transformed_text, referenced_text):
#             similar_list.append(candidate)
#     tokens_map[token] = similar_list

for item in all_tokens:
    token, pos = item
    referenced_text = AttackedText(token)
    candidates = similar_words(token, pos)
    similar_list = []
    for candidate in candidates:
        transformed_text = AttackedText(candidate)
        transformed_text.attack_attrs["newly_modified_indices"] = [0]
        if embeding_distance._check_constraint(transformed_text, referenced_text) and posconstraint._check_constraint(transformed_text, referenced_text) \
        and "_" not in candidate and "-" not in candidate and candidate.isalpha() and len(candidate)>1:
            similar_list.append(candidate)
    if len(similar_list)>0:
        tokens_map[f"{token}_{pos}"] = similar_list


json.dump(tokens_map, open("/home/weimin/CodeT5/CodeT5+/paraphrase/tokens_map.json", 'w'))