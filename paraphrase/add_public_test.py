from human_eval.data import stream_jsonl, write_jsonl
from copy import deepcopy

problems = stream_jsonl("/home/weimin/CodeT5/CodeT5+/paraphrase/human-eval-v2-20210705.jsonl")
problems = list(problems)

frontier = "\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\n\ndef check(candidate):\n"

def get_test_case(prompt, entry_point):
    index = prompt.find(entry_point)
    index = prompt.find(entry_point, index+1)
    cur_test = frontier
    while index !=-1:
        s_index = prompt.find('\n', index)
        e_index = prompt.find('\n', s_index+1)
        
        if "\u279e" in prompt[index: s_index].strip():
            temp = prompt[index: s_index].strip().replace("\u279e", "==")
            ass = f'    assert {temp}\n'
        elif "==>" in prompt[index: s_index].strip():
            ass = f"    assert {prompt[index: s_index].strip().replace('==>', '==')}\n"
        elif "=>" in prompt[index: s_index].strip():
            ass = f"    assert {prompt[index: s_index].strip().replace('=>', '==')}\n"
        elif "->" in prompt[index: s_index].strip():
            ass = f"    assert {prompt[index: s_index].strip().replace('->', '==')}\n"
        elif "#" in prompt[index: s_index].strip():
            temp = prompt[index: s_index].strip()
            idx = temp.find("#")
            ass = f"    assert {temp[:idx].strip()}\n"
        elif "==" in prompt[index: s_index].strip():
            ass = f"    assert {prompt[index: s_index].strip()}\n"
        elif "=" in prompt[index: s_index].strip():
            ass = f"    assert {prompt[index: s_index].strip().replace('=', '==')}\n"
        else:
            ass = f"    assert {prompt[index: s_index].strip()} == {prompt[s_index:e_index].strip() if len(prompt[s_index:e_index].strip())>0 else None}\n"
        cur_test += ass.replace(entry_point[:-1], "candidate")
        index = prompt.find(entry_point, index+1)
    return cur_test

new_problems=  []
for item in problems:
    new_item = deepcopy(item)
    prompt = item['prompt']
    entry_point = item['entry_point']+'('
    cur_test = get_test_case(prompt, entry_point)
    new_item['public_test'] = cur_test
    new_problems.append(new_item)
    print(cur_test)
        
write_jsonl( "human-eval.jsonl", new_problems)