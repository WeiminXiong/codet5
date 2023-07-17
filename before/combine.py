import json
import glob

dir_path = "/home/weimin/CodeT5/CodeT5+/humaneval/preds/codet5p-770m-py_T0.2_N20_all"
all_files = glob.glob(dir_path + "/*.json")

task_id_result = json.load(open("/home/weimin/CodeT5/CodeT5+/before/task_id_result.json", 'r'))

new_prompt_result = json.load(open("/home/weimin/CodeT5/CodeT5+/new_dict.json", 'r'))
li = []


for i, file in enumerate(all_files):
    data = json.load(open(file, 'r'))
    count = data['count']
    programs = data['data']
    f1 = {}
    for program in programs:
        task_id = program['task_id']
        completion = program['completion']
        if completion==None:
            f1[task_id] = task_id_result[task_id]
        else:
            f1[task_id] = new_prompt_result[str(count)+"_"+task_id]
            
    li.append(f1)

json.dump(li, open('li.json', 'w'))

