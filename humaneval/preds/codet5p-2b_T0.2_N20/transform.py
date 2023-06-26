import json

for i in range(164):
    fp = open(f'{i}.jsonl', 'r', encoding='utf-8')
    data = fp.readlines()

    data = [json.loads(d) for d in data]

    fw = open(f'{i}.txt', 'w', encoding='utf-8')
    for d in data:
        fw.write(d["all_code"] + '\n')
        
    fw.close()