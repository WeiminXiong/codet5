import json

original  = open("/home/weimin/CodeT5/CodeT5+/humaneval/preds/gpt3.5-turbo_T0.8_N20.jsonl", 'r').readlines()
test_results = open("/home/weimin/CodeT5/CodeT5+/humaneval/preds/gpt3.5-turbo_T0.8_N20.jsonl_results.jsonl", 'r').readlines()
output_file = open("output.jsonl", "w")


for a, b in zip(original, test_results):
    a_json = json.loads(a)
    b_json = json.loads(b)
    result = b_json['result']
    if "outside" in result:
        completion = a_json['completion']
        all_code = a_json['all_code']
        index = all_code.find(completion)
        new_completion = completion.replace("\n", "\n    ")
        a_json['completion'] = new_completion
        a_json['all_code'] = all_code[:index] + new_completion
        output_file.writeline(json.dumps(a_json))
    else:
        output_file.writeline(json.dumps(a_json))