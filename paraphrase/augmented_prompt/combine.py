import json

final_list = []
for i in range(10):
    # data = json.load(open(f'final_list_{i}.json', 'r'))
    # for li in data:
    #     di = {}
    #     di['count'] = li[0]['transform_id']
    #     transform = None
    #     for item in li:
    #         if item['transform'] !=None:
    #             transform = item['transform']
    #     di['transform'] = transform
    #     di ['data'] = li
    #     final_list.append(di)
        
    data = json.load(open(f'valid_transform_{i}.json', 'r'))
    final_list.extend(data)
    
json.dump(final_list, open('valid_transform.json', 'w'))

# json.dump(final_list, open('final_list.json', 'w'))