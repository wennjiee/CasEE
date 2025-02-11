import json

path = './datasets/FewFC/cascading_sampled/ty_args.json'

with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
records = []
for line in lines:
    record = json.loads(line)
    for key in record.keys():
        for item in record[key]:
            if item not in records:
                records.append(item)
print(records)