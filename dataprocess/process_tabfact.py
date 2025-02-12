import sys
import json
import pandas as pd
import re
import numpy as np


sys.path.append('./')
from table_utils import *



split = 'test'
# split = 'train'




# if split == 'train':
#     file_path = 'data/origin/Table-Fact-Checking/tokenized_data/train_examples.json'
# else:
#     file_path = 'data/origin/Table-Fact-Checking/tokenized_data/test_examples.json'

# with open(file_path, 'r') as f:
#     original_data = json.load(f)


# follow previous works to use small test ids
with open('data/origin/Table-Fact-Checking/data/small_test_id.json', 'r') as f:
    small_test_ids = json.load(f)


with open('data/origin/Table-Fact-Checking/collected_data/r1_training_all.json', 'r') as f:
    original_data_1 = json.load(f)

with open('data/origin/Table-Fact-Checking/collected_data/r2_training_all.json', 'r') as f:
    original_data_2 = json.load(f)

original_data = {**original_data_1, **original_data_2}


def table_to_array_list(table):
    # Split the table into lines
    lines = table.strip().split("\n")
    
    # Extract header and rows
    header = lines[0].split("#")
    rows = [line.split("#") for line in lines[1:]]
    
    # Combine into an array list
    return [header] + rows


data = []
idx = 0


for key, value in original_data.items():
    if key in small_test_ids:
        with open(f'data/origin/Table-Fact-Checking/data/all_csv/{key}', 'r') as f:
            table = f.read()
        # print(table)
        table = table_to_array_list(table)
        # table = np.array(table)
        for statement, label in zip(value[0], value[1]):
            D = {
                'id': f'tabfact-{idx}',
                'source': {
                    'dataset': 'tabfact',
                    'id': key,
                    'file_path': key
                },
                'table': table,
                'question': statement,
                # 'answer_type': 'boolean',
                # 'answer': 'true' if label == 1 else 'false'
                'answer': True if label == 1 else False
            }
            data.append(D)
            idx += 1
        # print(D['table'])
        # break


with open(f'data/tabfact/{split}.jsonl', 'w') as f:
    for D in data:
        f.write(json.dumps(D) + '\n')


print(len(small_test_ids))
print(len(data))