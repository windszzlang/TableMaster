import sys
import json
import pandas as pd
import re


sys.path.append('./')
from table_utils import *



split = 'train'
# split = 'test'



if split == 'train':
    data_path = 'data/origin/FeTaQA/fetaQA-v1_train.jsonl'
else:
    data_path = 'data/origin/FeTaQA/fetaQA-v1_test.jsonl'

with open(data_path, 'r') as f:
    original_data = [json.loads(line) for line in f]


data = []


for i, D in enumerate(original_data):
    D = {
        'id': f'fetaqa-{i}',
        'source': {
            'dataset': 'fetaqa',
            'id': D['feta_id'],
            'file_path': D['table_source_json']
        },
        'table': D['table_array'],
        'question': D['question'],
        # 'answer_type': 'long',
        'answer': D['answer']
    }
    data.append(D)
    # break

if split == 'train':
    with open(f'data/fetaqa/{split}.jsonl', 'w') as f:
        # for D in data[:30]:
        for D in data:
            f.write(json.dumps(D) + '\n')

elif split == 'test':
    with open(f'data/fetaqa/{split}.jsonl', 'w') as f:
        for D in data:
            f.write(json.dumps(D) + '\n')