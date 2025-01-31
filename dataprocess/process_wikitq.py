import sys
import json
import pandas as pd
import re
import numpy as np
from io import StringIO

sys.path.append('./')
from table_utils import *



split = 'test'
# split = 'train'




if split == 'train':
    test_file_path = 'data/origin/WikiTableQuestions/data/training.tsv'
else:
    test_file_path = 'data/origin/WikiTableQuestions/data/pristine-unseen-tables.tsv'
df = pd.read_csv(test_file_path, delimiter='\t')




def merge_spaces(input_string):
    # Replace multiple consecutive spaces and tabs (but not newlines) with a single space
    output_string = re.sub(r'[ \t]+', ' ', input_string)
    return output_string.strip()  # Optionally strip leading/trailing spaces



def table_to_array_list(table):
    # Split the table into lines
    lines = table.strip().split("\n")
    
    # Extract header and rows
    header = [col.strip() for col in lines[0].split("|")[1:-1]]
    rows = [
        [col.strip() for col in line.split("|")[1:-1]]
        for line in lines[1:]
    ]
    
    # Combine into an array list
    return [header] + rows

data = []

# do not use .table because of ignoring some info

for i, row in df.iterrows():

    table_pd = pd.read_csv(f'data/origin/WikiTableQuestions/{row['context']}', escapechar='\\', quotechar='\"')
    table = [table_pd.columns.tolist()] + table_pd.values.tolist()
    table_array = np.array(table) # verify the table is correct
    D = {
        'id': f'wikitq-{i}',
        'source': {
            'dataset': 'wikitq',
            'id': row['id'],
            'file_path': row['context']
        },
        'table': table,
        'question': row['utterance'],
        # 'answer_type': 'short',
        'answer': row['targetValue']
    }
    data.append(D)
    # print(D['table'])
    # break


with open(f'data/wikitq/{split}.jsonl', 'w') as f:
    for D in data:
        f.write(json.dumps(D) + '\n')