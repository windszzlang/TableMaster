import os
import sys
import json
import glob
import concurrent.futures
from tqdm import tqdm
import traceback
import random
import numpy as np

sys.path.append('.')
from azure_openai_api import get_o1_llm_response
from table_utils import format_table
from evaluate.evaluator import eval_qa




dataset = 'wikitq'

output_path = f'outputs/analysis/norm/new_data/{dataset}'

if not os.path.exists(output_path):
    os.makedirs(output_path)



with open(f'data/{dataset}/test.jsonl', 'r') as f:
# with open(f'data/{dataset}/subtest.jsonl', 'r') as f:
    test_data = [json.loads(line) for line in f]





PROMPT = """
Given a table, you need to generate a new table by disrupting the content in the table.

Table:
{table}

Rules:
- Your goal is to make the content in each row within the same column follows a different format to increase diversity as much as possible.
- You cannot change the structure of the table.
- You cannot add or remove any rows or columns.
- You cannot modify the column names in the first row.
- You can only alter the format of the content in each cell, not the actual values.
- You should not make the content in each row within the same column in the same format as much as possible.

Format Change Examples:
- Change a number format from 123456 to 123,456.
- Change a date format from 2024-01-01 to 2024/01/01.
- Simplify or abbreviate text content.


Provide your new table in the following JSON format:
```json
{{
    "table": [[...], [...], [...]],
}}
```
"""

    # "description": "what you have done to the table"


def llm_disrupt_table(table):
    prompt = PROMPT.format(table=table)
    response = get_o1_llm_response(prompt, model='o1-preview')
    response = response.replace('```json', '').replace('```', '').strip()
    json_response = json.loads(response)
    # print(prompt)
    # print(json_response)

    # print(json_response['description'])
    table = json_response['table']
    # change = json_response['description']
    change = '...'
    return table, change


DIRECT_PROMPT_TEMPLATE = """## Objective
You are provided with a table and a question related to the table.
Your task is to answer the question directly based on the table.

## Table
{table}

## Question
{question}

The answer should be short and simple. It can be a number, a word, or a phrase in the table, but not a full sentence.
Now, answer the question directly:
Answer: """



good_quality_count = 0

def worker(i, D):
    if os.path.exists(f'{output_path}/{i}.json'):
        return
    
    original_table = D['table']
    question = D['question']
    answer = D['answer']

    try:
        noised_table, change = llm_disrupt_table(original_table)
    except Exception as e:
        noised_table, change = [['Error Table']], 'error table'

    if random.random() < 0.5:
        transpose = True
        noised_table = np.transpose(noised_table).tolist()
    else:
        transpose = False


    noised_table_row = len(noised_table)
    noised_table_col = len(noised_table[0])
    original_table_row = len(original_table)
    original_table_col = len(original_table[0])
    # if not (noised_table_row == original_table_row and noised_table_col == original_table_col) and not (noised_table_col == original_table_row and noised_table_row == original_table_col):
    #     quality = 'size_mismatch'
    # else:
    #     quality = 'good'

    noised_table_md = format_table(noised_table, with_address=False)
    prompt = DIRECT_PROMPT_TEMPLATE.format(table=noised_table_md, question=question)
    o1_predicted_answer = get_o1_llm_response(prompt, model='o1-preview')
    if eval_qa(o1_predicted_answer, answer):
        quality = 'good'
        global good_quality_count
        good_quality_count += 1
        print(f'Total good quality count: {good_quality_count}')
    else:
        quality = 'cannot answer'

    saved_D = {}
    saved_D['id'] = D['id']
    saved_D['table'] = noised_table
    saved_D['original_table'] = original_table
    saved_D['transpose'] = transpose
    saved_D['change'] = change
    saved_D['quality'] = quality

    with open(f'{output_path}/{i}.json', 'w') as f:
        json.dump(saved_D, f)


MAX_WORKERS = 5
with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(worker, i, D) for i, D in enumerate(test_data)]
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        try:
            future.result()
        except Exception as e:
            traceback.print_exc()



# merge and save the noised subtest data
noised_subtest_data = []

files = glob.glob(f'{output_path}/*.json')
files = sorted(files, key=lambda x: int(x.split('/')[-1].split('.')[0]))


for i, file in enumerate(files):
    with open(file, 'r') as f:
        D = json.load(f)
    
    assert i == int(file.split('/')[-1].split('.')[0]), 'The index of the file is not correct'
    
    if D['quality'] == 'good':
        new_D = test_data[i]
        new_D['original_table'] = new_D['table']
        new_D['table'] = D['table']
        new_D['transpose'] = D['transpose']
        noised_subtest_data.append(new_D)

print(f'The number of good quality data is {len(noised_subtest_data)}')


with open(f'data/{dataset}/noised_test.jsonl', 'w') as f:
    for D in noised_subtest_data:
        f.write(json.dumps(D) + '\n')
