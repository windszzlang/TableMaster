import os
import sys
import json
import concurrent.futures
from tqdm import tqdm
import traceback

sys.path.append('.')
from azure_openai_api import get_o1_llm_response, get_openai_llm_response




dataset = 'wikitq'
# dataset = 'tabfact'

# output_path = f'outputs/analysis/number/{dataset}'
output_path = f'outputs/analysis/number-4m-self/{dataset}'


if not os.path.exists(output_path):
    os.makedirs(output_path)



with open(f'data/{dataset}/test.jsonl', 'r') as f:
    test_data = [json.loads(line) for line in f]


# o1
# PROMPT = """
# Table:
# {table}

# Question:
# {question}

# Determine whether a calculation is required to answer the question, or if the question can be directly answered using the information in the table.

# Provide your response in the following JSON format:
# {{
#     "need_calculation": true/false
# }}
# """

## Objective
PROMPT = """
You are provided with a table and a question related to the table.
Your task is to assess whether answering this question needs mathematical calculation.

## Table
{table}

## Question
{question}

## Instructions
1. If the question can be easily answered using the information in the table, respond with False.
2. If the question involves comparison, respond with False.
2. When the question involves counting a substantial number (more than 5) of items or rows, respond with True.
3. If the question demands complex calculations or multi-step mathematical operations based on the table's data, the response should be True.
4. For simple arithmetic or small-scale counting that requires minimal computational effort, respond with False.

## Response Format
The response should be in JSON format:
```json
{{
    "need_calculation": true/false
}}
```
"""

def llm_classify_qa(table, question):
    prompt = PROMPT.format(table=table, question=question)
    # response = get_o1_llm_response(prompt, model='o1-mini')
    response = get_openai_llm_response(prompt, model='gpt-4o-mini', json_output=True)
    response = response.replace('```json', '').replace('```', '').strip()
    json_response = json.loads(response)
    # print(prompt)
    # print(json_response)

    return json_response


def worker(i, D):
    if os.path.exists(f'{output_path}/{i}.json'):
        return
    table = D['table']
    question = D['question']
    response = llm_classify_qa(table, question)
    with open(f'{output_path}/{i}.json', 'w') as f:
        json.dump(response, f)


MAX_WORKERS = 5
with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(worker, i, D) for i, D in enumerate(test_data)]
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        try:
            future.result()
        except Exception as e:
            traceback.print_exc()





