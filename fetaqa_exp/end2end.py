import os
import sys
import json
import traceback
from tqdm import tqdm
import concurrent.futures
import numpy as np
import re

sys.path.append('./fetaqa_exp')
sys.path.append('./')

from azure_openai_api import get_openai_llm_response
import azure_openai_api




END2END_PROMPT_TEMPLATE = """
## Objective
You are given a table, a question.
Your task is to answer the question based on the information provided in the table.

## Table
{table}

Now, using the given table, answer the following question: 
Question: {question}
Answer: """



def worker(i, D):
    if os.path.exists(f'{output_path}/{i}.json'):
        return
    question = D['question']
    prompt = END2END_PROMPT_TEMPLATE.format(table=D['table'], question=question)
    final_answer = get_openai_llm_response(prompt, json_output=False)

    output_data = {
        **D,
        'question': question,
        'predicted_answer': final_answer,
        'reasoning_process': {'prompt': prompt}
    }
    # print(output_data)
    with open(f'{output_path}/{i}.json', 'w') as f:
        json.dump(output_data, f)




if __name__ == "__main__":


    MAX_WORKERS = 10
    dataset = 'fetaqa'


    azure_openai_api.GLOBAL_MODEL = 'gpt-4o'

    method = 'end2end'
    

    output_path = f'outputs/main/{dataset}/{method}'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if 'train' in method:
        with open(f'data/{dataset}/train.jsonl', 'r') as f:
            test_data = [json.loads(line) for line in f]
    else:
        with open(f'data/{dataset}/test.jsonl', 'r') as f:
            test_data = [json.loads(line) for line in f]

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(worker, i, D) for i, D in enumerate(test_data)]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            try:
                future.result()
            except Exception as e:
                print()
                traceback.print_exc()





