import os
import sys
import json
import traceback
from tqdm import tqdm
import concurrent.futures
import numpy as np
import re

sys.path.append('./')

from azure_openai_api import get_openai_llm_response
import azure_openai_api

from tablemaster.tableqa import tablemaster_table_understanding
from utils import fact_to_qa
from evaluate.evaluator import eval_fact, eval_qa



def test(data):
    table_array = data['table']
    question = data['question']    
    final_answer, reasoning_process = tablemaster_table_understanding(table_array, question)
    print(final_answer)
    print(reasoning_process)



def worker(i, D):
    if os.path.exists(f'{output_path}/{i}.json'):
        return

    if dataset == 'tabfact':
        question = fact_to_qa(D['question'], model=azure_openai_api.GLOBAL_MODEL)
        final_answer, reasoning_process = tablemaster_table_understanding(D['table'], question, task='fact')
    else:
        question = D['question']
        final_answer, reasoning_process = tablemaster_table_understanding(D['table'], question, task='qa')

    output_data = {
        **D,
        'question': question,
        'predicted_answer': final_answer,
        'reasoning_process': reasoning_process
    }
    # print(output_data)
    with open(f'{output_path}/{i}.json', 'w') as f:
        json.dump(output_data, f)





if __name__ == "__main__":
    # azure_openai_api.GLOBAL_MODEL = 'gpt-4o-mini'
    # with open('test_tables/test2.json', 'r') as f:
    # # with open('test3.json', 'r') as f:
    # # with open('test2.json', 'r') as f:
    #     data = json.load(f)
    # test(data)

    MAX_WORKERS = 10
    dataset = 'wikitq'
    # dataset = 'tabfact'



    # azure_openai_api.GLOBAL_MODEL = 'gpt-3.5-turbo'
    azure_openai_api.GLOBAL_MODEL = 'gpt-4o-mini'
    # azure_openai_api.GLOBAL_MODEL = 'gpt-4o'

    if azure_openai_api.GLOBAL_MODEL == 'gpt-3.5-turbo':
        method = 'tablemaster-35'
    elif azure_openai_api.GLOBAL_MODEL == 'gpt-4o-mini':
        # method = 'tablemaster-4m'
        method = 'tablemaster-4m-new'
    elif azure_openai_api.GLOBAL_MODEL == 'gpt-4o':
        method = 'tablemaster-4o'

    output_path = f'outputs/main/{dataset}/{method}'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

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



