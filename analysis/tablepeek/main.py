import os
import sys
import json
import traceback
from tqdm import tqdm
import concurrent.futures
import numpy as np
import re

sys.path.append('./')
sys.path.append('./analysis/tablepeek')

from azure_openai_api import get_openai_llm_response
import azure_openai_api

from tablemaster_peek.tableqa import tablemaster_table_understanding
from utils import fact_to_qa
from evaluate.evaluator import eval_fact, eval_qa



import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('--peek_size', type=int, default=1)
# args = parser.parse_args()

# PEEK_SIZE = args.peek_size
PEEK_SIZE = 50



def worker(i, D):
    # if os.path.exists(f'{output_path}/{i}.json'):
        # return

    if os.path.exists(f'{output_path}/{i}.json'):
        with open(f'{output_path}/{i}.json', 'r') as f:
            output_data = json.load(f)
            if eval_qa(output_data['predicted_answer'], output_data['answer']):
                return

    if dataset == 'tabfact':
        question = fact_to_qa(D['question'], model=azure_openai_api.GLOBAL_MODEL)
        final_answer, reasoning_process = tablemaster_table_understanding(D['table'], question, task='fact', peek_size=PEEK_SIZE)
    else:
        question = D['question']
        # D['table'] = D['table'][:50]
        final_answer, reasoning_process = tablemaster_table_understanding(D['table'], question, task='qa', peek_size=PEEK_SIZE)

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


    MAX_WORKERS = 10
    dataset = 'wikitq'
    # dataset = 'tabfact'



    # azure_openai_api.GLOBAL_MODEL = 'gpt-3.5-turbo'
    azure_openai_api.GLOBAL_MODEL = 'gpt-4o-mini'
    # azure_openai_api.GLOBAL_MODEL = 'gpt-4o'

    if azure_openai_api.GLOBAL_MODEL == 'gpt-3.5-turbo':
        method = 'tablemaster-35'
    elif azure_openai_api.GLOBAL_MODEL == 'gpt-4o-mini':
        method = 'tablemaster-4m'
    elif azure_openai_api.GLOBAL_MODEL == 'gpt-4o':
        method = 'tablemaster-4o'

    output_path = f'outputs/analysis/peek/{dataset}/{method}/{PEEK_SIZE}'

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



