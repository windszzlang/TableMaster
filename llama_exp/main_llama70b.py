# meta-llama/Meta-Llama-3.1-70B-Instruct
import os
import sys
import json
import traceback
from tqdm import tqdm
import concurrent.futures
import numpy as np
import re

sys.path.append('./')

import azure_openai_api

from tablemaster_llama70b.tableqa import tablemaster_table_understanding
from utils import fact_to_qa




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

    dataset = 'wikitq'
    # dataset = 'tabfact'



    output_path = f'outputs/main/{dataset}/tablemaster-llama70b'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(f'data/{dataset}/test.jsonl', 'r') as f:
        test_data = [json.loads(line) for line in f]


    MAX_WORKERS = 10
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(worker, i, D) for i, D in enumerate(test_data)]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            try:
                future.result()
            except Exception as e:
                traceback.print_exc()

