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

from utils import fact_to_qa
from evaluate.evaluator import eval_fact, eval_qa, eval_free_qa

# from examples import wikitq_examples, fetaqa_examples, tabfact_examples





def worker(i, D):
    if os.path.exists(f'{output_path}/{i}.json'):
        return

    question = D['question']
    final_answer, reasoning_process = tablemaster_table_understanding(D['table'], question, task='qa', question_i=i)

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
    dataset = 'fetaqa'
    # method = 'tablemaster-4m'
    method = 'tablemaster-4o'

    azure_openai_api.GLOBAL_MODEL = 'gpt-4o'

    from tablemaster_fetaqa.tableqa import tablemaster_table_understanding
    
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



