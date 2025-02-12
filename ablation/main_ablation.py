import os
import sys
import json
import traceback
from tqdm import tqdm
import concurrent.futures
import numpy as np
import re

sys.path.append('./')
sys.path.append('./ablation/tablemaster_variant')

from azure_openai_api import get_openai_llm_response
import azure_openai_api

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



    MAX_WORKERS = 8


    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ablation_method', type=str, default='wo_structure_extraction')
    parser.add_argument('--dataset', type=str, default='wikitq')
    args = parser.parse_args()

    # ablation_method = args.ablation_method
    # dataset = args.dataset

    dataset = 'wikitq'
    # dataset = 'tabfact'

    # ablation_method = 'wo_structure_extraction'
    ablation_method = 'wo_column_lookup'
    # ablation_method = 'wo_row_lookup'
    # ablation_method = 'wo_table_of_focus'
    # ablation_method = 'wo_reconstruction'
    # ablation_method = 'wo_verbalization'
    # ablation_method = 'wo_textual_reasoning'
    # ablation_method = 'wo_symbolic_reasoning'
    # ablation_method = 'wo_textual_guidance'




    if ablation_method == 'wo_structure_extraction':
        from tablemaster_wo_se.tableqa import tablemaster_table_understanding
    elif ablation_method == 'wo_column_lookup':
        from tablemaster_wo_cl.tableqa import tablemaster_table_understanding
    elif ablation_method == 'wo_row_lookup':
        from tablemaster_wo_rl.tableqa import tablemaster_table_understanding
    elif ablation_method == 'wo_table_of_focus':
        from tablemaster_wo_tof.tableqa import tablemaster_table_understanding
    elif ablation_method == 'wo_reconstruction':
        from tablemaster_wo_re.tableqa import tablemaster_table_understanding
    elif ablation_method == 'wo_verbalization':
        from tablemaster_wo_ver.tableqa import tablemaster_table_understanding
    elif ablation_method == 'wo_textual_reasoning':
        from tablemaster_wo_tr.tableqa import tablemaster_table_understanding
    elif ablation_method == 'wo_symbolic_reasoning':
        from tablemaster_wo_sr.tableqa import tablemaster_table_understanding
    elif ablation_method == 'wo_textual_guidance':
        from tablemaster_wo_tg.tableqa import tablemaster_table_understanding


    # azure_openai_api.GLOBAL_MODEL = 'gpt-3.5-turbo'
    azure_openai_api.GLOBAL_MODEL = 'gpt-4o-mini'
    # azure_openai_api.GLOBAL_MODEL = 'gpt-4o'

    output_path = f'outputs/ablation/{dataset}/{ablation_method}'
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



