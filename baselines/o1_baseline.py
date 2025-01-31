import os
import sys
import json
import traceback
from tqdm import tqdm
import concurrent.futures

sys.path.append('./')
from table_utils import format_table
from utils import fact_to_qa
from azure_openai_api import get_openai_llm_response, get_o1_llm_response
from direct_prompt import DIRECT_PROMPT_TEMPLATE
from evaluate.evaluator import eval_fact




# dataset = 'wikitq'
dataset = 'tabfact'

data_path = f'data/{dataset}/test.jsonl'



model = 'o1'
# model = 'o1m'


MAX_WORKERS = 3

output_path = f'outputs/baselines/{dataset}/{model}/'


if not os.path.exists(output_path):
    os.makedirs(output_path)



with open(data_path) as f:
    data = [json.loads(line) for line in f.readlines()]



def worker(i, D):
    if os.path.exists(f'{output_path}/{i}.json'):
        return

    table = D['table']
    question = D['question']
    gt_answer = D['answer']

    if dataset == 'tabfact':
        question = fact_to_qa(question, model='gpt-4o')
    
    while True:
        try:
            table_md = format_table(table, with_address=False)
            prompt = DIRECT_PROMPT_TEMPLATE.format(table=table_md, question=question)
            if model == 'o1':
                final_answer = get_o1_llm_response(prompt, model='o1-preview')
            elif model == 'o1m':
                final_answer = get_o1_llm_response(prompt, model='o1-mini')
            break

        except:
            traceback.print_exc()
            return

    output_data = {
        **D,
        'predicted_answer': final_answer,
    }
    with open(f'{output_path}/{i}.json', 'w') as f:
        f.write(json.dumps(output_data, indent=4, ensure_ascii=False))



with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(worker, i, D) for i, D in enumerate(data)]
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        future.result()
        