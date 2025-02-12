import os
import re
import sys
import json
import traceback
from tqdm import tqdm
import concurrent.futures

sys.path.append('./')
from table_utils import format_table
from azure_openai_api import get_openai_llm_response, get_o1_llm_response
from utils import fact_to_qa
from evaluate.evaluator import eval_fact

from direct_prompt import DIRECT_PROMPT_TEMPLATE
from cot_prompt import COT_PROMPT_TEMPLATE
from pot_prompt import POT_PROMPT_TEMPLATE


# dataset = 'wikitq'
dataset = 'tabfact'
# dataset = 'finqa'

data_path = f'data/{dataset}/test.jsonl'



model = 'gpt4m'
# model = 'gpt35'
# model = 'gpt4o'

# model = 'llama'


# prompt_type = 'direct'
# prompt_type = 'cot'
prompt_type = 'pot'



MAX_WORKERS = 10

output_path = f'outputs/baselines/{dataset}/{model}/{prompt_type}/'


if not os.path.exists(output_path):
    os.makedirs(output_path)



if prompt_type == 'cot':
    REASON_PROMPT_TEMPLATE = COT_PROMPT_TEMPLATE
elif prompt_type == 'pot':
    REASON_PROMPT_TEMPLATE = POT_PROMPT_TEMPLATE
elif prompt_type == 'direct':
    REASON_PROMPT_TEMPLATE = DIRECT_PROMPT_TEMPLATE


def llm_reason(table, question):
    prompt = REASON_PROMPT_TEMPLATE.format(table=table, question=question)
    global model


    if model == 'gpt4o':
        model_name = 'gpt-4o'
    elif model == 'gpt4m':
        model_name = 'gpt-4o-mini'
    elif model == 'gpt35':
        model_name = 'gpt-3.5-turbo'
    response = get_openai_llm_response(prompt, model=model_name, json_output=False, temperature=0)
    return response, prompt


with open(data_path) as f:
    data = [json.loads(line) for line in f.readlines()]



def execute_python_code(code: str, table: list[list[str]] = None, result_var_name: str = 'answer'):
    """
    Executes a Python code string and returns the result of the last expression.
    """
    # Create a local dictionary to store the result of the execution
    namespace = {"table": table}
    try:
        # Execute the code and capture any variables in local_vars
        exec(code, namespace)
        
        # Try to return the result of the last expression, if available
        # code_result = local_vars.get(result_var_name, None)
        code_result = namespace.get(result_var_name, None)
        if code_result is None:
            return 'Error: No explicit result found.'
        return code_result
    except Exception as e:
        # Handle and return any errors that occur during execution
        return f"Error: {str(e)}"



def worker(i, D):
    if os.path.exists(f'{output_path}/{i}.json'):
        return

    table = D['table']
    question = D['question']
    gt_answer = D['answer']

    if dataset == 'tabfact':
        question = fact_to_qa(question, model='gpt-4o-mini')
    elif dataset == 'finqa':
        question = D['pre_text'] + '\n' + D['post_text'] + '\n' + question

    # while True:
    try:
        table_md = format_table(table, with_address=False)
        response, prompt = llm_reason(table_md, question)
        if prompt_type == 'direct' or prompt_type == 'cot':
            reasoning_process = response
            predicted_answer = re.sub(r'^.*?answer:', '', response.lower().replace('\n', ' ')).strip()
        elif prompt_type == 'pot':
            reasoning_process = response
            code = response.replace('```python', '').replace('```', '')
            try:
                predicted_answer = str(execute_python_code(code, result_var_name='answer') )
            except:
                predicted_answer = 'failed'

        # break
    except:
        traceback.print_exc()
        return

    output_data = {
        **D,
        'predicted_answer': predicted_answer,
        'reasoning_process': reasoning_process
    }
    with open(f'{output_path}/{i}.json', 'w') as f:
        f.write(json.dumps(output_data, indent=4, ensure_ascii=False))



with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(worker, i, D) for i, D in enumerate(data)]
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        try:
            future.result()
        except:
            traceback.print_exc()
        
