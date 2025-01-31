import os
import re
import sys
import json
import traceback
from tqdm import tqdm
import concurrent.futures

sys.path.append('./')
from table_utils import format_table, remove_none_in_table
import azure_openai_api
from azure_openai_api import get_openai_llm_response, get_o1_llm_response
from utils import fact_to_qa
from evaluate.evaluator import eval_qa, eval_fact


dataset = 'wikitq'
# dataset = 'tabfact'

data_path = f'data/{dataset}/test.jsonl'



model = 'gpt4m'
# model = 'gpt35'
# model = 'gpt4o'


if model == 'gpt4m':
    azure_openai_api.GLOBAL_MODEL = 'gpt-4o-mini'
elif model == 'gpt4o':
    azure_openai_api.GLOBAL_MODEL = 'gpt-4o'
elif model == 'gpt35':
    azure_openai_api.GLOBAL_MODEL = 'gpt-3.5-turbo'


prompt_type = 'guided_pot'


MAX_WORKERS = 10

output_path = f'outputs/baselines/{dataset}/{model}/{prompt_type}/'


if not os.path.exists(output_path):
    os.makedirs(output_path)


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


# 35
# Keep the reasoning process concise and clear.
# Control the number of steps in the reasoning process in the range of 1-5.


def text_guided_symbolic_reasoning(table_array, question):
    GUIDANCE_GENERATION_PROMPT_TEMPLATE = '''
## Objective
You are provided with a table, and a question related to the table.
Your task is to give a step-by-step guidance to answer the question based on the table.

## Table Definition
The table is represented by cell-value pairs, where each pair consists of a cell address and a value of the content in that cell, separated by a comma (e.g., 'A1,Year').
Multiple cells are separated by '|' (e.g., 'A1,Year|A2,Profit').
Cells may contain empty values, represented as 'A1,|A2,Profit'.

## Table
{table}

You do not need to give the answer. You need to give a reasoning process as a guidance that will be used later.
Keep the reasoning process concise and clear.
Control the number of steps in the reasoning process in the range of 1-5.

## Response Format
The response should be a list of steps:
1. xxx
2. xxx
...

Now, give me the guidance to answer the question step by step:
Question: {question}
'''

# gpt4m
# 4. The answer should be short and simple. It can be a number, a word, or a phrase in the table, but not a full sentence.

    SYMBOLIC_REASONING_PROMPT_TEMPLATE = '''
## Objective
You are provided with a table, a guidance, and a question related to the table.
Your task is to generate Python code that answers the question using the table and the guidance as a guide.

## Table Definition
The table is represented by cell-value pairs, where each pair consists of a cell address and a value of the content in that cell, separated by a comma (e.g., 'A1,Year').
Multiple cells are separated by '|' (e.g., 'A1,Year|A2,Profit').
Cells may contain empty values, represented as 'A1,|A2,Profit'.

## Table
{table}

## Guidance
{textual_guidance}

## Question
{question}

## Instructions
1. The actual data of the table is stored in the variable `table` as a list of lists.
2. The result should be store in the variable `answer` as a string and do not need to print it.
3. You need to generate Python code within ```python``` code block.
4. The answer should be short and simple. It can be a number, a word, or a phrase in the table, but not a full sentence.

Now, give me the executable python code to answer the question:
```python
table = {table_array}
'''
    table_md = format_table(table_array, with_address=True)
    textual_guidance = get_openai_llm_response(prompt=GUIDANCE_GENERATION_PROMPT_TEMPLATE.format(table=table_md, question=question), json_output=False)
    
    retry_count = -1
    while True:
        retry_count += 1
        if retry_count > 3:
            code = 'None'
            code_result = 'None'
            break

        code = get_openai_llm_response(SYMBOLIC_REASONING_PROMPT_TEMPLATE.format(table=table_md, textual_guidance=textual_guidance, question=question, table_array=table_array), json_output=False)
        
        match = re.search(r'```python(.*?)```', code, re.DOTALL)
        if match:
            code = match.group(1).strip()
        else:
            continue
            
        code = code.replace('```python', '').replace('```', '')

        code_result = execute_python_code(code, table=table_array, result_var_name='answer')
        # print(code)
        if isinstance(code_result, str) and 'Error' in code_result:
            # print(f'Python code execution failed ({code_result}), retrying {retry_count} times...')
            continue
        else:
            break
    return textual_guidance, code, code_result


with open(data_path) as f:
    data = [json.loads(line) for line in f.readlines()]




def worker(i, D):
    if os.path.exists(f'{output_path}/{i}.json'):
        return

    table = D['table']
    question = D['question']
    gt_answer = D['answer']

    table = remove_none_in_table(table, replace_with='None')

    if dataset == 'tabfact':
        question = fact_to_qa(question, model='gpt-4o-mini')

    # while True:
    try:
        textual_guidance, code, code_result = text_guided_symbolic_reasoning(table, question)
        predicted_answer = code_result
        reasoning_process = textual_guidance + '\n' + code
        # break
    except:
        traceback.print_exc()
        return


    # predicted_answer = reasoning_process = 'violation'

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
        
