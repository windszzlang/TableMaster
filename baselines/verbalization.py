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

from direct_prompt import DIRECT_PROMPT_TEMPLATE
from cot_prompt import COT_PROMPT_TEMPLATE
from pot_prompt import POT_PROMPT_TEMPLATE

from evaluate.evaluator import eval_qa, eval_fact


dataset = 'wikitq'
# dataset = 'tabfact'

data_path = f'data/{dataset}/test.jsonl'



# model = 'gpt4m'
# model = 'gpt4o'
model = 'gpt35'

prompt_type = 'verbal'
# prompt_type = 'text' # bad


MAX_WORKERS = 10

output_path = f'outputs/baselines/{dataset}/{model}/{prompt_type}/'


if not os.path.exists(output_path):
    os.makedirs(output_path)




VERBALIZATION_PROMPT_TEMPLATE = '''## Objective
You are provided with a table in string format.
Your task is to convert the table into a detailed text description.

## Table
{table}

## Instructions
1. Provide a comprehensive description of the table.
2. Include all details and numerical values from the table in your response.
3. Do not omit or summarize any information from the table.
4. You may use external knowledge to enhance your understanding of the table, but the response must remain faithful to the table's content.

Now, please provide the verbalized description of the table:
'''


DIRECT_PLUS_PROMPT_TEMPLATE = """## Objective
You are provided with a table, a verbalized table, and a question related to the table.
Your task is to answer the question directly based on the table.

## Table
{table}

## Verbalized Table
{verbalized_table}

## Question
{question}

The answer should be short and simple. It can be a number, a word, or a phrase in the table, but not a full sentence.
Now, answer the question directly:
Answer: """


DIRECT_TEXT_PROMPT_TEMPLATE = """## Objective
You are provided with a verbalized table, and a question related to the table.
Your task is to answer the question directly based on the table.

## Verbalized Table
{verbalized_table}

## Question
{question}

The answer should be short and simple. It can be a number, a word, or a phrase in the table, but not a full sentence.
Now, answer the question directly:
Answer: """


def llm_reason(table, question):
    global model
    if model == 'gpt4o':
        model_name = 'gpt-4o'
    elif model == 'gpt4m':
        model_name = 'gpt-4o-mini'
    elif model == 'gpt35':
        model_name = 'gpt-3.5-turbo'

    # verbalize the table
    prompt = VERBALIZATION_PROMPT_TEMPLATE.format(table=table)
    verbalized_table = get_openai_llm_response(prompt, model=model_name, json_output=False, temperature=0)

    # reason the question
    if prompt_type == 'verbal':
        prompt = DIRECT_PLUS_PROMPT_TEMPLATE.format(table=table, verbalized_table=verbalized_table, question=question)
    elif prompt_type == 'text':
        prompt = DIRECT_TEXT_PROMPT_TEMPLATE.format(verbalized_table=verbalized_table, question=question)

    answer = get_openai_llm_response(prompt, model=model_name, json_output=False, temperature=0)
    return answer, verbalized_table



with open(data_path) as f:
    data = [json.loads(line) for line in f.readlines()]




def worker(i, D):
    if os.path.exists(f'{output_path}/{i}.json'):
        return

    table = D['table']
    question = D['question']
    gt_answer = D['answer']

    # while True:
    try:
        table_md = format_table(table, with_address=False)
        answer, verbalized_table = llm_reason(table_md, question)
        reasoning_process = verbalized_table
        predicted_answer = re.sub(r'^.*?answer:', '', answer.lower().replace('\n', ' ')).strip()
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
        
