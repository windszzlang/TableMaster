## Table Reasoning for Question Answering
import os
import sys
import json
import traceback
from tqdm import tqdm
import concurrent.futures
import numpy as np
import pandas as pd
import pandasql as ps
import re

sys.path.append('./')
from table_utils import format_table, column_to_index, cell_to_index
from azure_openai_api import get_openai_llm_response



CODE_MAX_RETRIES = 3


def reasoning_strategy_assessment(table_array, question):
    program_aided = True
    return program_aided


def textual_reasoning(table_array, verbalized_table, question):
    TEXTUAL_REASONING_PROMPT_TEMPLATE = '''
## Objective
You are provided with a table, a verbalization of the table, and a question related to the table.
Your task is to reason step by step to answer the question based on the table.

## Table Definition
The table is represented by cell-value pairs, where each pair consists of a cell address and a value of the content in that cell, separated by a comma (e.g., 'A1,Year').
Multiple cells are separated by '|' (e.g., 'A1,Year|A2,Profit').
Cells may contain empty values, represented as 'A1,|A2,Profit'.

## Table
{table}

## Verbalized Table
{verbalized_table}

The answer should be short and simple. It can be a number, a word, or a phrase in the table, but not a full sentence.
Your response should end with `Answer: xxx` (answer to the question).

Now, give me the answer step by step:
Question: {question}
'''
    table_md = format_table(table_array, with_address=True)
    textual_reasoning_process = get_openai_llm_response(TEXTUAL_REASONING_PROMPT_TEMPLATE.format(table=table_md, verbalized_table=verbalized_table, question=question), json_output=False)
    return textual_reasoning_process


def textual_reasoning_for_fact(table_array, verbalized_table, question):
    TEXTUAL_REASONING_PROMPT_TEMPLATE = '''
## Objective
You are provided with a table, a verbalization of the table, and a question related to the table.
Your task is to reason step by step to answer the question based on the table.

## Table Definition
The table is represented by cell-value pairs, where each pair consists of a cell address and a value of the content in that cell, separated by a comma (e.g., 'A1,Year').
Multiple cells are separated by '|' (e.g., 'A1,Year|A2,Profit').
Cells may contain empty values, represented as 'A1,|A2,Profit'.

## Table
{table}

## Verbalized Table
{verbalized_table}

Your response should end with `Answer: true` or `Answer: false` (answer to the question).
If the table only contain headers and no rows, it indicates there is no information available for this question, therefore the answer should be "false."

Now, give me the answer step by step:
Question: {question}
'''
    table_md = format_table(table_array, with_address=True)
    textual_reasoning_process = get_openai_llm_response(TEXTUAL_REASONING_PROMPT_TEMPLATE.format(table=table_md, verbalized_table=verbalized_table, question=question), json_output=False)
    return textual_reasoning_process


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


def text_guided_symbolic_reasoning(table_array, verbalized_table, question):
    GUIDANCE_GENERATION_PROMPT_TEMPLATE = '''
## Objective
You are provided with a table, a verbalized table, and a question related to the table.
Your task is to give a step-by-step guidance to answer the question based on the table.

## Table Definition
The table is represented by cell-value pairs, where each pair consists of a cell address and a value of the content in that cell, separated by a comma (e.g., 'A1,Year').
Multiple cells are separated by '|' (e.g., 'A1,Year|A2,Profit').
Cells may contain empty values, represented as 'A1,|A2,Profit'.

## Table
{table}

## Verbalized Table
{verbalized_table}


You do not need to give the answer. You need to give a reasoning process as a guidance that will be used later.

## Response Format
The response should be a list of steps:
1. xxx
2. xxx
...

Now, give me the guidance to answer the question step by step:
Question: {question}
'''
    SYMBOLIC_REASONING_PROMPT_TEMPLATE = '''
## Objective
You are provided with a table, a verbalized table, a guidance, and a question related to the table.
Your task is to generate Python code that answers the question using the table and the guidance as a guide.

## Table Definition
The table is represented by cell-value pairs, where each pair consists of a cell address and a value of the content in that cell, separated by a comma (e.g., 'A1,Year').
Multiple cells are separated by '|' (e.g., 'A1,Year|A2,Profit').
Cells may contain empty values, represented as 'A1,|A2,Profit'.

## Table
{table}

## Verbalized Table
{verbalized_table}

## Guidance
{textual_guidance}

## Question
{question}

## Instructions
1. The actual data of the table is stored in the variable `table` as a list of lists.
2. The result should be store in the variable `answer` as a string and do not need to print it.
3. You need to generate Python code within ```python``` code block.

Now, give me the executable python code to answer the question:
```python
table = {table_array}
'''
    table_md = format_table(table_array, with_address=True)
    textual_guidance = get_openai_llm_response(GUIDANCE_GENERATION_PROMPT_TEMPLATE.format(table=table_md, verbalized_table=verbalized_table, question=question), json_output=False)
    
    retry_count = -1
    while True:
        retry_count += 1
        if retry_count > CODE_MAX_RETRIES:
            code = 'None'
            code_result = 'None'
            break
        code = get_openai_llm_response(SYMBOLIC_REASONING_PROMPT_TEMPLATE.format(table=table_md, verbalized_table=verbalized_table, textual_guidance=textual_guidance, question=question, table_array=table_array), json_output=False)
        
        # match = re.search(r'```python(.*?)```', code, re.DOTALL)
        # if match:
            # code = match.group(1).strip()
        # else:
            # continue
            
        code = code.replace('```python', '').replace('```', '')
        code_result = execute_python_code(code, table=table_array, result_var_name='answer')
        # print(code)
        if isinstance(code_result, str) and 'Error' in code_result:
            # print(f'Python code execution failed ({code_result}), retrying {retry_count} times...')
            continue
        else:
            break
    return textual_guidance, code, code_result


def answer_formatting(table_array, question, textual_reasoning_process, symbolic_reasoning_process):
    ANSWER_FORMATTING_PROMPT_TEMPLATE = '''
## Objective
You are provided with a process of text-guided reasoning with programming and a question related to the table.
Your task is to answer the question using the reasoning process.

## Table Definition
The table is represented by cell-value pairs, where each pair consists of a cell address and a value of the content in that cell, separated by a comma (e.g., 'A1,Year').
Multiple cells are separated by '|' (e.g., 'A1,Year|A2,Profit').
Cells may contain empty values, represented as 'A1,|A2,Profit'.

## Table
{table}

## Textual Reasoning Process
{textual_reasoning_process}

## Programmed Reasoning Process
{symbolic_reasoning_process}

The answer should be short and simple. It can be a number, a word, or a phrase in the table, but not a full sentence.
Your response should be in the format of `Answer: xxx` (answer to the question).

Question: {question}
Answer:
'''
    table_md = format_table(table_array, with_address=True)
    final_answer = get_openai_llm_response(ANSWER_FORMATTING_PROMPT_TEMPLATE.format(table=table_md, textual_reasoning_process=textual_reasoning_process, symbolic_reasoning_process=symbolic_reasoning_process, question=question), json_output=False)
    return final_answer


# adaptive_program_aided_reasoning
def table_reasoning_for_qa(table_array, verbalized_table, question, task='qa'):
    program_aided = reasoning_strategy_assessment(table_array, question)

    if task == 'qa':
        if program_aided:
            textual_reasoning_process, code, code_result = text_guided_symbolic_reasoning(table_array, verbalized_table, question)
            symbolic_reasoning_process = f'Code:\n```python\n{code}\n```\nCode Result:{code_result}'
            final_answer = answer_formatting(table_array, question, textual_reasoning_process, symbolic_reasoning_process)
        else:
            textual_reasoning_process = textual_reasoning(table_array, verbalized_table, question)
            symbolic_reasoning_process = ''
            final_answer = re.sub(r'^.*?answer:', '', textual_reasoning_process.lower().replace('\n', ' ')).strip()
    elif task == 'fact':
        if len(table_array) == 1:
            return 'false', 'reject reasoning for empty table', ''
        question += ' (answer in true or false)'
        if program_aided:
            textual_reasoning_process, code, code_result = text_guided_symbolic_reasoning(table_array, verbalized_table, question)
            symbolic_reasoning_process = f'Code:\n```python\n{code}\n```\nCode Result:{code_result}'
            final_answer = code_result
        else:
            textual_reasoning_process = textual_reasoning_for_fact(table_array, verbalized_table, question)
            symbolic_reasoning_process = ''
            final_answer = re.sub(r'^.*?answer:', '', textual_reasoning_process.lower().replace('\n', ' ')).strip()
    return final_answer, textual_reasoning_process, symbolic_reasoning_process



if __name__ == '__main__':
    with open('./test_tables/test2.json', 'r') as f:
        data = json.load(f)
    # table_array = data['table']
    table_array = [['Year', 'Nominated work', 'Result'], ['2007', 'Leona Lewis', 'Won'], ['2007', '"Bleeding Love"', 'Won'], ['2008', 'Leona Lewis', 'Won'], ['2008', 'Leona Lewis', 'Won'], ['2008', 'Leona Lewis', 'Won'], ['2008', 'Leona Lewis', 'Won'], ['2008', '"Bleeding Love"', 'Won'], ['2008', '"Bleeding Love"', 'Won'], ['2008', 'Leona Lewis', 'Won'], ['2008', 'Leona Lewis', 'Won'], ['2008', 'Leona Lewis', 'Won'], ['2008', '"Bleeding Love"', 'Won'], ['2008', '"Spirit"', 'Nominated'], ['2008', 'Leona Lewis', 'Won'], ['2009', 'Leona Lewis', 'Nominated'], ['2009', 'Leona Lewis', 'Won'], ['2009', 'Leona Lewis', 'Won'], ['2009', '"Bleeding Love"', 'Won'], ['2009', 'Leona Lewis', 'Won'], ['2009', '"Bleeding Love"', 'Won'], ['2009', 'Leona Lewis', 'Won'], ['2009', 'Leona Lewis', 'Won']]
    question = data['question']
    example = 'Question: how many ships were launched in the year 1944?\nAnswer: 9'
    verbalized_table = 'The table provides a list of nominations and results for the artist Leona Lewis and her songs "Bleeding Love" and "Spirit" over the years 2007, 2008, and 2009. In 2007, Leona Lewis won for her work, and specifically for the song "Bleeding Love". Moving on to 2008, Leona Lewis continued her winning streak with multiple wins for her work and the song "Bleeding Love". Additionally, she was nominated for the song "Spirit" in the same year. In 2009, Leona Lewis was nominated for her work and won for both "Bleeding Love" and her other songs. Overall, Leona Lewis had a successful run with multiple wins and nominations for her music over the years.'
    final_answer, textual_reasoning_process, symbolic_reasoning_process = table_reasoning_for_qa(table_array, verbalized_table, question)
    print(final_answer)
    print(textual_reasoning_process)
    print(symbolic_reasoning_process)