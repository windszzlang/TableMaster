## Table Content Understanding
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




def subtable_extraction(table_array, row_indices, column_indices):
    rows = [int(i) - 1 for i in row_indices]
    columns = [column_to_index(j) for j in column_indices]
    # columns = [c for c in columns if c < table_np_array.shape[1]]
    table_np_array = np.array(table_array)
    assert max(rows) < table_np_array.shape[0] and max(columns) < table_np_array.shape[1], f'Subtable Extraction Error: The row or column indices are out of bounds. {max(rows)} < {table_np_array.shape[0]} or {max(columns)} < {table_np_array.shape[1]}'
    subtable_array = table_np_array[np.ix_(rows, columns)].tolist()
    # print('Subtable Extraction:', subtable_array)
    return subtable_array


def table_verbalization(table_md):
    return 'Not provided'
#     TABLE_VERBALIZATION_PROMPT_TEMPLATE = '''
# ## Objective
# You are provided with a table in string format.
# Your task is to convert the table into a detailed text description.

# ## Table
# {table}

# ## Instructions
# 1. Provide a detailed description of the table, covering all rows and columns.
# 2. Include every detail and numerical value without omitting or summarizing.
# 3. Use external knowledge only to enhance clarity, while staying faithful to the table's content.
# 4. If the table only contains headers and no rows, it should be described as an empty table.

# Now, please provide the verbalized description of the table:
# '''
#     verbalized_table = get_openai_llm_response(TABLE_VERBALIZATION_PROMPT_TEMPLATE.format(table=table_md), json_output=False)
#     # print('Table verbalization:', verbalized_table)
#     # verbalized_table = verbalized_table[:100] + '...' # truncate the response to 1000 characters
#     return verbalized_table


def information_estimation(table_md, topheader_info, question):
    INFORMATION_ESTIMATION_PROMPT_TEMPLATE = '''
## Objective
You are provided with information from a table and a question related to the table.
Your task is to estimate whether the current information of the table can answer the question.

## Table Information
Top Headers: {topheader_info}
Table Content:
{table}

## Question
{question}

## Response Format
The response should be in JSON format:
```json
{{
   "results": True of False
}}
```
'''
    response = get_openai_llm_response(INFORMATION_ESTIMATION_PROMPT_TEMPLATE.format(table=table_md, topheader_info=topheader_info, question=question), json_output=True)
    json_response = json.loads(response)
    return json_response['results']

def table_content_understanding(table_array, question, selected_row_indices, selected_column_indices, ranked_column_indices):
    candidate_row_indices = [idx for idx in ranked_column_indices if idx not in selected_column_indices]
    final_selected_column_indices = selected_column_indices.copy()
    while True:
        subtable_array = subtable_extraction(table_array, selected_row_indices, final_selected_column_indices)
        table_md = format_table(subtable_array, with_address=False)
        verbalized_table = table_verbalization(table_md)
        topheader_info = subtable_array[0]
        estimation_result = information_estimation(table_md, topheader_info, question)
        if estimation_result or len(candidate_row_indices) == 0:
            break
        else:
            final_selected_column_indices.append(candidate_row_indices.pop(0))
    
    # print('Final selected column indices:', final_selected_column_indices)
    return verbalized_table, subtable_array, final_selected_column_indices



if __name__ == '__main__':
    with open('./test_tables/test2.json', 'r') as f:
        data = json.load(f)
    table_array = data['table']
    question = data['question']
    selected_row_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    selected_column_indices = ['A', 'C', 'E']
    ranked_column_indices = ['A', 'B', 'E', 'D', 'C']
    verbalized_table, subtable_array, final_selected_column_indices = table_content_understanding(table_array, question, selected_row_indices, selected_column_indices, ranked_column_indices)
    print(verbalized_table, subtable_array, final_selected_column_indices)
