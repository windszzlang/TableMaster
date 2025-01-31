## Table Structure Understanding
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



TABLE_PEEK_SIZE = 10000
# TABLE_PEEK_SIZE = 100
# TABLE_PEEK_SIZE = 11


def structure_extraction(table_md):
    return [], ''

def column_lookup(table_md, topheaders, key_column_index, question):
    COLUMN_RANKING_PROMPT_TEMPLATE = '''
## Objective
You are provided with information of a table and a question related to the table.
Your task is to rank the column indices based on the relevance to the question.

## Table Definition
The table is represented by cell-value pairs, where each pair consists of a cell address and a value of the content in that cell, separated by a comma (e.g., 'A1,Year').
Multiple cells are separated by '|' (e.g., 'A1,Year|A2,Profit').
Cells may contain empty values, represented as 'A1,|A2,Profit'.

## Table Information
Table: 
{table}

## Question
{question}

## Instructions
1. The column indices must only contain letters, like ['A', 'B', 'C', ...].
2. You should first rank all the column indices based on the relevance to the question.
3. Your output should contain all the column indices.

## Response Format
The response should be in JSON format:
```json
{{
   "ranked_column_indices": ["column indexA", "column indexB", ...]
}}
```
'''
    COLUMN_LOOKUP_PROMPT_TEMPLATE = '''
## Objective
You are provided with information of a table and a question related to the table.
Your task is to lookup the column indices that are needed to answer the question based on the table.

## Table Definition
The table is represented by cell-value pairs, where each pair consists of a cell address and a value of the content in that cell, separated by a comma (e.g., 'A1,Year').
Multiple cells are separated by '|' (e.g., 'A1,Year|A2,Profit').
Cells may contain empty values, represented as 'A1,|A2,Profit'.

## Table Information
Table: 
{table}

## Question
{question}

## Instructions
1. The column indices must only contain letters, like ['A', 'B', 'C', ...].
2. Your output of the column indices should not any contain number, like ['A1', 'B2', 'C1', ...].
3. Your output of the column indices should not contain the column name.
4. You should select the column that are relevant and necessary to answer the question.

## Response Format
The response should be in JSON format:
```json
{{
   "selected_column_indices": ["column indexA", "column indexB", ...]
}}
```
'''
    response = get_openai_llm_response(COLUMN_RANKING_PROMPT_TEMPLATE.format(table=table_md, question=question), json_output=True)
    ranked_column_indices = json.loads(response)['ranked_column_indices']
    response = get_openai_llm_response(COLUMN_LOOKUP_PROMPT_TEMPLATE.format(table=table_md, question=question), json_output=True)
    selected_column_indices = json.loads(response)['selected_column_indices']
    # remove number of poor gpt-3.5-turbo
    for i in range(len(selected_column_indices)):
        selected_column_indices[i] = re.sub(r'\d+', '', selected_column_indices[i])
    selected_column_indices = list(set(selected_column_indices))

    # print('Column lookup:', ranked_column_indices, selected_column_indices)
    return ranked_column_indices, selected_column_indices


def row_lookup_all(table_array, key_column_index, question):
    SQL_GENERATION_PROMPT_TEMPLATE = '''
## Objective
You are provided with information of a table and a question related to the table.
Your task is to generate a SQL query that can be used to find the rows that answer the question.

## Table Information
Part of Table:
{table}

## Question
{question}

## Instructions
1. The SQL query must be in the format of `SELECT XXX, ... FROM Table WHERE XXX ...`,
where Table is the table name, XXX is the column name, and WHERE... is the criteria.
2. If the information is not enough to answer the question, you should return a sql to select all rows.
3. Do not give complex sql query, just simple query to select rows.
4. Use this SQL query only to select relevant rows, not for getting the final answer.

## Response Format
Provide the response in the following JSON format:
```json
{{
    "sql": "SELECT XXX, ... FROM Table WHERE XXX ..."
}}
```
'''
    table_md = format_table(table_array[:TABLE_PEEK_SIZE], with_address=False)

    # construct the table with row id
    tmp_table_array = []
    for row_id, row in enumerate(table_array):
        if row_id == 0:
            tmp_table_array.append(['row_id'] + row)
        else:
            tmp_table_array.append([row_id] + row)
    tmp_table_np_array = np.array(tmp_table_array)

    # create a database
    T = pd.DataFrame(tmp_table_np_array[1:], columns=tmp_table_np_array[0])

    response = get_openai_llm_response(SQL_GENERATION_PROMPT_TEMPLATE.format(table=table_md, question=question), json_output=True)
    sql = json.loads(response)['sql'].replace('Table', 'T')

    if ' * ' not in sql:
        sql = sql.replace('SELECT', 'SELECT row_id,')

    try:
        sql_result = ps.sqldf(sql)
    except Exception as e:
        sql_result = T

    row_indices = [1]
    for row in sql_result['row_id'].tolist():
        row_indices.append(int(row) + 1)     # + offset of top headers
    # print('Row lookup:', row_indices)
    return sql, row_indices


def table_structure_understanding(table_array, question, task='qa'):
    table_peek_md = format_table(table_array[:TABLE_PEEK_SIZE], with_address=True)
    topheaders, key_column_index = structure_extraction(table_peek_md)
    ranked_column_indices, selected_column_indices = column_lookup(table_peek_md, topheaders, key_column_index, question)
    sql, selected_row_indices = row_lookup_all(table_array, key_column_index, question)
    return topheaders, key_column_index, selected_column_indices, ranked_column_indices, sql, selected_row_indices



if __name__ == '__main__':
    # with open('./test_tables/test.json', 'r') as f:
    with open('./test_tables/test2.json', 'r') as f:
        data = json.load(f)
    table_array = data['table']
    question = data['question']
    topheaders, key_column_index, selected_column_indices, ranked_column_indices, sql, selected_row_indices = table_structure_understanding(table_array, question)
    print(topheaders, key_column_index, selected_column_indices, ranked_column_indices, sql, selected_row_indices)
