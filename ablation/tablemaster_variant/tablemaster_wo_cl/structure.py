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
    STRUCTURE_EXTRACTITON_PROMPT_TEMPLATE = '''
## Objective
You are provided with a text representation of a table in string format, detailing the content of each cell.
Your task is to identify and extract the Top Header and Key Column of the table.

## Table Definition
The table is represented by cell-value pairs, where each pair consists of a cell address and a value of the content in that cell, separated by a comma (e.g., 'A1,Year').
Multiple cells are separated by '|' (e.g., 'A1,Year|A2,Profit').
Cells may contain empty values, represented as 'A1,|A2,Profit'.

## Table
{table}

## Instructions
1. Top Header: The section at the top of the table, often spanning multiple columns horizontally, that describes the primary information presented in the table.
2. Key Column: A column where the values best represent the subject or key identifier for each row in the table, typically containing row labels or keys (e.g., year, date, number, name, etc.).
3. You should extract the top headers with address and value, like ['A1,Year', 'A2,Profit', ...].
4. key_column_index should be like 'A' or 'B' ...
5. The key column should contain meaningful values instead of id.

## Response Format
The response should be in JSON format:
```json
{{
   "topheaders": ["address1,header1", "address2,header2", ...],
   "key_column_index": "column1",
}}
```
'''
    response = get_openai_llm_response(STRUCTURE_EXTRACTITON_PROMPT_TEMPLATE.format(table=table_md), json_output=True)
    json_response = json.loads(response)
    # print('Structure extraction:', json_response)
    return json_response['topheaders'], json_response['key_column_index']

def column_lookup(table_md, topheaders, key_column_index, question):
    return [], []

def row_lookup(table_array, key_column_index, question):
    JUDGEMENT_QUESTION_PROMPT_TEMPLATE = '''
Question: {question}
Field Name: {key_column_name}
Decide if the question asks for information related to the given field `{key_column_name}`.

Provide the response in the following JSON format:
```json
{{
    "result": boolean,
}}
```
'''
    JUDGEMENT_CRITERIA_PROMPT_TEMPLATE = '''
Question: {question}
Field Name: {key_column_name}
Decide if the question explicitly mentions information of the given field `{key_column_name}`.

Provide the response in the following JSON format:
```json
{{
    "explanation": string,
    "result": boolean
}}
```
'''
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
The SQL query must be in the format of `SELECT row_id, {key_column_name} FROM Table WHERE {key_column_name} ...`,
where Table is the table name, {key_column_name} is the key column name, and ... is the criteria.
You must use the key column name in the WHERE clause.

If the information is not enough to answer the question, you should return a sql to select all rows.

## Response Format
Provide the response in the following JSON format:
```json
{{
    "sql": "SELECT row_id, {key_column_name} FROM Table WHERE {key_column_name} ..."
}}
```
'''
    key_column_name = table_array[0][column_to_index(key_column_index)]
    # remove non-alphanumeric characters for sql
    if key_column_name.strip() == '#':
        key_column_name = 'ID'
    key_column_name = re.sub(r'[^a-zA-Z0-9]', '', key_column_name).lower().strip()
    if key_column_name == '':
        key_column_name = 'Empty Header'

    # judge if the question asks for information related to the given field
    response = get_openai_llm_response(JUDGEMENT_QUESTION_PROMPT_TEMPLATE.format(question=question, key_column_name=key_column_name), json_output=True)
    has_criteria = not json.loads(response)['result']

    # judge if the question explicitly contains information or values of the given field
    if has_criteria:
        response = get_openai_llm_response(JUDGEMENT_CRITERIA_PROMPT_TEMPLATE.format(question=question, key_column_name=key_column_name), json_output=True)
        has_criteria = json.loads(response)['result']

    # construct the table with row id
    tmp_table_array = []
    for row_id, row in enumerate(table_array):
        if row_id == 0:
            tmp_table_array.append(['row_id', key_column_name])
        else:
            tmp_table_array.append([row_id, row[column_to_index(key_column_index)]])
    tmp_table_np_array = np.array(tmp_table_array)

    table_md = format_table(tmp_table_np_array[:TABLE_PEEK_SIZE], with_address=False)

    # create a database
    T = pd.DataFrame(tmp_table_np_array[1:], columns=tmp_table_np_array[0])

    if has_criteria:
        response = get_openai_llm_response(SQL_GENERATION_PROMPT_TEMPLATE.format(table=table_md, question=question, key_column_name=key_column_name), json_output=True)
        sql = json.loads(response)['sql'].replace('Table', 'T')
    else:
        sql = f'SELECT row_id, {key_column_name} FROM T'

    if 'row_id' not in sql:
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
    if task == 'qa':
        sql, selected_row_indices = row_lookup_all(table_array, key_column_index, question)
        # sql, selected_row_indices = row_lookup(table_array, key_column_index, question)
    elif task == 'fact':
        sql, selected_row_indices = row_lookup_all(table_array, key_column_index, question)
        # sql, selected_row_indices = row_lookup(table_array, key_column_index, question)
    return topheaders, key_column_index, selected_column_indices, ranked_column_indices, sql, selected_row_indices



if __name__ == '__main__':
    # with open('./test_tables/test.json', 'r') as f:
    with open('./test_tables/test2.json', 'r') as f:
        data = json.load(f)
    table_array = data['table']
    question = data['question']
    topheaders, key_column_index, selected_column_indices, ranked_column_indices, sql, selected_row_indices = table_structure_understanding(table_array, question)
    print(topheaders, key_column_index, selected_column_indices, ranked_column_indices, sql, selected_row_indices)
