COT_PROMPT_TEMPLATE = """## Objective
You are provided with a table and a question related to the table.
Your task is to answer the question step by step based on the table.

## Table
{table}

## Question
{question}

The answer should be short and simple. It can be a number, a word, or a phrase in the table, but not a full sentence.
Your response should end with `Answer: xxx` (answer to the question).
Now, answer the question step by step:
"""

