POT_PROMPT_TEMPLATE = """## Objective
You are provided with a table and a question related to the table.
Your task is to answer the question based on the table by writing python code as a solution.

## Table
{table}

## Reasoning Instructions
1. You must use executable python code to solve the question.
2. The final answer should be variable named "answer" in the code.
3. Do not execute the code in the response.
4. The python code should be in the following format:
```python
# your code here
```

Now, answer the question by writing python code as a solution:
Question: {question}
```python
"""

