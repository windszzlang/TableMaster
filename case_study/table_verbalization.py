import json
import sys

sys.path.append('.')

from table_utils import format_table


with open('test_tables/test.json', 'r') as f:
    data = json.load(f)

table = data['table']

print(format_table(table))
print(data['question'])

print(data['answer'])