import json
import sys
import glob

sys.path.append('.')

from table_utils import format_table
from azure_openai_api import get_openai_llm_response
import azure_openai_api
from evaluate.evaluator import eval_qa




output_path = 'outputs/main/wikitq/tablemaster-4m-new'

files = glob.glob(f'{output_path}/*.json')
files = sorted(files, key=lambda x: int(x.split('/')[-1].split('.')[0]))




# print(format_table(table))
# print(data['question'])

# print(data['answer'])

for i, file in enumerate(files):
    with open(file, 'r') as f:
        data = json.load(f)
        if not eval_qa(data['answer'], data['predicted_answer']):
            continue
        if data['reasoning_process']['symbolic_reasoning_process'] == '':
            continue
        
        print(file)
        print(data['question'])
        print(data['answer'])
        print(format_table(data['table']))
        print(data['reasoning_process'])
        print('--------------------------------')
        input()



