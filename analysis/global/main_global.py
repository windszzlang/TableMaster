import os
import sys
import json
from tqdm import tqdm
from glob import glob
import concurrent.futures
import traceback

sys.path.append('.')
from evaluate.evaluator import eval_qa
from azure_openai_api import get_openai_llm_response


dataset = 'wikitq'


## baselines

model = 'gpt4m'


output_path = f'outputs/analysis/global'

if not os.path.exists(output_path):
    os.makedirs(output_path)


wo_global_prediction_path = 'outputs/main/wikitq/tablemaster-4m-new'
w_global_prediction_path = 'outputs/main/wikitq/tablemaster-4m'



wo_global_prediction_files = glob(f'{wo_global_prediction_path}/*.json')
wo_global_prediction_files = sorted(wo_global_prediction_files, key=lambda x: int(x.split('/')[-1].split('.')[0]))

w_global_prediction_files = glob(f'{w_global_prediction_path}/*.json')
w_global_prediction_files = sorted(w_global_prediction_files, key=lambda x: int(x.split('/')[-1].split('.')[0]))




def llm_eval_results(answer):
    EVAL_PROMPT_TEMPLATE = '''You are provided with an answer to a question.
Answer: {answer}

Evaluate whether the answer indicates there is enough information to address the question:
- If the answer provides sufficient information to answer the question, the result is `True`.
- If the answer does not provide enough information to answer the question, the result is `False`.

Please respond in the following JSON format:
{{
    "result": true or false
}}
'''
    prompt = EVAL_PROMPT_TEMPLATE.format(answer=answer)
    response = get_openai_llm_response(prompt, model='gpt-4o-mini', json_output=True)
    result = json.loads(response)['result']
    return result





def worker(i, wo_global_prediction_file):
    if os.path.exists(f'{output_path}/{i}.json'):
        return

    with open(wo_global_prediction_file, 'r') as f:
        pred = json.load(f)
    answer = pred['predicted_answer']
    result = llm_eval_results(answer)

    with open(f'{output_path}/{i}.json', 'w') as f:
        json.dump(result, f)


with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(worker, i, wo_global_prediction_file) for i, wo_global_prediction_file in enumerate(wo_global_prediction_files)]
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        try:
            future.result()
        except Exception as e:
            print()
            traceback.print_exc()



groundtruth = []
final_prediction = []
eval_results = []

for i, (wo_global_prediction_file, w_global_prediction_file) in enumerate(zip(wo_global_prediction_files, w_global_prediction_files)):
    with open(wo_global_prediction_file, 'r') as f:
        wo_global_pred = json.load(f)['predicted_answer']
    with open(w_global_prediction_file, 'r') as f:
        w_global_pred = json.load(f)['predicted_answer']
    with open(w_global_prediction_file, 'r') as f:
        groundtruth.append(json.load(f)['answer'])

    with open(f'{output_path}/{i}.json') as f:
        eval_result = json.load(f)
    eval_results.append(eval_result)
    if eval_result == True:
        final_prediction.append(wo_global_pred)
    else:
        final_prediction.append(w_global_pred)

total = 0
correct = 0
for pred, gold in zip(final_prediction, groundtruth):
    total += 1
    res = eval_qa(pred, gold)
    if res:
        correct += 1




print('********************')
print(f'Rectified wo Global')
print('Total:', total)
print('Correct:', correct)
print('Accuracy:', correct / total)
print('Num of True:', sum(eval_results))
