import os
import sys
import json
from tqdm import tqdm
from glob import glob


sys.path.append('.')
from evaluate.evaluator import eval_qa
from azure_openai_api import get_openai_llm_response


dataset = 'wikitq'


## baselines

# model = 'gpt35'
model = 'gpt4m'
# model = 'gpt4o'



output_path = f'outputs/baselines/{dataset}/{model}/selfeval/'

if not os.path.exists(output_path):
    os.makedirs(output_path)



cot_prediction = []
cot_reasoning = []
pot_prediction = []
pot_reasoning = []

for prompt_type in ['cot', 'pot']:
    prediction_path = f'outputs/baselines/{dataset}/{model}/{prompt_type}/'

    prediction_files = glob(f'{prediction_path}/*.json')
    prediction_files = sorted(prediction_files, key=lambda x: int(x.split('/')[-1].split('.')[0]))

    tmp_question = []
    tmp_groundtruth = []
    tmp_table = []

    for i, prediction_file in enumerate(prediction_files):
        # print(prediction_file)
        with open(prediction_file, 'r') as f:
            pred = json.load(f)
        if prompt_type == 'cot':
            cot_prediction.append(pred['predicted_answer'])
            cot_reasoning.append(pred['reasoning_process'])
        elif prompt_type == 'pot':
            pot_prediction.append(pred['predicted_answer'])
            pot_reasoning.append(pred['reasoning_process'])
        tmp_question.append(pred['question'])
        tmp_groundtruth.append(pred['answer'])
        tmp_table.append(pred['table'])

groundtruth = tmp_groundtruth
question = tmp_question
table = tmp_table
correct = 0
total = 0



def llm_self_eval(question, table, cot_prediction, cot_reasoning, pot_prediction, pot_reasoning):
    SELF_EVAL_PROMPT_TEMPLATE = '''
Question: {question}

Table: {table}

Method 1 Solution: {cot_prediction}
Method 1 Reasoning: {cot_reasoning}

Method 2 Solution: {pot_prediction}
Method 2 Reasoning: {pot_reasoning}

Please evaluate which method is better.
Respond in the following JSON format:
{{
    "better_method": 1 or 2
}}
'''
    prompt = SELF_EVAL_PROMPT_TEMPLATE.format(question=question, table=table, cot_prediction=cot_prediction, cot_reasoning=cot_reasoning, pot_prediction=pot_prediction, pot_reasoning=pot_reasoning)
    global model
    if model == 'gpt4m':
        model_name = 'gpt-4o-mini'
    elif model == 'gpt4o':
        model_name = 'gpt-4o'
    response = get_openai_llm_response(prompt, model=model_name, json_output=True)
    better_method = int(json.loads(response)['better_method'])
    return better_method


final_prediction = []
selected_method = []
for i in tqdm(range(len(question))):
    tmp_question = question[i]
    tmp_cot_prediction = cot_prediction[i]
    tmp_cot_reasoning = cot_reasoning[i]
    tmp_pot_prediction = pot_prediction[i]
    tmp_pot_reasoning = pot_reasoning[i]
    tmp_table = table[i]

    if os.path.exists(f'{output_path}/{i}.json'):
        with open(f'{output_path}/{i}.json', 'r') as f:
            better_method = json.load(f)
    else:
        better_method = llm_self_eval(tmp_question, tmp_table, tmp_cot_prediction, tmp_cot_reasoning, tmp_pot_prediction, tmp_pot_reasoning)
        with open(f'{output_path}/{i}.json', 'w') as f:
            json.dump(better_method, f)

    if better_method == 1:
        final_prediction.append(tmp_cot_prediction)
        selected_method.append('cot')
    elif better_method == 2:
        final_prediction.append(tmp_pot_prediction)
        selected_method.append('pot')
    else:
        print(f'Error: {better_method}')



for pred, gold in zip(final_prediction, groundtruth):
    total += 1
    res = eval_qa(pred, gold)
    if res:
        correct += 1


print('********************')
print(f'SELF EVAL')
print('Total:', total)
print('Correct:', correct)
print('Accuracy:', correct / total)

print(selected_method.count('cot'))
print(selected_method.count('pot'))
