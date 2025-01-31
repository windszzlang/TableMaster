import json
from glob import glob
import sys

sys.path.append('./')
from evaluate.evaluator import eval_qa



# model = 'gpt35'
model = 'gpt4m'
# model = 'gpt4o'

# prompt_type = 'direct'
prompt_type = 'cot'
# prompt_type = 'pot'


# norm = True
norm = False


MAX_WORKERS = 10

if norm:
    prediction_path = f'outputs/baselines/wikitq/{model}/{prompt_type}/'
else:
    prediction_path = f'outputs/analysis/norm/{model}/{prompt_type}/'


## start evaluation


prediction_files = glob(f'{prediction_path}/*.json')
prediction_files = sorted(prediction_files, key=lambda x: int(x.split('/')[-1].split('.')[0]))


correct = 0
total = 0

for i,prediction_file in enumerate(prediction_files):
    print(prediction_file)
    with open(prediction_file, 'r') as f:
        pred = json.load(f)
    if i >= 1000:
        break
    total += 1
    res = eval_qa(pred['predicted_answer'], pred['answer'])
    if res:
        correct += 1

print('********************')
print(f'Prediction Path: {prediction_path}')
print('Total:', total)
print('Correct:', correct)
print('Accuracy:', correct / total)
