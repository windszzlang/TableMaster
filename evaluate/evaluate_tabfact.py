import json
from glob import glob

from evaluator import eval_fact



dataset = 'tabfact'



## main
# # method = 'tablemaster'
# method = 'tablemaster-4m'
method = 'tablemaster-4m-new'
# method = 'tablemaster-35'
# method = 'tablemaster-4o'
# method = 'tablemaster-llama13b'
# method = 'tablemaster-llama70b'


prediction_path = f'outputs/main/{dataset}/{method}'

## ablation

# ablation_method = 'wo_structure_extraction' 
# ablation_method = 'wo_column_lookup'
# ablation_method = 'wo_row_lookup'
# ablation_method = 'wo_table_of_focus'
# ablation_method = 'wo_reconstruction'
# ablation_method = 'wo_verbalization'
# ablation_method = 'wo_textual_reasoning'
# ablation_method = 'wo_symbolic_reasoning' 
# ablation_method = 'wo_textual_guidance'

# prediction_path = f'outputs/ablation/{dataset}/{ablation_method}'


## baselines

# model = 'gpt35'
# model = 'gpt4m'
# model = 'gpt4o'

# prompt_type = 'direct'
# prompt_type = 'cot'
# prompt_type = 'pot'
# prompt_type = 'verbal'
# prompt_type = 'text'

# prediction_path = f'outputs/baselines/{dataset}/{model}/{prompt_type}'


# prediction_path = f'outputs/baselines/{dataset}/o1'
# prediction_path = f'outputs/baselines/{dataset}/o1m'


## start evaluation


prediction_files = glob(f'{prediction_path}/*.json')
prediction_files = sorted(prediction_files, key=lambda x: int(x.split('/')[-1].split('.')[0]))


correct = 0
total = 0

for i, prediction_file in enumerate(prediction_files):

    # print(prediction_file)
    with open(prediction_file, 'r') as f:
        pred = json.load(f)
    total += 1
    res = eval_fact(pred['predicted_answer'], pred['answer'])
    if res:
        correct += 1

print('********************')
print(f'Prediction Path: {prediction_path}')
print('Total:', total)
print('Correct:', correct)
print('Accuracy:', correct / total)
