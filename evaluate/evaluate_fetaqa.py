import json
from glob import glob

from evaluator import eval_free_qa


dataset = 'fetaqa'

size = -1


## main
# method = 'tablemaster'
# method = 'tablemaster-35'
# method = 'tablemaster-4m'
method = 'tablemaster-4o'

# method = 'end2end'


# size = 1400

prediction_path = f'outputs/main/{dataset}/{method}'


## start evaluation


prediction_files = glob(f'{prediction_path}/*.json')
prediction_files = sorted(prediction_files, key=lambda x: int(x.split('/')[-1].split('.')[0]))


total = 0
predicted_answer_list = []
answer_list = []

for i, prediction_file in enumerate(prediction_files):
    # print(prediction_file)
    with open(prediction_file, 'r') as f:
        pred = json.load(f)
    total += 1
    if size != -1 and i > size:
        break
    predicted_answer_list.append(str(pred['predicted_answer']))
    answer_list.append(str(pred['answer']))



results = eval_free_qa(predicted_answer_list, answer_list)

print('********************')
print(f'Prediction Path: {prediction_path}')
print('Total:', total)
print("Evaluation Results:")
for metric, score in results.items():
    print(f"{metric}: {score:.4f}")
