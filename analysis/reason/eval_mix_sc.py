import sys
import json
from glob import glob


sys.path.append('.')
from evaluate.evaluator import eval_qa



dataset = 'wikitq'


## baselines

# model = 'gpt35'
model = 'gpt4m'
# model = 'gpt4o'


REPEAT_NUMBER = 3
# REPEAT_NUMBER = 5


prediction = []
groundtruth = []

for i in range(1, REPEAT_NUMBER+1):
    for prompt_type in ['cot', 'pot']:
    # for prompt_type in ['cot']:
    # for prompt_type in ['pot']:
        prediction_path = f'outputs/baselines/{dataset}/{model}/mix_sc/{i}/{prompt_type}/'

        prediction_files = glob(f'{prediction_path}/*.json')
        prediction_files = sorted(prediction_files, key=lambda x: int(x.split('/')[-1].split('.')[0]))

        tmp_prediction = []
        tmp_groundtruth = []
        for j, prediction_file in enumerate(prediction_files):
            # print(prediction_file)
            with open(prediction_file, 'r') as f:
                pred = json.load(f)
            tmp_prediction.append(pred['predicted_answer'])
            tmp_groundtruth.append(pred['answer'])

        prediction.append(tmp_prediction)
        groundtruth = tmp_groundtruth


correct = 0
total = 0

# print(prediction)


# voting
final_prediction = []
for i in range(len(prediction[0])):
    vote = {}
    # for j in range(REPEAT_NUMBER):
    for j in range(REPEAT_NUMBER * 2):
        if prediction[j][i] in vote:
            vote[prediction[j][i]] += 1
        else:
            vote[prediction[j][i]] = 1
    final_prediction.append(max(vote, key=vote.get))



for pred, gold in zip(final_prediction, groundtruth):
    total += 1
    res = eval_qa(pred, gold)
    if res:
        correct += 1


print('********************')
print(f'MIX SC {REPEAT_NUMBER}')
print('Total:', total)
print('Correct:', correct)
print('Accuracy:', correct / total)

