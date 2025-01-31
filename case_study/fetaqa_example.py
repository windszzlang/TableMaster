import json
import sys

sys.path.append('./')
from table_utils import format_table

data = {"id": "fetaqa-164", "source": {"dataset": "fetaqa", "id": 12425, "file_path": "totto_source/train_json/example-4724.json"}, "table": [["Party", "Party", "Candidate", "Votes", "%"], ["-", "Republican", "Phil Garwood", "5,467", "17"], ["-", "Republican", "Victoria Napolitano", "5,580", "18"], ["-", "Republican", "Pete Palko", "5,321", "17"], ["-", "Democratic", "J. Greg Newcomer", "5,345", "17"], ["-", "Democratic", "Brian Sattinger", "4,899", "15"], ["-", "Democratic", "Mark Hines", "4,869", "15"]], "question": "How did Napolitano perform compared to the other candidates?", "answer": "On election day, Napolitano was the top vote-getter with 5,580 votes, outpacing her Republican running mates as well as her Democrat opponents."}


answer = 'on election day napolitano was top vote getter with 5580.0 votes outpacing her republican running mates as well as her democrat opponents'
prediction = 'victoria napolitano performed best among candidates receiving highest percentage of votes at 18.0% with total of 5580.0 votes'


with open('outputs/main/fetaqa/tablemaster-4o/164.json', 'r') as f:
    prediction_data = json.load(f)

evaluation = {
    'BLEU': 0.0411,
    'ROUGE-1': 0.2791,
    'ROUGE-2': 0.0976,
    'ROUGE-L': 0.279
}


print(data['question'])
print(data['answer'])
print(format_table(data['table']))
print(prediction_data['predicted_answer'])
