import json



IDs = [i for i in range(20)]


fetaqa_examples = []


table_question_answer_list = []

with open('data/fetaqa/train.jsonl', 'r') as f:
    for line in f:
        D = json.loads(line)
        table_question_answer_list.append((D['table'], D['question'], D['answer']))


for i in IDs:
    fetaqa_examples.append(table_question_answer_list[i])

# fetaqa_examples_str = 'Table: {table}\nQuestion: {question}\nAnswer: {answer}\n'
fetaqa_examples_str = 'Question: {question}\nAnswer: {answer}\n'

for D in fetaqa_examples:
    # print(fetaqa_examples_str.format(table=D[0], question=D[1], answer=D[2]))
    print(fetaqa_examples_str.format(question=D[1], answer=D[2]))


