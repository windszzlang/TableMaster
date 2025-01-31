import sys
import json

sys.path.append('./ablation/tablemaster_variant/tablemaster_wo_rl')
sys.path.append('./')

from structure import table_structure_understanding
from content import table_content_understanding
from reasoning import table_reasoning_for_qa



def tablemaster_table_understanding(table_array, question, task='qa'):
    topheaders, key_column_index, selected_column_indices, ranked_column_indices, sql, selected_row_indices = table_structure_understanding(table_array, question, task=task)
    verbalized_table, subtable_array, final_selected_column_indices = table_content_understanding(table_array, question, selected_row_indices, selected_column_indices, ranked_column_indices)
    final_answer, textual_reasoning_process, symbolic_reasoning_process = table_reasoning_for_qa(table_array, verbalized_table, question, task=task)
    reasoning_process = {
        'topheaders': topheaders,
        'key_column_index': key_column_index,
        'selected_column_indices': selected_column_indices,
        'ranked_column_indices': ranked_column_indices,
        'sql': sql,
        'selected_row_indices': selected_row_indices,
        'verbalized_table': verbalized_table,
        'subtable_array': subtable_array,
        'final_selected_column_indices': final_selected_column_indices,
        'textual_reasoning_process': textual_reasoning_process,
        'symbolic_reasoning_process': symbolic_reasoning_process
    }
    return final_answer, reasoning_process




if __name__ == '__main__':
    with open('./test_tables/test.json', 'r') as f:
        data = json.load(f)
    table_array = data['table']
    question = data['question']
    example = 'Question: how many ships were launched in the year 1944?\nAnswer: 9'
    final_answer, reasoning_process = tablemaster_table_understanding(table_array, question)
    print(final_answer)
    print(reasoning_process)