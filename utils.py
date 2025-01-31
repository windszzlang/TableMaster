from azure_openai_api import get_openai_llm_response


def fact_to_qa(statement: str, model: str = 'gpt-4o'):
    question = 'Is it true that ' + statement
    return question



if __name__ == '__main__':
    print(fact_to_qa('portmouth fc win all the game it play in january 1950'))