from azure.identity import ManagedIdentityCredential, get_bearer_token_provider
from openai import AzureOpenAI, OpenAI


# token_provider = get_bearer_token_provider(
    # ManagedIdentityCredential(client_id=""), ""
# )

# client = AzureOpenAI(
#     api_version="2024-08-01-preview",
#     azure_endpoint=""
#     azure_ad_token_provider=token_provider
# )


client = OpenAI(
    api_key='',
)



# GLOBAL_MODEL = 'gpt-4o'
# GLOBAL_MODEL = 'gpt-4o-mini'
GLOBAL_MODEL = 'gpt-3.5-turbo'


GLOBAL_TEMPERATURE = 0

def get_openai_llm_response(prompt, model='', json_output=False, json_schema=None, temperature=0):
# def get_openai_llm_response(prompt, model='gpt-3.5-turbo', json_output=False, json_schema=None, temperature=0):
    if model == '':
        model = GLOBAL_MODEL
    if temperature == 0:
        temperature = GLOBAL_TEMPERATURE

    if model == 'gpt-4o':
        model = 'gpt4o-retran'
        # model = 'gpt-4o'
    elif model == 'gpt-3.5-turbo':
        model = 'gpt-35-turbo' # 'gpt-3.5-turbo-0125'
    elif model == 'gpt-4o-mini':
        model = 'gpt-4o-mini'
    elif model == 'gpt-4-turbo':
        model = 'gpt-4' # gpt-4-turbo-2024-04-09

    if json_output and json_schema:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": json_schema["title"],
                    "description": json_schema.pop("description", ""),
                    "schema": json_schema,
                    "strict": True
                }
            },
            temperature=temperature
        ).choices[0].message.content
    else:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object" if json_output else "text"},
            temperature=temperature
        ).choices[0].message.content

    return response


def get_o1_llm_response(prompt, model='o1-preview'):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
    ).choices[0].message.content

    return response



if __name__ == '__main__':
    # prompt = "What is the capital of France?"
    # prompt = "What is the capital of France? step by step"
    prompt = "667x226+31=?"
    print(get_openai_llm_response(prompt, model='gpt-3.5-turbo'))