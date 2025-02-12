import json
from openai import OpenAI



# Create an OpenAI client with your deepinfra token and endpoint
client = OpenAI(
    api_key="",
    base_url="https://api.deepinfra.com/v1/openai",
)

GLOBAL_TEMPERATURE = 0

def get_openai_llm_response(prompt, model='', json_output=False, json_schema=None, temperature=0):
    model = 'meta-llama/Meta-Llama-3.1-70B-Instruct'
    if temperature == 0:
        temperature = GLOBAL_TEMPERATURE


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


if __name__ == '__main__':
    print(get_openai_llm_response('Hello, return a json object with a name and age', json_output=True, json_schema=json.loads('''{
        "title": "Person",
        "description": "A person's name",
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "The person's name"},
            "age": {"type": "integer", "description": "The person's age"}
        },
        "required": ["name", "age"]
    }''')))
