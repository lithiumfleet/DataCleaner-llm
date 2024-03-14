from openai import OpenAI
import json

with open('./config.json', 'r') as config_fp:
    config = json.load(config_fp)

base_url = "http://127.0.0.1:6800/v1"
api_key = "sk-"
client = OpenAI(base_url=base_url, api_key=api_key)


sys_prompt = config['sys_prompt']


completion = client.chat.completions.create(
  model="Baichuan2-13B-Chat",
  messages=[
    {"role": "system", "content": sys_prompt },
    {"role": "user", "content": "你是谁?"}
  ]
)

print(completion.choices[0].message)