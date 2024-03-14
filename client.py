from openai import OpenAI
import json


# get config
with open('./config.json', 'r') as config_fp:
  config = json.load(config_fp)

sys_prompt = config['sys_prompt']
few_shot_examples = config['few_shot_examples']
base_url = config['base_url']
api_key = config['api_key']


# create client
client = OpenAI(base_url=base_url, api_key=api_key)


# make message with history
message = [
  {"role": "system", "content": sys_prompt }
]

for index, example in enumerate(few_shot_examples):
  message.append({"role": "user", "content": example['input']})
  message.append({"role": "assistant", "content": example['output']})


# user inputs
message.append({"role": "user", "content": "what exactly do you need to call me?"})

completion = client.chat.completions.create(
  model = "Baichuan2-13B-Chat",
  messages = message
)

print(completion.choices[0].message.content)