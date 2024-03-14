from openai import OpenAI
import json
from TXTDoc import TXTDoc


# get config
with open('./config.json', 'r') as config_fp:
  config = json.load(config_fp)

sys_prompt = config['sys_prompt']
few_shot_examples = config['few_shot_examples']
base_url = config['base_url']
api_key = config['api_key']
test_file_path = config['test_file_path']


# create client
client = OpenAI(base_url=base_url, api_key=api_key)


# make message with history
message = [
  {"role": "system", "content": sys_prompt }
]

for index, example in enumerate(few_shot_examples):
  message.append({"role": "user", "content": example['input']})
  message.append({"role": "assistant", "content": example['output']})



txtDoc = TXTDoc()
txtDoc.open_file(test_file_path)
chunck = " "

while chunck != "":

  chunck = txtDoc.get_chunck(1024)

  message.append({"role": "user", "content": chunck})

  completion = client.chat.completions.create(
    model = "Baichuan2-13B-Chat",
    messages = message
  )

  print(completion.choices[0].message.content)

  message.pop()