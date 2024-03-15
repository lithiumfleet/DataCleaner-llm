from openai import OpenAI
import json
from TXTDoc import TXTDoc
import os
from tqdm import tqdm


# get config
with open('./config.json', 'r') as config_fp:
  config = json.load(config_fp)

sys_prompt = config['sys_prompt']
few_shot_examples = config['few_shot_examples']
base_url = config['base_url']
api_key = config['api_key']
test_file_path = config['test_file_path']
output_dir = config['output_dir']
chunck_size = config['chunck_size']
file_encoding = config['file_encoding']


# create client
client = OpenAI(base_url=base_url, api_key=api_key)

# make message with history
message = [
  {"role": "system", "content": sys_prompt }
]

for index, example in enumerate(few_shot_examples):
  message.append({"role": "user", "content": example['input']})
  message.append({"role": "assistant", "content": example['output']})


# start clean
txtDoc = TXTDoc()
txtDoc.open_file(test_file_path, file_encoding)

with open(output_dir+os.sep+txtDoc.fp.name, 'w') as output_fp:
  with tqdm(total=len(txtDoc.fp.read())) as qbar:
    txtDoc.fp.seek(0)
    while True:

      chunck = txtDoc.get_chunck(chunck_size=chunck_size)
      if not chunck:
        txtDoc.close_file()
        break

      message.append({"role": "user", "content": chunck})

      completion = client.chat.completions.create(
        model = "Baichuan2-13B-Chat",
        messages = message,
        temperature = 0.8
      )

      output_fp.write(completion.choices[0].message.content.strip())
      output_fp.flush()

      message.pop()
      qbar.update(chunck_size)
      
