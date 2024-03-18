from openai import OpenAI
import json
from TXTDoc import TXTDoc
import os
from tqdm import tqdm


# get config
with open('./config.json', 'r') as config_fp:
  config = json.load(config_fp)

QA_config = "QA_config"
sdft_prompt = "sdft_prompt"
## following four vars are all parts of prompt.
fixed_prompt = config[QA_config][sdft_prompt]['fixed_prompt']
instruction = config[QA_config][sdft_prompt]['instruction']
reference = config[QA_config][sdft_prompt]['reference']
response = config[QA_config][sdft_prompt]['response']

# read config file to get fieldnames to solve "where can i get the original instruction/input/output"
fieldname_instruction = config[QA_config]["fieldname_instruction"] # FIXME: check whether instruction exists
fieldname_input = config[QA_config]["fieldname_input"]
fieldname_output = config[QA_config]["fieldname_output"]

base_url = config['base_url']
api_key = config['api_key']
input_filepath = config['input_filepath']
output_dir = config['output_dir']
file_encoding = config['file_encoding']


# create client
client = OpenAI(base_url=base_url, api_key=api_key)

# start clean
txtDoc = TXTDoc()
txtDoc.open_file(input_filepath, file_encoding)

# new dataset
new_dataset = []

# process
for index, sample in enumerate(txtDoc.get_Samples()):

  # make message with instruction
  message = [
    { "role": "user", "content": 
      fixed_prompt +
      instruction.format(sample.instruction+sample.input) +
      reference.format(sample.refernce) +
      response
    }
  ]

  # get response
  completion = client.chat.completions.create(
    model = "Baichuan2-13B-Chat",
    messages = message,
    temperature = 0.8
  )

  new_sample = {
    fieldname_instruction: sample.instruction,
    fieldname_input: sample.input,
    fieldname_output: completion.choices[0].message.content.strip()
  }
  new_dataset.append(new_sample)

  # process circle
  print(["[  ]","[>  ]","[=> ]","[==>]","[ ==]","[  =]"][index%6] + " Current index: {}".format(index), end="\r")

# save to file: 
output_file_path = output_dir+os.sep+txtDoc.fp.name
print("Saving to {}".format(output_file_path))
with open(output_file_path, 'w') as output_fp:
  json.dump(new_dataset, output_fp, ensure_ascii=False, indent=4)
print("Finish!")