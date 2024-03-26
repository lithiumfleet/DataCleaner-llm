from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput
from datasets import load_dataset
from timethis import timethis
import json

# hypers from config.json
with open("./config.json", "r") as config_fp:
    config = json.load(config_fp)

model_path = config['model_path']
input_filepath= config['input_filepath']
output_dir = config['output_dir']
if output_dir[-1] != '/': output_dir += '/' 
encoding = config['file_encoding']
sdft = config['QA_config']['sdft_prompt']

# segments names of original dataset
inst_name = config['QA_config']['fieldname_instruction']
in_name = config['QA_config']['fieldname_input']
out_name = config['QA_config']['fieldname_output']


# load model
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=1024)
model = LLM(model=model_path, tokenizer=model_path)


# get raw dataset
raw_dataset = load_dataset(input_filepath)

# process one example
def to_sdft_prompt(example:dict) -> dict:
    r"""
        process examples using llm
        args: 
            example: 
                a dict in following structure: {instruction:str, input: str, output:str} 
        return:
            sdft prompt
    """

    assert len(example.keys()) == 3, "error: incorrect example structure."

    insturction = example[inst_name]
    inputs = example[inst_name]
    output = example[out_name]

    sdft_prompt = sdft['fixed_prompt'] + \
                  sdft['instruction'] + insturction + inputs + \
                  sdft['reference'] + output + \
                  sdft['response']
    return {'sdft_prompts': sdft_prompt}


# main
dataset = raw_dataset.map(to_sdft_prompt)

with timethis("total time:"):
    resp:list[RequestOutput] = model.generate(dataset['train']['sdft_prompts'], sampling_params)

resp_dict = [{
    inst_name: dataset['train'][inst_name][i],
    in_name  : dataset['train'][in_name][i],
    out_name : resp[i].outputs[0].text
} for i in range(len(resp)) ]

with open(output_dir + "cleaned_ds.json", 'w') as output_fp:
    json.dump(resp_dict, output_fp, ensure_ascii=False, indent=4)