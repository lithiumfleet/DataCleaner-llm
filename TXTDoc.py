import os
import logging
import json
from collections import namedtuple

# definition of Sample
Sample = namedtuple("Sample", "instruction input refernce")

# read config file to get fieldnames to solve "where can i get the original instruction/input/output"
with open('./config.json', 'r') as config_fp:
  config = json.load(config_fp)

QA_config = "QA_config"
sdft_prompt = "sdft_prompt"
fieldname_instruction = config[QA_config]["fieldname_instruction"] # FIXME: check whether instruction exists
fieldname_input = config[QA_config]["fieldname_input"]
fieldname_output = config[QA_config]["fieldname_output"]


# txt handler
class TXTDoc:
    def __init__(self) -> None:
        self.fp = None

    def open_file(self, file_path:str, encoding='utf8'):
        assert os.access(file_path, os.R_OK&os.F_OK),"[DEBUG] "+file_path+" is not avaliable!"
        self.fp = open(file_path, 'r', encoding=encoding)


    def get_chunck(self, chunck_size:int) -> str:
        assert self.fp != None and not self.fp.closed, "[DEBUG] "+self.fp.name+" is not avaliable!"
        return self.fp.read(chunck_size)

    def get_Samples(self):
        assert self.fp != None and not self.fp.closed, "[DEBUG] "+self.fp.name+" is not avaliable!"
        datalist = json.load(self.fp)
        # dataset is a list contains many samples.
        for data in datalist:
            original_instruction,original_input,original_output = data[fieldname_instruction],data[fieldname_input],data[fieldname_output]
            yield Sample(original_instruction, original_input, original_output)



    def close_file(self) -> bool:
        logging.info("finish clean {}".format(self.fp.name))
        self.fp.close()
