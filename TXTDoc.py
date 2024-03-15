import os
import logging



class TXTDoc:
    def __init__(self) -> None:
        self.fp = None

    def open_file(self, file_path:str, encoding='utf8'):
        assert os.access(file_path, os.R_OK&os.F_OK),"[DEBUG] "+file_path+" is not avaliable!"
        self.fp = open(file_path, 'r', encoding=encoding)


    def get_chunck(self, chunck_size:int):
        assert self.fp != None and not self.fp.closed, "[DEBUG] "+self.fp.name+" is not avaliable!"
        return self.fp.read(chunck_size)

    def close_file(self) -> bool:
        logging.info("finish clean {}".format(self.fp.name))
        self.fp.close()
