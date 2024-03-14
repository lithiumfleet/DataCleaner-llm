import os



class TXTDoc:
    def __init__(self) -> None:
        self.fp = None

    def open_file(self, file_path:str):
        assert os.access(file_path, os.R_OK&os.F_OK),"[DEBUG] "+file_path+" is not avaliable!"
        self.fp = open(file_path, 'r', encoding="gbk")

    def get_chunck(self, chunck_size:int):
        assert self.fp != None and not self.fp.closed, "[DEBUG] "+self.fp.name+" is not avaliable!"
        return self.fp.read(chunck_size)

    def check_eof(self) -> bool:
        if self.fp.tell()==0:
            self.fp.close()
            return True
        else:
            return False

