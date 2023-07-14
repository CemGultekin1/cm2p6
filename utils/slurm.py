
import os
from constants.paths import JOBS


def flushed_print(*args,**kwargs):
    print(*args,**kwargs,flush = True)
def read_args(line_num,filename :str = 'trainjob.txt'):
    models = os.path.join(JOBS,filename)
    file1 = open(models, 'r')
    lines = file1.readlines()
    file1.close()
    return lines[line_num - 1].strip().split()

class ArgsReader:
    def __init__(self,filename:str):
        self.path = os.path.join(JOBS,filename)
        file1 = open(self.path, 'r')
        lines = file1.readlines()
        file1.close()
        self.lines = [line.strip() for line in lines]
    def __len__(self,):
        return len(self.lines)
    def iterate_lines(self,):
        for line in self.lines:
            yield line