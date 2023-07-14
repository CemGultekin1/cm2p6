
import os
from constants.paths import JOBS
from math import ceil

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
class PartitionedArgsReader(ArgsReader):
    def __init__(self, filename: str,part_id:int,num_parts:int):
        super().__init__(filename)
        n = len(self)
        d = ceil(n/num_parts)
        st = d*(part_id -1)
        tr = d*part_id
        tr = min(tr,n)
        slc = slice(st,tr)
        self.slice = (st,tr)
        self.lines = self.lines[slc]
        self.part_id = part_id
        self.num_parts = num_parts
    def iterate_lines(self):
        for i,line in enumerate(super().iterate_lines()):
            yield i + self.slice[0],line