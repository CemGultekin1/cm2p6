
import os
from constants.paths import JOBS
from math import ceil
from utils.arguments import is_consistent, options

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
        self.lines = []
    def read_model_list(self,):
        if not os.path.exists(self.path):
            return
        file1 = open(self.path, 'r')
        lines = file1.readlines()
        file1.close()
        self.lines = [line.strip() for line in lines]
    def __len__(self,):
        return len(self.lines)
    def iterate_lines(self,):
        self.read_model_list()
        for line in self.lines:
            yield line
class PartitionedArgsReader(ArgsReader):
    def __init__(self, filename: str,part_id:int,num_parts:int,):
        super().__init__(filename,)
        self.part_id = part_id
        self.num_parts = num_parts
    def read_model_list(self):
        super().read_model_list()
        n = len(self)
        d = ceil(n/self.num_parts)
        st = d*(self.part_id -1)
        tr = d*self.part_id
        tr = min(tr,n)
        slc = slice(st,tr)
        self.slice = (st,tr)
        self.lines = self.lines[slc]
        
    def iterate_lines(self):
        for i,line in enumerate(super().iterate_lines()):
            yield i + self.slice[0],line
            
class ArgsFinder(ArgsReader):
    def find_fits(self,argstr:str,key:str ='model'):
        args = argstr.split()
        runargs,_ = options(args,key = key)
        lines = []
        di = runargs.__dict__
        dikeys = list(di.keys())
        for dikey in dikeys:
            if dikey not in argstr:
                di.pop(dikey)
            
        for line in self.iterate_lines():
            args_ = line.split()
            if is_consistent(args_,key = key,**di):
                lines.append(line)
        return lines