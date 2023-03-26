import os
import shutil
from utils.paths import MODELS,JOBS,EVALS,TRAINING_LOGS
from utils.arguments import options
from params import replace_param
def main():
    roots = [MODELS,EVALS,TRAINING_LOGS]
    exts = ['.pth','.nc','.json']
    path = os.path.join(JOBS,'trainjob.txt')
    file = open(path,'r')
    lines = file.readlines()
    file.close()
    lines = [l.strip() for l in lines if 'gaussian' in l]
    for l in lines:
        _,modelid = options(l.split(),key = 'model')
        path = os.path.join(MODELS,modelid+'.pth')
        if not os.path.exists(path):
            continue
        newl = replace_param(l.split(),'filtering','greedy_gaussian')
        _,newmodelid = options(newl,key = 'model')
        print(modelid,'->',newmodelid)
        for root,ext in zip(roots,exts):
            f0 = os.path.join(root,modelid + ext)
            f1 = os.path.join(root,newmodelid + ext)
            if os.path.exists(f0):
                os.rename(f0,f1)
        
if __name__ == '__main__':
    main()