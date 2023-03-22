import os
from utils.paths import JOBS,statedict_path
from utils.arguments import options
import os 
def main():
    models = os.path.join(JOBS,'trainjob.txt')
    file1 = open(models, 'r')
    lines = file1.readlines()
    for line in lines:
        _,modelid = options(line.split(),key = 'model')
        spath = statedict_path(modelid)
        if os.path.exists(spath):
            spath = spath.replace('.pth','.txt')
            if os.path.exists(spath):
                continue
            print(spath)
            target = open(spath,'w')        
            target.write(line)
            target.close()
            print(spath)
        # return

if __name__ == '__main__':
    main()