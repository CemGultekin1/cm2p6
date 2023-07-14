from copy import deepcopy
import os
from typing import List
from options.params import ARCH_PARAMS
from models.nets.cnn import adjustcnn
from models.search import is_trained
from utils.arguments import get_default
from jobs.job_body import create_slurm_job
from options.taskgen import python_args
from utils.arguments import options,replace_param,replace_params
from constants.paths import JOBS, JOBS_LOGS
from utils.slurm import flushed_print
from data.coords import DEPTHS
TRAINJOB = 'offline_sweep2'
root = JOBS

NCPU = 8


class Task:
    line:str
    key_group:str
    def __init__(self,line:str,key_group:str = "run"):
        self.line = line
        self.key_group = key_group
    @property
    def modelid(self,)->str:
        return options(self.line.split(),key = self.key_group)
    def replace_args(self,*args):
        spl = self.line.split()
        spl = replace_params(spl,*args)
        self.line = " ".join(spl)

def get_arch_defaults():
    return {nm : get_default(nm) for nm in ARCH_PARAMS.keys()}

def constant_nparam_model(sigma,kernel_factor = None):
    if kernel_factor is None:
        kernel_factor = 4/sigma
    kwargs = get_arch_defaults()
    widths,kernels = adjustcnn(**kwargs,kernel_factor = kernel_factor,constant_nparam = True)
    return widths,kernels
def getarch(args,**kwargs):
    modelargs,_ = options(args,'model')
    widths,kernels = constant_nparam_model(modelargs.sigma,**kwargs)
    if modelargs.temperature:
        widthin = 3 
    else:
        widthin = 2
    widths[-1] = 2*widthin
    if modelargs.latitude:
        widthin+=2
    widths[0] = widthin
    return tuple(widths),tuple(kernels)
def fix_architecture(args,**kwargs):
    widths,kernels = getarch(args,**kwargs)
    args = replace_param(args,'widths',widths)
    args = replace_param(args,'kernels',kernels)
    return args
def fix_minibatch(args):
    datargs,_ = options(args,key = "data")
    # if datargs.domain == 'global':
    optminibatch = int((datargs.sigma/4)**2*4)
    # else:
    #     optminibatch = int(64*(datargs.sigma/4)**2)
    args = replace_param(args,'minibatch',optminibatch)
    return args
def check_training_task(args):
    runargs,_ = options(args,key = "run")
    # if runargs.lsrp==1:# or runargs.lossfun == 'heteroscedastic':
    #     return True
    # if runargs.gz21:
    #     return True
    if runargs.depth > 0:
        return True
    
    if not (runargs.lossfun == 'MSE' and runargs.filtering == 'gaussian'):
        return True
    # else:
    #     return True
    # if runargs.seed > 0 or runargs.lossfun == 'heteroscedastic':
    #     return True
    _,modelid = options(args,key = "model")
    return is_trained(modelid)

def fix_model_type(args):
    if 'MVARE' in args:
        args = replace_param(args,'model','dfcnn')
    return args
def combine_all(kwargs:List[dict],base_kwargs):
    argslist = []
    for kwarg in kwargs:
        base_kwargs_ = deepcopy(base_kwargs)
        base_kwargs_.update(kwarg)
        argslist =  argslist + python_args(**base_kwargs_)
    return argslist


def lr_select(lossfun:str = 'heteroscedastic',**kwargs):
    if lossfun == 'heteroscedastic':
        return 1e-4
    elif lossfun == 'MSE':
        return 1e-2

def generate_training_tasks():
    base_kwargs = dict(
        filtering = ['gcm','gaussian'],
        num_workers = NCPU,
        disp = 50,
        lr = lr_select,#1e-4#1e-2
        batchnorm = tuple([1]*7 + [0]),
        lossfun = ['MSE','heteroscedastic'],
        latitude = False,
    )
    kwargs = [
        dict(
            lsrp = 0,     
            depth = 0,
            sigma = [4,8,12,16],
            temperature = [False,True],
            domain = ['four_regions','global'],
        ),
        dict(
            lsrp = 0,
            depth =[int(d) for d in DEPTHS],
            sigma = [4,8,12,16],
            temperature = True,
            domain = 'global',
        )
    ]
    
    argslist = combine_all(kwargs,base_kwargs)
    
    for i in range(len(argslist)):
        args = fix_architecture(argslist[i].split())
        args = fix_minibatch(args)
        args = fix_model_type(args)
        argslist[i] = ' '.join(args)


    def kernel_size_switched(kernelscale):
        kwargs = dict(
            depth =0,
            sigma = [4,8,12,16],
            temperature = True,
            domain = 'global',
        )
        argslist = python_args(**kwargs,**base_kwargs)
        import numpy as np
        _,idx = np.unique(np.array(argslist),return_index=True)
        argslist = np.array(argslist)
        argslist = argslist[np.sort(idx)].tolist()
        
        for i in range(len(argslist)):
            args = fix_architecture(argslist[i].split(),kernel_factor = kernelscale)
            args = fix_minibatch(args)
            args = fix_model_type(args)
            argslist[i] = ' '.join(args)
        return argslist
    kernel_factors = [float(f)/21. for f in [21,15,11,9,7,5,4,3,2,1]]
    argslist_ = []
    for kf in kernel_factors:
        argslist_.extend(kernel_size_switched(kf))
    argslist.extend(argslist_)

    import numpy as np
    _,idx = np.unique(np.array(argslist),return_index=True)
    argslist = np.array(argslist)
    argslist = argslist[np.sort(idx)].tolist()



    
    njobs = len(argslist)
    istrained = []
    for i in range(njobs):
        flag = check_training_task(argslist[i].split())
        flushed_print(flag,argslist[i])
        istrained.append(flag)
    jobarray = ','.join([str(i+1) for i in range(njobs) if not istrained[i]])
    njobs = len(argslist)
    lines = '\n'.join(argslist)
    argsfile = TRAINJOB + '.txt'
    path = os.path.join(root,argsfile)
    with open(path,'w') as f:
        f.write(lines)
    slurmfile =  os.path.join(JOBS,TRAINJOB + '.s')
    out = os.path.join(JOBS_LOGS,TRAINJOB+ '_%A_%a.out')
    err = os.path.join(JOBS_LOGS,TRAINJOB+ '_%A_%a.err')
    create_slurm_job(slurmfile,\
        time = "36:00:00",array = jobarray,\
        mem = str(NCPU*10) + "GB",job_name = TRAINJOB,\
        output = out,error = err,\
        cpus_per_task = str(NCPU),
        nodes = "1",
        gres="gpu:1",
        ntasks_per_node = "1",\
        add_eval = True)


def main():
    generate_training_tasks()

if __name__=='__main__':
    main()
