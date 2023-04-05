import os
from typing import Dict
from models.nets.cnn import adjustcnn
from models.search import is_trained
from params import get_default, replace_param,replace_params
from jobs.job_body import create_slurm_job
from jobs.taskgen import python_args
from utils.arguments import options
from utils.paths import JOBS, JOBS_LOGS
from data.coords import DEPTHS
from utils.slurm import flushed_print
TRAINJOB = 'trainjob'
root = JOBS

NCPU = 16

def get_arch_defaults():
    nms = ('widths','kernels','batchnorm','seed','model')
    return {nm : get_default(nm) for nm in nms}
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
    if runargs.lsrp==1:# or runargs.lossfun == 'heteroscedastic':
        return True
    if runargs.lossfun == 'MVARE':
        mse_model_args = replace_params(args.copy(),'model','fcnn','lossfun','MSE')
        _,modelid = options(mse_model_args,key = "model")
        if not is_trained(modelid):
            return True
        if '--seed' in args:
            if runargs.seed > 0:
                return True
    _,modelid = options(args,key = "model")
    return is_trained(modelid)

def fix_model_type(args):
    if 'MVARE' in args:
        args = replace_param(args,'model','dfcnn')
    return args
def combine_all(kwargs:Dict[int,dict],base_kwargs):
    argslist = []
    for kwarg in kwargs.values():
        argslist =  argslist + python_args(**kwarg,**base_kwargs)
    return argslist


def generate_training_tasks():
    base_kwargs = dict(
        num_workers = NCPU,
        disp = 50,
        batchnorm = tuple([1]*7 + [0]),
        lossfun = ['heteroscedastic','MSE','MVARE'],
    )
    kwargs = {}
    kwargs[0] = dict(
        lsrp = 0,     
        depth = 0,
        sigma = 4,
        filtering = 'gaussian',
        temperature = False,
        latitude = False,
        interior = [True,False],
        min_precision = [0,0.024],
        spacing = ['asis','long_flat'],
        domain = 'four_regions',
    )
    kwargs[1] = dict(
        lsrp = 0,     
        depth = 0,
        sigma = [4,8,12,16],
        filtering = 'gcm',
        temperature = False,
        latitude = False,
        domain = ['four_regions','global'],
        seed = list(range(3))
    )
    kwargs[2] = dict(
        lsrp = [0,1],     
        depth = 0,
        sigma = [4,8,12,16],
        filtering = 'gcm',
        temperature = True,
        latitude = [False,True],
        domain = ['four_regions','global'],
        seed = list(range(3))
    )
    kwargs[3] = dict(
        lsrp = [0,1],     
        depth =[int(d) for d in DEPTHS],
        sigma = [4,8,12,16],
        filtering = 'gcm',
        temperature = True,
        latitude = [False,True],
        domain = 'global',
        seed = list(range(3))
    )
    
    argslist = combine_all(kwargs,base_kwargs)
    
    
    for i in range(len(argslist)):
        args = fix_architecture(argslist[i].split())
        args = fix_minibatch(args)
        args = fix_model_type(args)
        argslist[i] = ' '.join(args)


    def kernel_size_switched(kernelscale):
        kwargs = dict(
            lsrp = [0,1],
            depth =0,
            sigma = [4,8,12,16],
            temperature = True,
            latitude = True,
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
        time = "12:00:00",array = jobarray,\
        mem = "150GB",job_name = TRAINJOB,\
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
