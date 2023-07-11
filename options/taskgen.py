import itertools
from typing import Callable
from models.nets.cnn import adjustcnn
from utils.arguments import options
from constants.paths import JOBS
import numpy as np
def get_arch_defaults():
    args = "--sigma 4".split()
    archargs,_ = options(args,key = "arch")
    return archargs.widths,archargs.kernels,archargs.batchnorm,archargs.skipconn
def constant_nparam_model(sigma):
    widths,kernels,batchnorm,skipconn = get_arch_defaults()
    widths,kernels = adjustcnn(widths,kernels,batchnorm,skipconn,0,kernel_factor = 4/sigma,constant_nparam = True)
    return widths,kernels

def python_args(**kwargs):
    def givearg(inds,ninds):
        args = []
        subdict = {}
        funss = {}
        
        for i,(key,vals),r in zip(inds,kwargs.items(),ninds):
            if isinstance(vals,list):
                val = vals[i]
            elif isinstance(vals,tuple):
                val = vals
            elif isinstance(vals,Callable):
                funss[key] = vals
                val = None
            else:
                val = vals
            subdict[key] = val
        for key,fun in funss.items():
            val = fun(**subdict)
            subdict[key] = val
        # sortedkeys = np.sort(list(subdict.keys()))
        sortedkeys = list(subdict.keys())
        for key in sortedkeys:
            val = subdict[key]
            if isinstance(val,tuple):
                stval = ' '.join([str(v) for v in val])
            else:
                stval = str(val)
            args.append(f'--{key} {stval}')
        return ' '.join(args)
    indices = []
    for val in kwargs.values():
        if isinstance(val,list):
            indices.append(len(val))
        else:
            indices.append(1)
    rindices = [range(k) for k in indices] 
    args = []
    for prodind in itertools.product(*rindices):
        args.append(givearg(prodind,indices))
    return args
        