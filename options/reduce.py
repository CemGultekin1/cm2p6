from typing import List
from models.nets.cnn import kernels2spread
from utils.arguments import options
import numpy as np

def kernels_to_stencil(kernels:List[int]):
    ks = kernels2spread(kernels)*2 + 1
    return ks


scalar_transforms = dict(
    kernels = ('stencil',kernels_to_stencil)
)
class ScalarArguments:
    def transform_arguments(self,*args):
        runargs,_ = options(args,key = 'run')
        rdict = {}
        for key,val in runargs.__dict__.items():
            if np.isscalar(val):
                rdict[key] = val
            elif key in scalar_transforms:
                new_key,tfun = scalar_transforms[key]
                val = tfun(val)
                assert np.isscalar(val)
                rdict[new_key] = val
            else:
                continue
        return rdict