from copy import deepcopy
import itertools
from typing import Any, Dict, List
from options.params import ARCH_PARAMS
from models.nets.cnn import adjustcnn
from models.search import is_trained
from utils.arguments import get_default
from utils.arguments import options,replace_param,replace_params




class Task:
    line:str
    key_group:str
    def __init__(self,line:str,key_group:str = "run"):
        self.line = line
        self.key_group = key_group
    @property
    def modelid(self,)->str:
        return options(self.line.split(),key = self.key_group)[1]
    def replace_args(self,*args):
        spl = self.line.split()
        spl = replace_params(spl,*args)
        self.line = " ".join(spl)

class Architectural(Task):
    def __init__(self, line: str, key_group: str = "run",kernel_factor:float = 1):
        super().__init__(line, key_group)
        self.kernel_factor = kernel_factor
        self.adjusted = False
    @property
    def modelid(self) -> str:
        if not self.adjusted:
            raise Exception
        return  super().modelid
        
    def replace_args(self, *args):
        super().replace_args(*args)
        self.adjusted = False
    @staticmethod
    def get_arch_defaults():
        return {nm : get_default(nm) for nm in ARCH_PARAMS.keys()}
    @staticmethod
    def constant_nparam_model(sigma,kernel_factor = None):
        if kernel_factor is None:
            kernel_factor = 4/sigma
        kwargs = Architectural.get_arch_defaults()
        widths,kernels = adjustcnn(**kwargs,kernel_factor = kernel_factor,constant_nparam = True)
        return widths,kernels
    def getarch(self,**kwargs):
        modelargs,_ = options(self.line.split(),'model')
        widths,kernels = self.constant_nparam_model(modelargs.sigma,kernel_factor=self.kernel_factor,**kwargs)
        if modelargs.temperature:
            widthin = 3 
        else:
            widthin = 2
        widths[-1] = 2*widthin
        if modelargs.latitude:
            widthin+=2
        widths[0] = widthin
        return tuple(widths),tuple(kernels)
    def fix_architecture(self,**kwargs):
        args = self.line.split()
        widths,kernels = self.getarch(**kwargs)
        args = replace_params(args,'widths',widths,'kernels',kernels)
        self.line = " ".join(args)
    def fix_minibatch(self,):
        args = self.line.split()
        datargs,_ = options(args,key = "data")
        optminibatch = int((datargs.sigma/4)**2*4)
        args = replace_param(args,'minibatch',optminibatch)
        self.line = " ".join(args)
        
    def check_training_task(self,):
        args = self.line.split()
        runargs,_ = options(args,key = "run")
        if runargs.lsrp==1:# or runargs.lossfun == 'heteroscedastic':
            return True
        if runargs.gz21:
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
    def fix_model_type(self,):
        args = self.line.split()
        if 'MVARE' in args:
            args = replace_param(args,'model','dfcnn')
        self.line = " ".join(args)
    def adjust(self,):
        if self.adjusted:
            return
        self.fix_architecture()
        self.fix_minibatch()
        self.fix_model_type()
        self.adjusted = True
        
class JobMultiplier:
    argslist:List[Architectural]
    def __init__(self,**base_kwargs:Dict[str,Any]):
        self.base_kwargs = base_kwargs
        self.argslist = []
    def for_each(self,fun):
        for arch in self.argslist:
            fun(arch)
    def adjust_architecture(self,):
        for arch in self.argslist:
            arch.adjust()
    def unique(self,):
        modelids = {}
        for arch in self.argslist:
            md = arch.modelid
            if md in modelids:
                continue
            modelids[md] = arch
        jm = JobMultiplier(**self.base_kwargs)
        jm.argslist = list(modelids.values())
        return jm
    @property
    def copy_kwargs(self,):
        return deepcopy(self.base_kwargs)
    def add(self,**kwargs):
        base_kwargs = self.copy_kwargs
        base_kwargs.update(kwargs)
        margs = self.multiply_(**base_kwargs)
        for arg in margs:
            arch = Architectural(arg,kernel_factor=kwargs.get("kernel_factor",1.))
            self.argslist.append(arch)
    def multiply_(self,**kwargs):
        def givearg(inds):
            args = []
            for i,(key,vals) in zip(inds,kwargs.items()):
                if isinstance(vals,list):
                    val = vals[i]
                elif isinstance(vals,tuple):
                    val = ' '.join([str(v) for v in vals])
                else:
                    val = vals
                args.append(f'--{key}')
                args.append(str(val))
            return ' '.join(args)
        indices = []
        for val in kwargs.values():
            if isinstance(val,list):
                indices.append(len(val))
            else:
                indices.append(1)
        indices = [range(k) for k in indices] 
        args = []
        for prodind in itertools.product(*indices):
            args.append(givearg(prodind))
        return args
        
