import argparse
import hashlib
import itertools
from typing import List
from constants.params import DATA_PARAMS,MODEL_PARAMS,ARCH_PARAMS,RUN_PARAMS, SCALAR_PARAMS, TRAIN_PARAMS,USUAL_PARAMS,PARAMS
import numpy as np


def tuple2string(tpl):
    if tpl is None:
        return ""
    if isinstance(tpl,tuple):
        return " ".join([str(x) for x  in tpl])
    else:
        return str(tpl)

def defaulting_dict(d:dict,key:str,**kwargs):
    return d.get(key,get_default(key,**kwargs))

def get_default(key,instr = False):
    if key not in PARAMS:
        return_val = None
    else:
        return_val = PARAMS[key]["default"]
    if instr:
        return tuple2string(return_val)
    else:
        return return_val
    


def replace_params(args,*ARGS):
    for i in range(len(ARGS)//2):
        args = replace_param(args,ARGS[2*i],ARGS[2*i+1])
    return args


def replace_param(args,param,newval):
    if not isinstance(newval,str):
        newval = tuple2string(newval)
    param_ = f'--{param}'
    if param_ in args:
        args[args.index(param_)+1] = newval
    else:
        args.append(param_)
        args.append(newval)
    return args



def is_legacy_run(args):
    runargs,_ = options(args,key = 'run')
    return runargs.gz21
def populate_data_options(args,non_static_params = ["depth","co2"],**kwargs):
    prms,_ = options(args,key = "run")
    d = prms.__dict__
    def isarray(val):
        return isinstance(val,list) or isinstance(val,tuple)
    for key,val in kwargs.items():
        if not isarray(val):
            d[key] = val
        else:
            d[key] = ' '.join([str(v) for v in val])
    for key,val in d.items():
        if not isarray(val):
            d[key] = val
        else:
            d[key] = ' '.join([str(v) for v in val])
    prods = []
    paramnames = []
    for param in DATA_PARAMS:
        if param in kwargs:
            continue
        if param not in non_static_params:
            continue
        opts = DATA_PARAMS[param]
        paramnames.append(param)
        if isinstance(opts['default'],bool):
            prods.append((False,True))
            continue
        if param in USUAL_PARAMS:
            prods.append(USUAL_PARAMS[param])
            continue
        if "choices" in opts:
            prods.append(opts["choices"])
            continue
        raise NotImplemented
    arglines = []
    for pargs in itertools.product(*prods):
        arglines.append(" ".join([f"--{name} {arg_}" for name,arg_ in zip(paramnames,pargs)]))
    static_part = ""
    for param in d:
        if param  in paramnames:
            continue
        static_part += f" --{param} {d[param]}"
    arglines = [argl + static_part for argl in arglines]
    arglines = np.unique(arglines).tolist()
    arglines = [argl.split() for argl in arglines]
    return arglines



def options(string_input,key:str = "model"):
    if key == "model":
        prms = MODEL_PARAMS
    elif key == "arch":
        prms = ARCH_PARAMS
    elif key == "data":
        prms = DATA_PARAMS
    elif key == "run":
        prms = RUN_PARAMS
    elif key == "scalars":
        prms = SCALAR_PARAMS
    elif key == "train":
        prms = TRAIN_PARAMS
    else:
        raise Exception('not implemented')


    model_parser=argparse.ArgumentParser()
    st_inputs = []
    for argname,argdesc in prms.items():
        model_parser.add_argument(f"--{argname}",**argdesc)
        if f"--{argname}" in string_input:
            i = string_input.index(f"--{argname}")
            j = i+1
            st_inputs.append(f"--{argname}")
            while '--' not in string_input[j]:
                # print('\t\t',string_input[j])
                st_inputs.append(string_input[j])
                j+=1
                if j == len(string_input):
                    break
            
    args,_ = model_parser.parse_known_args(st_inputs)
    return args,args2num(prms,args)

def args2num(prms:dict,args:argparse.Namespace):
    s = []
    nkeys = len(prms)
    def append_el(l:List,i,el):
        if isinstance(el,list):
            ell = hash(tuple(el))
        elif isinstance(el,float):
            ell = int(el*1e6)
        else:
            ell = el
        l.append(i)
        l.append(ell)
        return l

    for i,(u,v) in zip(range(nkeys),args.__dict__.items()):
        if prms[u]["default"] != v:
            s = append_el(s,u,v)


    s = tuple(s)
    return hashlib.sha224(str(s).encode()).hexdigest()


def args2dict(args,key = 'model',coords = dict(), transform_funs = dict()):
    modelargs,modelid = options(args,key = key)
    coordvals = {}
    for coord in coords:
        if coord not in transform_funs:
            val = modelargs.__getattribute__(coord)
        else:
            inputs1 = [modelargs.__getattribute__(v) for v in transform_funs[coord]['inputs']]
            val = transform_funs[coord]['fun'](*inputs1)
        if isinstance(val,bool):
            val = int(val)
        coordvals[coord] = [val]
    return coordvals,(modelargs,modelid)