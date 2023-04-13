from models.search import find_best_match
from utils.arguments import replace_param
from utils.arguments import options
import itertools
from models.load import get_statedict
from constants.paths import ONLINE_MODELS
from utils.slurm import read_args
import os
import torch
from datetime import date
import numpy as np
def model_transfer(state_dict):
    return state_dict
    lyrnums = []
    for key in state_dict:
        num = int(key.split('.')[1])
        lyrnums.append(num)
    lyrnums = np.unique(lyrnums).tolist()
    new_state_dict = {}
    for key in state_dict:
        keysp = key.split('.')
        num = int(keysp[1])
        i = lyrnums.index(num)
        keysp[1] = str(i)
        newkey = '.'.join(keysp)
        new_state_dict[newkey] = state_dict[key]
    return new_state_dict
def main():
    # args = '--batchnorm 1 1 1 1 1 1 1 0 --lossfun heteroscedastic --filtering gaussian --interior False --min_precision 0.024 --domain four_regions --widths 2 128 64 32 32 32 32 32 4 --kernels 5 5 3 3 3 3 3 3'
    
    # replace_values = {
    #     'interior' : (['False',],['']),
    # }
    argnums = [1]
    put_dict_keys = 'widths kernels seed batchnorm min_precision'.split()

    models_dict = {}
    names = []
    for argnum in argnums:
        args_ = read_args(argnum)
        print(args_)
        name = 'gaussian_four_regions'
        modelargs,modelid = options(args_,key = 'model')
        modelargsdict = {
            key:modelargs.__dict__[key] for key in put_dict_keys
        }
        modelargsdict['modelid'] = args_
        models_dict[name] = (
            modelargsdict,{}
        )
        names.append(name)
        
    today = date.today().strftime("%Y%m%d")
    path = os.path.join(ONLINE_MODELS,"cem_"+today+'.pth')

    for name in names:
        modelargsdict,_ = models_dict[name]
        args_ = modelargsdict['modelid']
        statedict,_,_,modelid = get_statedict(args_)
        modelargsdict['modelid'] = modelid
        if statedict is None:
            print(f'\t\t{name} is missing')
            models_dict.pop(name)
            continue            
        models_dict[name] = (modelargsdict,model_transfer(statedict['best_model']))
        print(f'models_dict[{name}] = ({modelargsdict["modelid"]},...')
    if not os.path.exists(ONLINE_MODELS):
        os.makedirs(ONLINE_MODELS)
    torch.save(models_dict,path)
    print(path)
        

if __name__== '__main__':
    main()