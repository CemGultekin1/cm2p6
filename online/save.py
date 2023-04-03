from models.search import find_best_match
from params import replace_param
from utils.arguments import options
import itertools
from models.load import get_statedict
from utils.paths import ONLINE_MODELS
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
    args = '--lsrp 0 --depth 0 --sigma 4 --min_precision 0.024 --filtering gaussian --temperature False --latitude False --interior False --domain four_regions --num_workers 16 --disp 50 --batchnorm 1 1 1 1 1 1 1 0 --lossfun heteroscedastic --widths 2 128 64 32 32 32 32 32 4 --kernels 5 5 3 3 3 3 3 3 --minibatch 4'
    
    replace_values = {
        'filtering' : ['gaussian','gcm'],
        'domain' : ['four_regions','global'],
    }
    put_dict_keys = 'widths kernels seed batchnorm min_precision'.split()
    replace_values = [
       [ (key,val) for val in vals ]for key,vals in replace_values.items()]

    models_dict = {}
    names = []
    for keyvalpairs in itertools.product(*replace_values):
        args_ = args.split()
        name = []
        for key,val in keyvalpairs:
            args_ = replace_param(args_,key,val)
            name.append(val)
        name = '_'.join(name)
        modelargs,modelid = options(args_,key = 'model')
        modelargsdict = {
            key:modelargs.__dict__[key] for key in put_dict_keys
        }
        modelargsdict['modelid'] = modelid
        models_dict[name] = (
            modelargsdict,{}
        )
        names.append(name)
        
        
    today = date.today().strftime("%Y%m%d")
    path = os.path.join(ONLINE_MODELS,"cem_"+today+'.pth')

    for name in names:
        modelargsdict,_ = models_dict[name]
        modelid = modelargsdict['modelid']
        statedict,_ = get_statedict(modelid)
        if statedict is None:
            print(f'\t\t{name} is missing')
            models_dict.pop(name)
            continue            
        models_dict[name] = (modelargsdict,model_transfer(statedict['best_model']))
    
    if not os.path.exists(ONLINE_MODELS):
        os.makedirs(ONLINE_MODELS)
    torch.save(models_dict,path)
    print(path)
        

if __name__== '__main__':
    main()