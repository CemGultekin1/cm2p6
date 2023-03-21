from models.search import find_best_match
from params import replace_param
from utils.arguments import options
import itertools
from models.load import get_statedict
from utils.paths import ONLINE_MODELS
import os
import torch
from datetime import date


def main():
    args = '--lsrp 0 --depth 0 --sigma 4 --filtering gaussian --temperature False --latitude False --interior False --domain four_regions --num_workers 16 --disp 50 --batchnorm 1 1 1 1 1 1 1 0 --lossfun heteroscedastic --widths 2 128 64 32 32 32 32 32 4 --kernels 5 5 3 3 3 3 3 3 --minibatch 4'
    
    replace_values = {
        'filtering' : ['gaussian','gcm'],
        'domain' : ['four_regions','global'],
    }
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
        _,modelid = options(args_,key = 'model')
        models_dict[name] = (
            modelid,{}
        )
        names.append(name)
        
        
    today = date.today().strftime("%Y%m%d")
    path = os.path.join(ONLINE_MODELS,"cem_"+today+'.pth')

    for name in names:
        modelid,_ = models_dict[name]
        statedict,_ = get_statedict(modelid)
        if statedict is None:
            print(f'\t\t{name} is missing')
            models_dict.pop(name)
            continue            
        models_dict[name] = (modelid,statedict['best_model'])
    
    if not os.path.exists(ONLINE_MODELS):
        os.makedirs(ONLINE_MODELS)
    torch.save(models_dict,path)
    print(path)
        

if __name__== '__main__':
    main()