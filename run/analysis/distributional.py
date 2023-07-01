import os
import sys
from data.exceptions import RequestDoesntExist
from plots.metrics import  moments_dataset
import torch
from data.load import get_data
from models.load import load_model
import matplotlib.pyplot as plt
from utils.arguments import options, populate_data_options
from utils.parallel import get_device
from constants.paths import DISTS
from utils.slurm import flushed_print
import numpy as np
from utils.xarray import fromtensor, fromtorchdict, fromtorchdict2tensor
import xarray as xr
from run.helpers import PrecisionToStandardDeviation,AdaptiveHistogram
def lsrp_pred(respred,tr):
    keys= list(respred.data_vars.keys())
    data_vars = {}
    coords = {key:val for key,val in tr.coords.items()}
    for key in  keys:
        trkey = key.replace('_res','')
        trval = tr[trkey] - tr[key] # true - (true - lsrp) = lsrp
        data_vars[trkey] = (trval.dims,trval.values)
        respred[key] = trval + respred[key]
        respred = respred.rename({key:trkey})
        tr = tr.drop(key)
    lsrp = xr.Dataset(data_vars =data_vars,coords = coords)
    return (respred,lsrp),tr
def update_stats(stats,prd,tr,key):
    stats_ = moments_dataset(prd,tr)
    if key not in stats:
        stats[key] = stats_
    else:
        stats[key] = stats[key] + stats_
    return stats
def get_lsrp_modelid(args):
    runargs,_ = options(args,key = "model")
    lsrp_flag = runargs.lsrp > 0 and runargs.temperature
    if not lsrp_flag:
        return False, None
    keys = ['model','sigma']
    vals = [runargs.__getattribute__(key) for key in keys]
    lsrpid = runargs.lsrp - 1
    vals[0] = f'lsrp:{lsrpid}'
    line =' '.join([f'--{k} {v}' for k,v in zip(keys,vals)])
    _,lsrpid = options(line.split(),key = "model")
    return True, lsrpid

def main():
    # args = sys.argv[1:]
    
    from utils.slurm import read_args
    from utils.arguments import replace_params
    from utils.slurm import read_args
    args = read_args(25,filename = 'dists.txt')
    args = replace_params(args,'mode','eval')
    
    
    runargs,_ = options(args,key = "run")
    prec2std = PrecisionToStandardDeviation(args)
    if not os.path.exists(DISTS):
        os.makedirs(DISTS)
    modelid,_,net,_,_,_,_,runargs=load_model(args)
    net.eval()
    device = get_device()
    net.to(device)
    lsrp_flag, lsrpid = get_lsrp_modelid(args)
    
    kwargs = dict(contained = '' if not lsrp_flag else 'res')
    assert runargs.mode == "eval"
    multidatargs = populate_data_options(args,non_static_params=['depth','co2'],domain = 'global',interior = False,wet_mask_threshold = 0.5)

    
    for datargs in multidatargs:
        try:
            test_generator, = get_data(datargs,half_spread = net.spread, torch_flag = False, data_loaders = True,groups = ('test',))
        except RequestDoesntExist:
            print('data not found!')
            test_generator = None
        if test_generator is None:
            continue
        nt = 0
        # timer = Timer()
        adaptive_histogram = None
        for fields,forcings,forcing_mask,_,forcing_coords in test_generator:
            fields_tensor = fromtorchdict2tensor(fields).type(torch.float32)
            depth = forcing_coords['depth'].item()
            co2 = forcing_coords['co2'].item()
            kwargs = dict(contained = '' if not lsrp_flag else 'res', \
                expand_dims = {'co2':[co2],'depth':[depth]},\
                drop_normalization = True,
                )
            if nt ==  0:
                flushed_print(depth,co2)

            with torch.set_grad_enabled(False):
                mean,prec =  net.forward(fields_tensor.to(device))
                mean = mean.to("cpu")
                prec = prec.to("cpu")
            std = prec2std(prec)
            
            predicted_forcings = fromtensor(mean,forcings,forcing_coords, forcing_mask,denormalize = True,**kwargs)

            predicted_std = fromtensor(std,forcings,forcing_coords, forcing_mask,denormalize = True,**kwargs)
            true_forcings = fromtorchdict(forcings,forcing_coords,forcing_mask,denormalize = True,**kwargs)

            normalized_err = (true_forcings - predicted_forcings)/predicted_std
            
            masked_normalized_err = xr.where(np.isnan(predicted_forcings) ,np.nan,normalized_err)
            masked_normalized_err = masked_normalized_err.sel(lat = slice(-85,85))
            if adaptive_histogram is None:
                adaptive_histogram = AdaptiveHistogram(masked_normalized_err,5000)
            adaptive_histogram.update(masked_normalized_err)            
            if runargs.disp > 0 and nt%runargs.disp==0:
                density = adaptive_histogram.get_density_xarray()
                filename = os.path.join(DISTS,modelid+'.nc')
                density.to_netcdf(filename,mode = 'w')
                density.Su_density.plot()
                plt.savefig('density_su.png')
                plt.close()
                flushed_print(nt)
            nt += 1
    density = adaptive_histogram.get_density_xarray()
    filename = os.path.join(DISTS,modelid+'.nc')    
    density.to_netcdf(filename,mode = 'w')
    print(filename)


            

            






if __name__=='__main__':
    main()
