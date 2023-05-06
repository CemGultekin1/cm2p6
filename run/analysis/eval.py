import itertools
import os
import sys
from data.exceptions import RequestDoesntExist
from plots.metrics import metrics_dataset, moments_dataset
from run.train import Timer
import torch
from data.load import get_data
from data.vars import get_var_mask_name
from models.load import load_model, load_old_model
import matplotlib.pyplot as plt
from utils.arguments import options, populate_data_options
from utils.parallel import get_device
from constants.paths import EVALS
from utils.slurm import flushed_print
import numpy as np
from utils.xarray import fromtensor, fromtorchdict, fromtorchdict2tensor, plot_ds
import xarray as xr


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
    args = sys.argv[1:]
    
    
    # from utils.slurm import read_args
    # from utils.arguments import replace_params
    # args = read_args(20,filename = 'temp_trainjob.txt')
    # args =replace_params(args,'num_workers','3','disp','1','mode','eval')#'reset','True')
    
    runargs,_ = options(args,key = "run")
    
    modelid,_,net,_,_,_,_,runargs=load_model(args)
    device = get_device()
    net.to(device)
    lsrp_flag, lsrpid = get_lsrp_modelid(args)
    
    kwargs = dict(contained = '' if not lsrp_flag else 'res')
    assert runargs.mode == "eval"
    multidatargs = populate_data_options(args,non_static_params=[],domain = 'global',interior = False)
    # multidatargs = [args]
    allstats = {}
    for datargs in multidatargs:
        try:
            test_generator, = get_data(datargs,half_spread = net.spread, torch_flag = False, data_loaders = True,groups = ('test',))
        except RequestDoesntExist:
            print('data not found!')
            test_generator = None
        if test_generator is None:
            continue
        stats = {}
        nt = 0
        # timer = Timer()
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
                mean,_ =  net.forward(fields_tensor.to(device))
                mean = mean.to("cpu")


            
            # outfields = fromtorchdict2tensor(forcings).type(torch.float32)
            # mask = fromtorchdict2tensor(forcing_mask).type(torch.float32)
            # yhat = mean.numpy()[0]
            # y = outfields.numpy()[0]
            # m = mask.numpy()[0] < 0.5
            # y[m] = np.nan
            # yhat[m[:3]] = np.nan
            # prst = lambda y: print(np.mean(y[y==y]),np.std(y[y==y]))
            # prst(y),prst(yhat),prst(fields_tensor.numpy())
            # nchan = yhat.shape[0]
            # import matplotlib.pyplot as plt
            # fig,axs = plt.subplots(nchan,2,figsize = (2*5,nchan*6))
            # for chani in range(nchan):
            #     ax = axs[chani,0]
            #     ax.imshow(y[chani,::-1])
            #     ax = axs[chani,1]
            #     ax.imshow(yhat[chani,::-1])
            # fig.savefig('eval_intervention.png')
            # return


            predicted_forcings = fromtensor(mean,forcings,forcing_coords, forcing_mask,denormalize = True,**kwargs)
            true_forcings = fromtorchdict(forcings,forcing_coords,forcing_mask,denormalize = True,**kwargs)

            if lsrp_flag:
                predicted_forcings,true_forcings = lsrp_pred(predicted_forcings,true_forcings)
                predicted_forcings,lsrp_forcings = predicted_forcings
                stats = update_stats(stats,lsrp_forcings,true_forcings,lsrpid)
            stats = update_stats(stats,predicted_forcings,true_forcings,modelid)
            
            # err = np.log10(np.abs(true_forcings - predicted_forcings))
            # plot_ds(predicted_forcings,'predicted_forcings_2',ncols = 1)
            # plot_ds(true_forcings,'true_forcings_2',ncols = 1)           
            # plot_ds(err,'err_2',ncols = 1,cmap = 'magma')
            
            

            # return
            nt += 1
            if runargs.disp > 0 and nt%runargs.disp==0:
                flushed_print(nt)

            # break
            # if nt == 16:
            # for key,stats_ in stats.items():
            #     stats__ = metrics_dataset(stats_/nt,reduckwargs = {})
            #     names = list(stats__.data_vars.keys())
            #     ncols = 2
            #     nrows  = int(np.ceil(len(names)/ncols))
                
            #     fig,axs = plt.subplots(nrows,ncols,figsize = (ncols*5,nrows*5))
            #     for z,(i,j)  in enumerate(itertools.product(range(nrows),range(ncols))):
            #         ax = axs[i,j]
            #         kwargs = dict(vmin =0.5,vmax = 1) if 'r2' in names[z] else dict()
            #         kwargs = dict(vmin =-1,vmax = 1) if 'corr' in names[z] else kwargs
            #         stats__[names[z]].isel(co2 = 0,depth = 0).plot(ax = ax,**kwargs)
            #         ax.set_title(names[z])
                    
            #     fig.savefig(f'_{nt}_{key}.png')
            #     plt.close()
            # if nt >900:
            #     break
            # break

        for key in stats:
            stats[key] = stats[key]/nt
            if key not in allstats:
                allstats[key] = []
            allstats[key].append(stats[key].copy())

    for key in allstats:
        filename = os.path.join(EVALS,key+'.nc')
        print(filename)
        xr.merge(allstats[key]).to_netcdf(filename,mode = 'w')


            

            






if __name__=='__main__':
    main()
