import os
import sys
from data.exceptions import RequestDoesntExist
from plots.metrics_ import moments_dataset
import torch
from data.load import get_data
from utils.arguments import options, populate_data_options
from constants.paths import EVALS
from utils.slurm import flushed_print
from utils.xarray import fromtensor, fromtorchdict, fromtorchdict2tensor, plot_ds
import xarray as xr
from utils.arguments import replace_params

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
    runargs,_ = options(args)
    args = f'--model lsrp:0 --sigma {runargs.sigma} --filtering {runargs.filtering}'.split()
    _,lsrpid = options(args,key = "model")
    return True, lsrpid


def main():
    args = sys.argv[1:]
    # coarse_graining_factor = int(sys.argv[1])
    args = f'--model lsrp:0 --sigma 12 --lsrp True --num_workers 1 --temperature True --filtering gaussian --mode eval'.split()
    
    # from utils.slurm import read_args
    # args = read_args(289,filename = 'offline_sweep.txt')
    args = replace_params(args,'mode','eval','lsrp',1,'temperature','True',)
    
    lsrp_flag, lsrpid = get_lsrp_modelid(args)
    
    non_static_params=['depth','co2',]
    multidatargs = populate_data_options(args,non_static_params=non_static_params,domain = 'global',interior = False)
    # multidatargs = [args]
    allstats = {}
    for datargs in multidatargs:
        try:
            test_generator, = get_data(datargs,half_spread = 0, torch_flag = False, data_loaders = True,groups = ('test',))
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
            # for key,val in forcing_coords.items():
            #     if np.isscalar(val) or isinstance(val,str):
            #         print(f'{key} : {val}')
            #     else:
            #         if len(val.shape)>=1:
            #             print(f'{key} : {val.shape}')
            #         else:
            #             print(f'{key} : {val}')
            depth = forcing_coords['depth'].item()
            co2 = forcing_coords['co2'].item()
            if abs(depth - 5)>1 or co2 > 0:
                break
            kwargs = dict(contained = '' if not lsrp_flag else 'res', \
                expand_dims = {key:forcing_coords[key].item() for key in non_static_params},\
                drop_normalization = True,
                masking = False
                )
            if nt ==  0:
                flushed_print(depth,co2)
            mean = fields_tensor*0
            mean = mean[:,:3]


            predicted_forcings = fromtensor(mean,forcings,forcing_coords, forcing_mask,denormalize = True,**kwargs)
            true_forcings = fromtorchdict(forcings,forcing_coords,forcing_mask,denormalize = True,**kwargs)
            (predicted_forcings,lsrp_forcings),true_forcings = lsrp_pred(predicted_forcings,true_forcings)
            from utils.xarray import plot_ds
            plot_ds(true_forcings,'true_forcings.png',ncols = 3)
            print(lsrp_forcings)
            # # return
            plot_ds(lsrp_forcings,'lsrp_forcings.png',ncols = 3)
            stats = update_stats(stats,lsrp_forcings,true_forcings,lsrpid)                        
            plot_ds(stats[lsrpid],'eval_interp.png',ncols = 3)

            raise Exception
            nt += 1
            if nt%20 == 0:
                flushed_print(nt)


        for key in stats:
            stats[key] = stats[key]/nt
            if key not in allstats:
                allstats[key] = []
            allstats[key].append(stats[key].copy())
    
    for key in allstats:
        filename = os.path.join(EVALS,key+'.nc')
        print(filename)
        # xr.merge(allstats[key]).to_netcdf(filename,mode = 'w')
        ds = xr.merge(allstats[key])
        print(ds)
        ds.to_netcdf(filename,mode = 'w')


            

            






if __name__=='__main__':
    main()
