    
import itertools
import logging
import os
import sys
from data.exceptions import RequestDoesntExist
from plots.metrics_ import moments_dataset_xr
from run.train import Timer
import torch
from data.load import get_data
from data.vars import get_var_mask_name
from models.load import load_model, load_old_model
import matplotlib.pyplot as plt
from utils.arguments import options, populate_data_options
from utils.parallel import get_device
from constants.paths import EVALS,TEMPORARY_DATA
from utils.slurm import flushed_print
import numpy as np
from utils.xarray_oper import fromtensor, fromtorchdict, fromtorchdict2tensor, plot_ds
import xarray as xr
from utils.arguments import replace_params
from utils.slurm import basic_config_logging

def get_lsrp_modelid(args):
    runargs,_ = options(args)
    args = f'--model lsrp:0 --sigma {runargs.sigma} --filtering {runargs.filtering}'.split()
    _,lsrpid = options(args,key = "model")
    return True, lsrpid

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
    stats_ = moments_dataset_xr(prd,tr)
    if key not in stats:
        stats[key] = stats_
    else:
        stats[key] = stats[key] + stats_
    return stats

def dataset_split(ds):
    dv = 'Su Sv Stemp'.split()
    dropkeys = [key for key in ds.data_vars.keys() if key not in dv]
    ds0 = ds.drop(dropkeys)
    
    dv1 = [d + '_linear' for d in dv]
    dropkeys = [key for key in ds.data_vars.keys() if key not in dv1]
    ds1 = ds.drop(dropkeys)
    ds1 = ds1.rename(dict(tuple(zip(dv1,dv))))
    return ds0,ds1


def main():
    args = sys.argv[1:]
    args = f'--model lsrp:0 --sigma {args[0]} --lsrp True --num_workers 1 --temperature True --filtering gcm --mode eval'.split()
    args = replace_params(args,'mode','eval','lsrp',1,'temperature','True',)    
    basic_config_logging()
    
    runargs,_ = options(args,key = 'run')

    lsrp_flag, lsrpid = get_lsrp_modelid(args)
    

    
    assert runargs.mode == "eval"
    non_static_params=['depth','co2',]
    multidatargs = populate_data_options(args,non_static_params=non_static_params,domain = 'global',interior = False)
    # multidatargs = [args]
    allstats = {}
    for datargs in multidatargs:
        runargs,_ = options(datargs)
        if runargs.co2:
            filename = f'linear_sgm_{runargs.sigma}_dpth_{int(runargs.depth)}_co2.zarr'
        else:
            filename = f'linear_sgm_{runargs.sigma}_dpth_{int(runargs.depth)}.zarr'        
        path = os.path.join(TEMPORARY_DATA,filename)
        if not os.path.exists(path):
            continue
        if runargs.filtering not in 'gcm gaussian'.split():
            continue
        logging.info(filename)
        ds = xr.open_zarr(path)
        ds = ds.fillna(0)
        # ds = ds.isel(time = slice(0,16))
        ds0,ds1 = dataset_split(ds)
        stats = {}
        stats = update_stats(stats,ds1,ds0,lsrpid)
        coords = dict(
            depth = runargs.depth,
            co2 = 0.01 if runargs.co2  else 0.,
            filtering = runargs.filtering,
        )

        ds = stats[lsrpid].compute().load()
        for key in coords:
            if key not in ds.coords:
                ds = ds.expand_dims({key:[coords[key]]},axis = 0)
            else:
                ds[key] = [coords[key]]
        stats[lsrpid] = ds
        for key in stats:
            if key not in allstats:
                allstats[key] = []
            allstats[key].append(stats[key].copy())
            
        
    for key in allstats:
        filename = os.path.join(EVALS,key+'_.nc')
        logging.info(f'merging...')
        ds = xr.merge(allstats[key])
        logging.info(f'saving to {filename}')
        ds.to_netcdf(filename,mode = 'w')
        logging.info(f'\t\t...{filename} is saved')
        logging.info(ds)


            

            






if __name__=='__main__':
    main()
