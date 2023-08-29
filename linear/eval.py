import logging
import numpy as np
from constants.paths import FILTER_WEIGHTS,TEMPORARY_DATA, EVALS
import os
from utils.arguments import options, replace_params
import xarray as xr
import itertools
from utils.slurm import basic_config_logging
import matplotlib.pyplot as plt
import matplotlib

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
    
    # args = f'--model lsrp:0 --sigma 12 --lsrp True --num_workers 1 --temperature True --filtering gcm --mode eval'.split()
    
    # from utils.slurm import read_args
    # args = read_args(289,filename = 'offline_sweep.txt')
    # args = replace_params(args,'mode','eval','lsrp',1,'temperature','True',)
    # args = f'--model lsrp:0 --sigma {4} --filtering gcm'.split()
    # _,lsrpid = options(args,key = "model")
    # print(lsrpid)
    # path = os.path.join(EVALS,lsrpid + '.nc')
    # ds = xr.open_dataset(path)
    # print(ds)
    # return
    basic_config_logging()
    fs = os.listdir(TEMPORARY_DATA)
    fs = [f for f in fs if 'linear' in f]
    paths = [os.path.join(TEMPORARY_DATA,f) for f in fs]
    logging.info(f'# paths found = {len(paths)}')
    paths = paths[:10]
    for path,f in zip(paths,fs):        
        ds = xr.open_zarr(path)
        ds = ds.fillna(0)
        logging.info(f)
        logging.info(f'length = {len(ds.time)}')
        ds0,ds1 = dataset_split(ds)
        err =  np.square(ds0 - ds1)
        sc2 = np.square(ds0)
        mse = err.mean(dim = "time")
        sc2 = sc2.mean(dim = "time")
        
        r2 = 1 - mse/sc2
        r2val = 1 - mse.mean()/sc2.mean()
        # for var in r2.data_vars:
        #     r2[var].plot(vmin = 0, vmax = 1,cmap = matplotlib.cm.magma)
        #     r2v = r2val[var].values.item()
        #     formatter = "{:.2e}"
        #     plt.title(formatter.format(r2v))
        #     plt.savefig(f'{f.split(".")[0]}_r2map_{var}.png')
        #     plt.close()
            
            
        # return
if __name__ == '__main__':
    main()