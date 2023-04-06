import itertools
import os
import matplotlib.pyplot as plt
from plots.metrics import metrics_dataset
from utils.paths import LEGACY_PLOTS,LEGACY
from utils.xarray import skipna_mean
import xarray as xr
from utils.arguments import options
from utils.slurm import flushed_print
import numpy as np

def main():
    root = LEGACY
    target = LEGACY_PLOTS
    if not os.path.exists(target):
        os.makedirs(target)
    lines = ['--filtering gaussian --min_precision 0.024 --domain four_regions --batchnorm 1 1 1 1 1 1 1 0 --widths 2 128 64 32 32 32 32 32 4 --kernels 5 5 3 3 3 3 3 3 --minibatch 4']
    for line in lines:
        modelargs,modelid = options(line.split(),key = "model")
        snfile = os.path.join(root,modelid + '_.nc')
        if not os.path.exists(snfile):
            continue
        sn = xr.open_dataset(snfile).sel(lat = slice(-85,85))#.isel(depth = [0],co2 = 0).drop(['co2'])

        sn = sn.isel(co2 = 0,depth= 0).drop('co2 depth'.split())



        names = "Su Sv".split()
        unames = np.unique([n.split('_')[0] for n in list(sn.data_vars)])
        names = [n for n in names if n in unames]
        ftypes = ['mean','std']
        
        nrows = len(names)
        ncols = len(ftypes)
        _names = np.empty((nrows,ncols),dtype = object)
        for ii,jj in itertools.product(range(nrows),range(ncols)):
            n = f"{names[ii]}_{ftypes[jj]}"
            _names[ii,jj] = n

        targetfile = os.path.join(target,f'{modelid}.png')

        fig,axs = plt.subplots(nrows,ncols,figsize = (ncols*6,nrows*5))
        for ir,ic in itertools.product(range(nrows),range(ncols)):
            name = _names[ir,ic]
            mse = sn[name + '_mse']
            sc2 = sn[name + '_sc2']
            pkwargs = dict(vmin = 0,vmax = 1,cmap = 'seismic')
            var = 1-mse/sc2
            var.plot(ax = axs[ir,ic],**pkwargs)
            mse = np.mean(var.values).item()
            subtitle = f"R2({name})"
            axs[ir,ic].set_title(subtitle,fontsize=24)
        suptitle = 'asdf'
        fig.suptitle(suptitle,fontsize=24)
        fig.savefig(targetfile)
        flushed_print(targetfile)
        plt.close()



if __name__=='__main__':
    main()