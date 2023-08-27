import itertools
import logging
from constants.paths import TEMPORARY_DATA
from data.coords import TIMES
from data.load import load_xr_dataset
from linear.coarse_graining_inverter import CoarseGrainingInverter
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.sparse as sp
from transforms.grids import get_grid_vars
from utils.xarray_oper import concat
import xarray as xr
import logging
import sys
from transforms.subgrid_forcing import GcmSubgridForcing


def main():    
    sigma,depth = 16,5
    filename = f'linear_sgm_{sigma}_dpth_{depth}.zarr'
    path = os.path.join(TEMPORARY_DATA,filename)
    ds = xr.open_zarr(path)
    logging.info(ds)
    ds = ds.isel(time = 0,depth = 0).fillna(0)
    

    forcings = 'Su Sv Stemp'.split()
    forcings_linear = [v+ '_linear' for v in forcings]
    comps = [(ds[f0] ,ds[f1]) for f0, f1 in zip(forcings,forcings_linear)]
    r2vals = [1 - np.square(x-y).mean().values.item()/np.square(x).mean().values.item() for x,y in comps]
    print(r2vals)
    
    vns = 'Su Sv Stemp'.split()
    vns.extend([v + '_linear' for v in vns])
    datasets = [ds[key] for key in vns]
    
    ncols = 3
    nrows = 1
    fig,axs = plt.subplots(nrows,ncols,figsize = (5*ncols,5*nrows))#,sharex=True, sharey=True)
    axs = axs.flat
    for i,ax in enumerate(axs):
        x,y = datasets[i].values.flatten(),datasets[i+3].values.flatten()
        x = np.where(x>=0,np.log10(x),-np.log10())
        # ax.plot(vecs[0],vecs[2],'.')
        # ax.set_xlim([])
        # for vec,lbl in zip(vecs,'true pred err'.split()):
        #     n = len(vec)
        #     l = int(n*0.5)
        #     # print(f'{l},{n}')
            
        #     vec = vec[l:n]
        #     # print(f'len(vec) = {len(vec)}')
        #     ax.semilogy(vec,label = lbl)
        # np.log10(np.abs(dataset)).plot(ax = ax,vmin = -10,vmax = -5,cmap = 'binary')
        
        # n = len(vecs[0])
        # ax.set_xlim([n*0.8,n*1.05])
        # ax.set_ylim([1e-8,1e-4])
    fig.savefig(f'ds_sgm_{sigma}_dpth_{depth}.png')
    plt.close()
    
if __name__ == '__main__':
    main()