
from data.load import load_learned_deconv
from transforms.learned_deconv import Eval
import numpy as np
from utils.xarray import plot_ds
import xarray as xr 
from transforms.grids import get_grid_vars
from utils.arguments import options
import matplotlib.pyplot as plt
import itertools
def main():
    sigma = 4
    args = f'--sigma {sigma} --filtering gcm --lsrp 1 --mode data'.split()
    dfw = load_learned_deconv(args).isel(grid = 0,depth = 0)
    print(dfw)
    
    # xx = dfw.xx.values
    # xy = dfw.xy.values
    # xxhalf = np.linalg.cholesky(xx + 1e-3*np.eye(xx.shape[0]))
    # coeffs = np.linalg.solve(xxhalf.T,np.linalg.solve(xxhalf,xy))
    # coeffs = dfw.coeffs.values.reshape([11,11,4,4,2,9,9])
    # nr = 9
    # fig,axs = plt.subplots(nr,nr,figsize = (30,30))
    # for i,j in itertools.product(range(nr),range(nr)):
    #     y = coeffs[...,1,1,1,i,j]
    #     vmax = np.amax(np.abs(y))
    #     ax = axs[i,j]
    #     pos = ax.imshow(y,cmap = 'bwr',vmin = -vmax,vmax = vmax)
    #     fig.colorbar(pos,ax = ax)
    # fig.savefig('filters_look.png')
        
        
        
    # return
    from data.load import load_xr_dataset
    args = '--sigma 4 --filtering gcm'.split()
    cds,_ = load_xr_dataset(args,high_res = False)
    ev = Eval(sigma,dfw.coeffs)
    y = ev.eval(cds.u.isel(time = 0),limits = (100,130,100,130))
    print(y.shape)
    vmax = np.amax(np.abs(y))
    fig,axs = plt.subplots(1,2,figsize = (20,10))
    ax = axs[0]
    pos = ax.imshow(y,cmap = 'bwr',vmin = -vmax,vmax = vmax)
    fig.colorbar(pos,ax = ax)
    fig.savefig('predicted_hres.png')
    

if __name__ == '__main__':
    main()