
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
    args = f'--sigma {sigma} --filtering gcm --lsrp 0 --mode data'.split()
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
    fds,_ = load_xr_dataset(args,high_res = True)
    varname = 'u'
    cds = cds[varname].isel(time = 0)
    
    coeffs = dfw.coeffs
    degree = 3 #int(np.sqrt(coeffs.shape[0]//11**2//2))
    ev = Eval(sigma,coeffs,degree = degree)
    
    eff = ev.effective_filter(cds,110,110)
    nrow,ncol = 9,9
    # rowslice = slice(eff.shape[2] - nrow//2,eff.shape[2] + nrow//2 + 1)
    # colslice = slice(eff.shape[2] - ncol//2,eff.shape[2] + ncol//2 + 1)
    # eff = eff[:,:,rowslice,:]
    # eff = eff[:,:,:,colslice]
    print(eff.shape)
    fig,axs = plt.subplots(nrow,ncol,figsize = (10*ncol,10*nrow))
    
    for i,j in itertools.product(range(nrow),range(ncol)):
        ax = axs[i,j]
        f = eff[:,:,i,j]
        vmax = np.amax(np.abs(f))
        pos = ax.imshow(f,cmap = 'bwr',vmin = -vmax,vmax = vmax)
        fig.colorbar(pos,ax = ax)
    fig.savefig('eff_inv_filter.png')
    plt.close()
    nrow,ncol = 1,1
    fig,axs = plt.subplots(nrow,ncol,figsize = (10*ncol,10*nrow))
    
    f = np.log10(np.mean(np.mean(eff**2,axis = 3),axis = 2))
    f = f - np.amax(f)
    # vmax = np.amax(np.abs(f))
    pos = axs.imshow(f,cmap = 'seismic',)
    fig.colorbar(pos,ax = axs)
    fig.savefig('filter_concentration.png')
    return
    limits = (100,150,100,150)
    ypred = ev.eval(cds,limits = limits)
    key = 't' if varname == 'temp' else 'u'
    iseldict = {key+dim:slice(limits[2*i]*sigma,limits[2*i+1]*sigma) for i,dim in enumerate('lat lon'.split())}
    ytrue = fds[varname].isel(time = 0,**iseldict).values
    
    fig,axs = plt.subplots(1,2,figsize = (20,10))
    
    ax = axs[0]
    vmax = np.amax(np.abs(ypred))
    pos = ax.imshow(ypred,cmap = 'bwr',vmin = -vmax,vmax = vmax)
    fig.colorbar(pos,ax = ax)
    
    ax = axs[1]
    vmax = np.amax(np.abs(ytrue))
    pos = ax.imshow(ytrue,cmap = 'bwr',vmin = -vmax,vmax = vmax)
    fig.colorbar(pos,ax = ax)
    
    fig.savefig('predicted_hres.png')
    

if __name__ == '__main__':
    main()