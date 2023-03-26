
from data.load import load_learned_deconv
from transforms.learned_deconv import DeconvolutionTransform
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
    varname = 'temp'
    time = 1234
    cds = cds[varname].isel(time = time)
    
    coeffs = dfw.coeffs
    degree = 3 #int(np.sqrt(coeffs.shape[0]//11**2//2))
    ev = DeconvolutionTransform(sigma,coeffs,degree = degree)
    ev.create_feature_maps(cds)
    ypred = ev.eval(cds,)

    # key = 't' if varname == 'temp' else 'u'
    iseldict = {}#key+dim:slice(limits[2*i]*sigma,limits[2*i+1]*sigma) for i,dim in enumerate('lat lon'.split())}
    ytrue = fds[varname].isel(time = time,**iseldict).fillna(0).values
    
    limits = ((),(1000,1500,1000,1500),(2000,2500,1000,1500),(500,1000,2000,2500))
    
    fig,axs = plt.subplots(len(limits),3,figsize = (10*3,len(limits)*10))
    
    
    vmax1 = np.amax(np.abs(ypred))
    vmax2 = np.amax(np.abs(ytrue))
    vmax = np.maximum(vmax1,vmax2)*1.1
    ypred[ytrue == 0] = np.nan
    ytrue[ytrue == 0] = np.nan
    
    for i in range(len(limits)):
        yp = ypred[::-1]
        yt = ytrue[::-1]
        
        lim = limits[i]
        if bool(lim):
            yp = yp[lim[0]:lim[1],lim[2]:lim[3]]
            yt = yt[lim[0]:lim[1],lim[2]:lim[3]]
        ax = axs[i,0]
        pos = ax.imshow(yp,cmap = 'bwr',)
        fig.colorbar(pos,ax = ax)
        
        ax = axs[i,1]
        pos = ax.imshow(yt,cmap = 'bwr',)
        fig.colorbar(pos,ax = ax)
        
        ax = axs[i,2]
        pos = ax.imshow(np.abs(yp - yt), cmap = 'seismic',vmin = 0)
        fig.colorbar(pos,ax = ax)
    
    fig.savefig('predicted_hres.png')
    

if __name__ == '__main__':
    main()