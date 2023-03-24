import sys
from data.paths import get_learned_deconvolution_location
from data.load import  get_deconvolution_generator
from run.train import Timer
from utils.arguments import options
from utils.slurm import flushed_print
from utils.xarray import plot_ds
import xarray as xr


def disp(fw,wm,coords,t):
    coords.pop('lat')
    coords.pop('depth')
    coords.pop('lon')

    fweights = xr.DataArray(
        data = fw.numpy(),
        dims = list(coords.keys()),
        coords = {key:val.numpy() for key,val in coords.items()}
    )
    wet_mask = xr.DataArray(
        data = wm.numpy(),
        dims = list(coords.keys()),
        coords = {key:val.numpy() for key,val in coords.items()}
    )
    plot_ds(dict(fweights = fweights,wet_mask = wet_mask),f'fweights_{t}.png',ncols = 2,dims = fweights.dims)
    return t == 32

def main():
    # datargs = sys.argv[1:]
    datargs = '--minibatch 1 --prefetch_factor 1 --disp 1 --depth 0 --disp 100 --sigma 4 --section 0 1 --mode data --num_workers 1 --filtering gcm'.split()
   
    generators, = get_deconvolution_generator(datargs,data_loaders = True)
    filename = get_learned_deconvolution_location(datargs,preliminary=True)
    flushed_print(f'filename = {filename}')
    args,_ = options(datargs,key = "run")
    time = Timer()
    time.start('data')
    fields = None
    for t,(coords,(indims,xx),(outdims,xy)) in enumerate(generators):
        time.end('data')
        dims = list(coords.keys())
        indims = [dims[i] for i in indims]
        outdims = [dims[i] for i in outdims]
        data_vars = dict(
            xx = (indims,xx.numpy()),
            xy = (outdims,xy.numpy()),
        )
        if xy.numpy().sum() == 0:
            continue
        print(xy.shape)
        coeffs = xy.numpy()
        coeffs = coeffs[0,0].reshape([11,11,4,4,2,9,9])
        nr = 4
        import matplotlib.pyplot as plt
        import itertools
        import numpy as np
        fig,axs = plt.subplots(nr,nr,figsize = (30,30))
        for i,j in itertools.product(range(nr),range(nr)):
            y = coeffs[...,0,0,1,i,j]
            vmax = np.amax(np.abs(y))
            ax = axs[i,j]
            pos = ax.imshow(y,cmap = 'bwr',vmin = -vmax,vmax = vmax)
            fig.colorbar(pos,ax = ax)
        fig.savefig('filters_look.png')
        return
        subfield  = xr.Dataset(
            data_vars = data_vars,
            coords = coords
        )
        if fields is None:
            fields = subfield
        else:
            fields += subfield
        if t%args.disp == 0:
            flushed_print(t,str(time))
        if t % 256 == 0:
            fields.to_netcdf(filename)
        time.start('data')
    flushed_print(f'now saving...')        
    fields.to_netcdf(filename)
    flushed_print(f'done')
if __name__=='__main__':
    main()
