import sys
from data.paths import get_filter_weights_location, get_learned_deconvolution_weights
from data.load import  get_deconvolution_generator, get_filter_weights_generator
from run.train import Timer
from utils.arguments import options
from utils.slurm import flushed_print
from utils.xarray import plot_ds
import xarray as xr
import numpy as np

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
    datargs = sys.argv[1:]
    # datargs = '--minibatch 1 --prefetch_factor 1 --disp 1 --depth 0 --disp 100 --sigma 4 --section 0 1 --mode data --num_workers 1 --filtering gcm'.split()
   
    generators, = get_deconvolution_generator(datargs,data_loaders = True)
    filename = get_learned_deconvolution_weights(datargs,preliminary=True)
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
        subfield  = xr.Dataset(
            data_vars = data_vars,
            coords = coords
        )
        # print('\n',subfield,'\n')
        if fields is None:
            fields = subfield
        else:
            fields += subfield
        if t%args.disp == 0:
            flushed_print(t,str(time))
        if t % 128 == 0:
            fields.to_netcdf(filename)
        time.start('data')
    flushed_print(f'now saving...')        
    fields.to_netcdf(filename)
    flushed_print(f'done')
if __name__=='__main__':
    main()
