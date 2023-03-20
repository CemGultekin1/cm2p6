import sys
from data.paths import get_filter_weights_location
from data.load import  get_filter_weights_generator
from run.train import Timer
from utils.arguments import options
from utils.slurm import flushed_print
from utils.xarray import plot_ds
import xarray as xr
import numpy as np

def disp(fw,wm,coords,t):
    coords.pop('lat')
    coords.pop('lon')
    shp = [len(v) for v in coords.values()]

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

def run():
    # datargs = '--minibatch 1 --prefetch_factor 1 --disp 1 --depth 0 --disp 100 --sigma 4 --section 5 10 --mode data --num_workers 1 --filtering gcm'.split()
    datargs = sys.argv[1:]
    generators = get_filter_weights_generator(datargs,data_loaders = True)
    for ut,grid_gen in zip('u t'.split(),generators):
        filename = get_filter_weights_location(datargs,preliminary=True,utgrid = ut)
        flushed_print(f'filename = {filename}')
        args,_ = options(datargs,key = "run")
        time = Timer()
        time.start('data')
        fweights = None
        # np.random.seed(0)
        for t,(coords,locs,fw,_) in enumerate(grid_gen):
            time.end('data')
            
            # if not disp(fw,wm,coords,t):
            #     continue
            # else:
            #     return
            if fweights is None:
                shp = [len(v) for v in coords.values()]
                fweights = xr.DataArray(
                    data = np.zeros(shp,dtype = float),
                    dims = list(coords.keys()),
                    coords = {key:val.numpy() for key,val in coords.items()}
                )
            fweights.data[locs[0],locs[1]] = fw.numpy()
            if t%args.disp == 0:
                flushed_print(t,str(time))
            if t % 128 == 0:
                fweights.to_netcdf(filename)
            time.start('data')
        fweights.to_netcdf(filename)
    
if __name__=='__main__':
    run()
