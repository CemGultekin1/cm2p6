import sys
from data.paths import get_filter_weights_location
from data.load import  get_filter_weights_generator
from run.train import Timer
from utils.arguments import options
from utils.slurm import flushed_print
import xarray as xr
import numpy as np
def run():
    datargs = sys.argv[1:]
    generators = get_filter_weights_generator(datargs,data_loaders = True)
    for ut,grid_gen in zip('u t'.split(),generators):
        if ut == 'u':
            continue
        
        filename = get_filter_weights_location(datargs,preliminary=True,utgrid = ut)
        flushed_print(f'filename = {filename}')
        args,_ = options(datargs,key = "run")
        time = Timer()
        time.start('data')
        fweights = None
        for t,(coords,locs,fw) in enumerate(grid_gen):
            time.end('data')
            
            if fweights is None:
                shp = [len(v) for v in coords.values()]
                fweights = xr.DataArray(
                    data = np.zeros(shp,dtype = float),
                    dims = list(coords.keys()),
                    coords = {key:val.numpy() for key,val in coords.items()}
                )
            fweights.data[locs[0].item(),locs[1].item()] = fw.numpy()
            if t%args.disp == 0:
                flushed_print(t,str(time))
            if t % 128 == 0:
                fweights.to_netcdf(filename)
            time.start('data')
        fweights.to_netcdf(filename)
    
if __name__=='__main__':
    run()
