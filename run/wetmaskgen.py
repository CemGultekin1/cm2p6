import sys
from data.paths import get_low_res_data_location, get_preliminary_low_res_data_location
from data.load import get_data
from run.train import Timer
from utils.arguments import options
from utils.xarray import plot_ds
from utils.slurm import flushed_print
import xarray as xr
import torch
def torch2numpy(data_vars,coords):
    for key in data_vars:
        dims,val = data_vars[key]
        if isinstance(val,torch.Tensor):
            data_vars[key] = (dims,val.numpy())
    for key in coords:
        if isinstance(coords[key],torch.Tensor):
            coords[key] = coords[key].numpy()
    return data_vars,coords
def drop_timeless(ds:xr.Dataset):
    timeless_params = []
    for key in ds.data_vars.keys():
        if 'time' not in ds[key].dims:
            timeless_params.append(key)
    for tp in timeless_params:
        ds = ds.drop(tp)
    return ds

def run():
    # datargs = sys.argv[1:]
    datargs = '--minibatch 1 --prefetch_factor 1 --spacing long_flat --depth 0 --sigma 4 --section 0 1 --mode data --num_workers 1 --filtering gaussian'.split()
    generator,= get_data(datargs,half_spread = 0, torch_flag = False, data_loaders = True,groups = ('all',))
    filename = get_low_res_data_location(datargs)
    print(f'filename = {filename}')
    datargs,_ = options(datargs,key = "data")
    time = Timer()
    time.start('data')
    for data_vars,coords in generator:
        time.end('data')
        
        data_vars,coords = torch2numpy(data_vars,coords)
        ds = xr.Dataset(data_vars = data_vars,coords = coords)
        dsvars = [k for k in ds.data_vars.keys() if k not in 'wet_density interior_wet_mask'.split()]
        ds = ds.drop(dsvars).drop('time').load()
        chk = {k:len(ds[k]) for k in list(ds.coords)}
        ds = ds.chunk(chunks=chk)
        print(filename)
        print(ds)
        ds.to_zarr(filename,mode='a')
        break
        
if __name__=='__main__':
    run()