import os
import sys
from data.paths import get_low_res_data_location, get_preliminary_low_res_data_location
from utils.slurm import flushed_print
import xarray as xr
import numpy as np
NSEC= 15
def append_zarr(path0,path1,overwrite):
    if not os.path.exists(path1):
        print(path1.split('/')[-1])
        return 
    ds = xr.open_zarr(path1)
    batchsize = 20
    nt =len(ds.time.values)
    batchinds = np.arange(0,nt,batchsize).tolist() + [nt]
    for i0,i1 in zip(batchinds[:-1],batchinds[1:]):
        dst = ds.isel(time = slice(i0,i1)).load()
        if i0==0 and overwrite:
            dst.to_zarr(path0,mode = 'w')
        else:
            dst.to_zarr(path0,mode = 'a',append_dim = 'time')
        flushed_print(path0.split('/')[-1],path1.split('/')[-1],i0,i1)
def get_interior_wet_mask(datargs):
    path0 = get_low_res_data_location(datargs)
    ds = xr.open_zarr(path0)
    interior_wet_mask = ds.interior_wet_mask
    interior_wet_mask.load()
    return interior_wet_mask
def run():
    arg = int(sys.argv[1]) - 1
    from constants.paths import JOBS
    path = os.path.join(JOBS,'datagen.txt')
    with open(path) as f:
        ls = f.readlines()

    ls = [l.strip() for l in ls]
    upper_limit = (arg+1)*NSEC#40
    lower_limit = arg*NSEC #30
    flushed_print('lower_limit,upper_limit\t',lower_limit,upper_limit)
    for i in range(lower_limit,upper_limit):
        datargs = ls[i].split()
        path1 = get_preliminary_low_res_data_location(datargs)
        path0 = get_low_res_data_location(datargs).replace('.zarr','_.zarr')
        overwrite =  i == lower_limit
        # print(f'append_zarr({path0},{path1},{overwrite})')
        # break
        append_zarr(path0,path1,overwrite)
                

        




if __name__=='__main__':
    run()
