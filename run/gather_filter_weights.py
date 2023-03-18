import sys
from data.paths import get_filter_weights_location
from transforms.gcm_inversion import FilterWeightCompression
from utils.slurm import flushed_print
from utils.paths import JOBS
from utils.arguments import options
import xarray as xr
import os
NSEC = 10
def run():
    arg = int(sys.argv[1]) - 1
    path = os.path.join(JOBS,'filter_weights.txt')
    with open(path) as f:
        ls = f.readlines()

    ls = [l.strip() for l in ls]
    upper_limit = (arg+1)*NSEC#40
    lower_limit = arg*NSEC #30
    flushed_print('lower_limit,upper_limit\t',lower_limit,upper_limit)

    for i in range(lower_limit,upper_limit):
        datargs = ls[i].split()   
        path1 = get_filter_weights_location(datargs,preliminary=True)
        if i == 0:
            fw0 = xr.open_dataset(path1)
        else:
            fw1 = xr.open_dataset(path1)
            fw0 += fw1
        path0 = get_filter_weights_location(datargs,preliminary=False)
        fw0.to_netcdf(path0)
        print(i,lower_limit,upper_limit)
    datargs,_ = options(datargs,key = 'data')
    fwc = FilterWeightCompression(datargs.sigma,fw0.__xarray_dataarray_variable__)
    print('ranking filters...')
    ds = fwc.get_separable_components()
    ds.to_netcdf(path0,mode='w',)
        
if __name__=='__main__':
    run()
