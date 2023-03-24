import sys
from data.paths import  get_learned_deconvolution_location
from utils.slurm import flushed_print
from utils.paths import JOBS
from utils.arguments import options
import xarray as xr
import numpy as np
import os
NSEC = 10
def run():
    arg = int(sys.argv[1]) - 1
    path = os.path.join(JOBS,'learned_deconv.txt')
    with open(path) as f:
        ls = f.readlines()

    ls = [l.strip() for l in ls]
    upper_limit = (arg+1)*NSEC
    lower_limit = arg*NSEC 
    flushed_print('lower_limit,upper_limit\t',lower_limit,upper_limit)
    for i in range(lower_limit,upper_limit):
        datargs = ls[i].split()   
        path1 = get_learned_deconvolution_location(datargs,preliminary=True)
        if i == 0:
            fw0 = xr.open_dataset(path1)
        else:
            fw1 = xr.open_dataset(path1)
            fw0 += fw1
        path0 = get_learned_deconvolution_location(datargs,preliminary=False)
        flushed_print(i,lower_limit,upper_limit)
    datargs,_ = options(datargs,key = 'data')
    coords = 'grid depth'.split()
    nslice = [[{c:cc} for cc in fw0[c].values] for c in coords]
    import itertools
    coeffsmat = np.zeros(fw0.xy.shape)
    for sels in itertools.product(*nslice):
        flushed_print(f'sels = {sels}')
        x = fw0
        for sel in sels:
            x = x.sel(sel)
        xx = x.xx
        xy = x.xy
        xxhalf = np.linalg.cholesky(xx + 1e-9 *np.eye(xx.shape[0]))
        coeffs = np.linalg.solve(xxhalf.T,np.linalg.solve(xxhalf,xy))
        coeffsmat[sels[0]['grid'],sels[1]['depth']] = coeffs
    fw0['coeffs'] = (fw0.xy.dims,coeffsmat)
    fw0.to_netcdf(path0,mode='w',)
    flushed_print(f'saved {path0}')
        
if __name__=='__main__':
    run()
