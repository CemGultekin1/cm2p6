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
    shp = list(fw0.xy.shape)
    maxdegree = 1
    numcoeffs = 2*11**2*maxdegree**2
    shp[2] = numcoeffs
    coeffsmat = np.zeros(shp)
    for sels in itertools.product(*nslice):
        flushed_print(f'sels = {sels}')
        x = fw0
        for sel in sels:
            x = x.sel(sel)
        xx = x.xx.values[:numcoeffs,:numcoeffs]
        xy = x.xy.values[:numcoeffs,:]
        
        flushed_print(f'np.amax(xx) = {np.amax(xx)},\t np.amin(xx) = {np.amin(xx)},\t np.mean(np.abs(xx)) = {np.mean(np.abs(xx))}')
        # xxhalf = np.linalg.cholesky(xx )#+ 1e-5 *np.eye(xx.shape[0]))
        # coeffs = np.linalg.solve(xxhalf.T,np.linalg.solve(xxhalf,xy))
        u,s,vh = np.linalg.svd(xx,full_matrices = False)
        sinv = np.diag(1/np.where(s/s[0]<1e-5,np.inf,s))
        coeffs = vh.T@sinv@u.T@xy
        coeffsmat[sels[0]['grid'],sels[1]['depth']] = coeffs
    fw0 = fw0.isel(
        ninputs1 = range(numcoeffs),ninputs2 = range(numcoeffs)
    )
    fw0['coeffs'] = (fw0.xy.dims,coeffsmat)
    fw0.to_netcdf(path0,mode='w',)
    flushed_print(f'saved {path0}')
        
if __name__=='__main__':
    run()
