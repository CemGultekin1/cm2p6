import sys
from data.paths import  get_learned_deconvolution_location
from utils.slurm import flushed_print
from constants.paths import JOBS
from utils.arguments import options
import xarray as xr
import numpy as np
import os
NSEC = 10
def run():
    arg = int(sys.argv[1]) - 1
    path = os.path.join(JOBS,'learndeconv.txt')
    with open(path) as f:
        ls = f.readlines()

    ls = [l.strip() for l in ls]
    upper_limit = (arg+1)*NSEC
    lower_limit = arg*NSEC 
    flushed_print('lower_limit,upper_limit\t',lower_limit,upper_limit)
    for i in range(lower_limit,upper_limit):
        datargs = ls[i].split()   
        break
        path1 = get_learned_deconvolution_location(datargs,preliminary=True).replace('.nc','_.nc')
        if i == 0:
            fw0 = xr.open_dataset(path1)
        else:
            fw1 = xr.open_dataset(path1)
            fw0 += fw1
        
        flushed_print(i,lower_limit,upper_limit)
    path0 = get_learned_deconvolution_location(datargs,preliminary=False)
    # fw0.to_netcdf(path0,mode='w',)
    # flushed_print(f'saved {path0}')
    # return
    
    fw0 = xr.open_dataset(path0,mode = 'r').load()
    
    spatial_encoding_degree=5
    spatial_encoding_degree_=3
    coarse_spread=10
    correlation_spread=3
    correlation_distance=2
    correlation_spatial_encoding_degree=1#2
    fspan = 9
    cspan = coarse_spread*2 + 1
   
    num_chan2 = (2*correlation_spatial_encoding_degree**2 - 1)*(correlation_distance+1)**2
    num_outs = fspan**2
    
    num_chan1 = (2*spatial_encoding_degree**2 - 1)
    num_feats1 = num_chan1*cspan**2
    
    num_chan1_ = (2*spatial_encoding_degree_**2 - 1)
    num_feats1_ = num_chan1_*cspan**2
    
    corrspan = correlation_spread*2+1
    num_feats2 = num_chan2*corrspan**2
    
    
    inds = list(range(num_feats1_)) #+ list(range(num_feats1,num_feats2+num_feats1))
    datargs,_ = options(datargs,key = 'data')
    coords = 'grid depth'.split()
    nslice = [[{c:cc} for cc in fw0[c].values] for c in coords]
    import itertools
    
    fw0 = fw0.isel(ninputs1 = inds,ninputs2 = inds)
    shp = list(fw0.xy.shape)
    coeffsmat = np.zeros(shp)
    print(f'coeffsmat.shape = {coeffsmat.shape}')
    for sels in itertools.product(*nslice):
        flushed_print(f'sels = {sels}')
        x = fw0
        for sel in sels:
            x = x.sel(sel)
        xx = x.xx.values
        xy = x.xy.values
        g = 1/np.sqrt(np.diag(xx)).reshape([-1,1])
        
        xx = (g*xx)*g.T
        xy = g*xy
        
        # flushed_print(f'np.amax(xx) = {np.amax(xx)},\t np.amin(np.abs(xx)) = {np.amin(np.abs(xx))},\t np.mean(np.abs(xx)) = {np.mean(np.abs(xx))}')
        # xxhalf = np.linalg.cholesky(xx + 1e-5 *np.eye(xx.shape[0]))
        # coeffs = g*np.linalg.solve(xxhalf.T,np.linalg.solve(xxhalf,xy))
        u,s,vh = np.linalg.svd(xx)
        sinv = np.diag(1/np.where(s/s[0]<1e-4,np.inf,s))
        coeffs = g*(vh.T@(sinv@(u.T@xy)))
        coeffsmat[sels[0]['grid'],sels[1]['depth']] = coeffs
    fw0['coeffs'] = (fw0.xy.dims,coeffsmat)
    path0 = path0.replace('.nc','_.nc')
    fw0.to_netcdf(path0,mode='w',)
    flushed_print(f'saved {path0}')
        
if __name__=='__main__':
    run()
