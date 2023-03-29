# from data.load import load_xr_dataset
from utils.xarray import plot_ds
import xarray as xr
import numpy as np
import os 
def main():
    dss = {}
    root = '/scratch/zanna/data/cm2.6/coarse_datasets/'
    sigma = 4
    paths = os.listdir(root)
    sigmas = [int(p.split('_')[1]) for p in paths]
    surf = ['beneath' not in p for p in paths]
    paths = [p for p,s,sr in zip(paths,sigmas,surf) if s == sigma and sr]
    for path in paths:
        ds = xr.open_zarr(os.path.join(root,path))
        if 'u' not in ds.data_vars:
            print(path)
            continue
        dss[path] = ds
   
    
    vars = 'u Su'.split()
    slc = slice(150,250)
    for var in vars:        
        dsvar = {x:y[var].isel(time = 0,lat = slc,lon = slc,depth = 0) for x,y in dss.items()}
        dsvar = {f'{x0}-{x1}':np.log10(np.abs(y0 - y1)) for x0,y0 in dsvar.items() for x1,y1 in dsvar.items()}
        plot_ds(dsvar,f'gcm_gaussian_{var}.png',ncols = len(dss))
    
    

if __name__ == '__main__':
    main()