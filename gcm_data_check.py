from data.load import load_xr_dataset
from utils.xarray import plot_ds
import numpy as np

def main():
    dss = {}
    args = '--lsrp 1 --sigma 4 --filtering gcm'.split()
    dss['gcm1'],_ = load_xr_dataset(args,high_res = False)

    args = '--lsrp 0 --sigma 4 --filtering gcm'.split()
    dss['gcm0'],_ = load_xr_dataset(args,high_res = False)

    args = '--lsrp 0 --sigma 4 --filtering gaussian'.split()
    dss['gauss'],_ = load_xr_dataset(args,high_res = False)
    
    
    vars = 'u Su'.split()
    slc = slice(150,250)
    for var in vars:        
        dsvar = {x:y[var].isel(time = 0,lat = slc,lon = slc) for x,y in dss.items()}
        dsvar = {f'{x0}-{x1}':np.log10(np.abs(y0 - y1)) for x0,y0 in dsvar.items() for x1,y1 in dsvar.items()}
        plot_ds(dsvar,f'gcm_gaussian_{var}.png',ncols = 3)
        
    

if __name__ == '__main__':
    main()