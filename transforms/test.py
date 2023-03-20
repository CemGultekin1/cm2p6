
from data.load import load_filter_weights, load_xr_dataset
from transforms.gcm_compression_spatially_variant import FilterWeightSpaceVariantCompression,Variant2DMatmult
import numpy as np
from utils.xarray import plot_ds
import xarray as xr 
from transforms.grids import get_grid_vars
from utils.arguments import options
import matplotlib.pyplot as plt

def main():
    sigma = 4
    args = f'--sigma {sigma} --filtering gcm --lsrp 1 --mode data'.split()
    dfw = load_filter_weights(args,utgrid='u').load()
    coords = dict(lat = (0,3),lon = (-140,-137))
    crs_ic = {
        dim: [np.argmin(np.abs(dfw[dim].values - lims[j])) for j in range(2)] for dim,lims in coords.items()
    }
    
    
    datargs,_ = options(args,key = 'data')
    ds,_ = load_xr_dataset(args,high_res = True)
    fine_isel = {
        dim: slice(lims[0]*sigma,lims[1]*sigma) for dim,lims in crs_ic.items()
    }
    cors_isel = {
        dim: slice(lims[0],lims[1]) for dim,lims in crs_ic.items()
    }
    
    
    
    ugrid,_ = get_grid_vars(ds.isel(time = 0))
    
    
    ugrid = ugrid.isel(**fine_isel)
    

    ds = FilterWeightSpaceVariantCompression(sigma, dfw.filters).get_separable_components()
    print(ds)
    return 

    rank = 512
    v2dm = Variant2DMatmult(sigma,ugrid,dfw,rank = rank)

    u = ds.u.isel(time = 0).rename(
        {'u'+dim : dim for dim in coords}
    )
    u = u.isel(**fine_isel)
    ubar = v2dm(u,inverse = False,separated= False)
    plot_ds(dict(ubar = ubar,),f'ubar_.png',ncols = 1)
    return
    for i in range(rank):
        ubar = v2dm(u,inverse = False,separated= False,special = i)
        plot_ds(dict(ubar = ubar,),f'ubar_{i}.png',ncols = 1)

if __name__ == '__main__':
    main()