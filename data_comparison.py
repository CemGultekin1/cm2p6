from data.arthur_datagen import eddy_forcing,spatial_filter_dataset
from data.high_res_dataset import HighResCm2p6
from utils.paths import FINE_CM2P6_PATH,TEMPORARY_DATA
from utils.xarray import plot_ds,fromtorchdict
from data.load import load_grid
import xarray as xr
import os

import numpy as np

def just_filtering(u_v_dataset, grid_data, scale_filter):
    return spatial_filter_dataset(u_v_dataset, grid_data, scale_filter)

def main():
    path = FINE_CM2P6_PATH(True,False)
    
    ds = xr.open_zarr(path).isel(time = [0,])
    sigma = 4
    # ds = ds.drop('surface_temp').drop('xt_ocean yt_ocean'.split())
    grid_data = load_grid(ds.copy())
    isel_dict = {
        v + key+'_ocean':slice(1500,2500) for key in 'u t'.split() for v in 'x y'.split()
    }
    ds = ds.isel(**isel_dict)
    grid_data = grid_data.isel(**isel_dict)
    forces = eddy_forcing(ds.drop('surface_temp'),grid_data,sigma)
    
    
    rename = {'yu_ocean':'lat','xu_ocean':'lon',\
                'usurf':'u','vsurf':'v'}#'S_x':'Su','S_y':'Sv',

    rename1 = {'yu_ocean':'ulat','xu_ocean':'ulon',\
                'yt_ocean':'tlat','xt_ocean':'tlon',\
                'usurf':'u','vsurf':'v','surface_temp':'temp'}
    forces = forces.rename(
        rename
    ).drop('xt_ocean yt_ocean'.split()).u

    # path1 = os.path.join(TEMPORARY_DATA,'arthur_data.nc')
    # forces.to_netcdf(path1)

    filtered = just_filtering(ds.drop('surface_temp').fillna(0),grid_data,(sigma/2,sigma/2)).rename(**rename).drop('xt_ocean yt_ocean'.split())
    ds = grid_data.rename(rename1)
    ds['depth'] = [0]
    hrcm = HighResCm2p6(ds.fillna(0),sigma,filtering = 'gaussian')
    
    ds1 = ds.rename({'ulat':'lat','ulon':'lon'})

    _,(cu,_) = hrcm.ugrid_subgrid_forcing(
        dict(u = ds1.u,v = ds1.v),'u v'.split(),'Su Sv'.split()
    )
    cu = cu['u'].drop('depth')

    coarsened =filtered.coarsen({'lon': int(sigma),
                                    'lat': int(sigma)},
                                    boundary='trim').mean().u
    coarsened,forces,cu = [x.isel(time = 0) for x in [coarsened,forces,cu]]
    
    data_vars,coords = hrcm[0]
    x = xr.Dataset(data_vars = data_vars,coords = coords)
    x = x.isel(time = 0,depth = 0).drop('time depth'.split()).u
    plot_ds(
        {'arthur_coarsened':coarsened,
         'arthur_coarsened1':forces,
        'my_coarsened':cu,
        'my_coarsened1':x,
        'err0': np.log10(np.abs(coarsened - cu)),
        'err1': np.log10(np.abs(coarsened - forces)),
        'err2': np.log10(np.abs(x - cu))},'arthur_forces.png',ncols = 3
    )
    
if __name__ == '__main__':
    main()