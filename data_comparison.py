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
                'usurf':'u','vsurf':'v','S_x':'Su','S_y':'Sv'}

    rename1 = {'yu_ocean':'ulat','xu_ocean':'ulon',\
                'yt_ocean':'tlat','xt_ocean':'tlon',\
                'usurf':'u','vsurf':'v','surface_temp':'temp'}
    forces = forces.rename(
        rename
    ).drop('xt_ocean yt_ocean'.split()).isel(time = 0)

    # path1 = os.path.join(TEMPORARY_DATA,'arthur_data.nc')
    # forces.to_netcdf(path1)


    ds = grid_data.rename(rename1)
    ds['depth'] = [0]
    hrcm = HighResCm2p6(ds,sigma,filtering = 'gaussian')
    
    data_vars,coords = hrcm[0]
    x = xr.Dataset(data_vars = data_vars,coords = coords)
    x = x.isel(time = 0,depth = 0).drop('time depth'.split()).drop('Stemp temp'.split())
    print(x)
    print(forces)
    x1 = x.rename(
        {key:'cem'+key for key in x.data_vars.keys()}
    )
    f = xr.merge([x1,forces])
    plot_ds(f,'arthur_forces.png',ncols = 3,)
    err = np.log10(np.abs(x - forces))
    plot_ds(err,'arthur_forces_err.png',ncols = 3,)
if __name__ == '__main__':
    main()