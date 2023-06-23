from data.paths import get_filename,get_high_res_grid_location
from intake import open_catalog
from run.train import Timer
from utils.slurm import flushed_print
import os
import xarray as xr

def hres_grid():
    cat = open_catalog("https://raw.githubusercontent.com/pangeo-data/pangeo-datastore/master/intake-catalogs/ocean/GFDL_CM2.6.yaml")
    ds = cat["GFDL_CM2_6_grid"].to_dask()
    # ds = xr.open_dataset(get_high_res_grid_location())
    coords = list(ds.coords)
    actual_coords = []
    for coord in coords:
        actual_coords.extend(ds[coord].dims)
    import numpy as np
    actual_coords = list(np.unique(np.array(actual_coords)))
    actual_vars = [c for c in coords if c not in actual_coords]
    data_vars = {}
    for var in actual_vars:
        data_vars[var] = (ds[var].dims,ds[var].values)
    _coords = {c: ds[c].values for c in actual_coords}
    import xarray as xr
    ds_ = xr.Dataset(data_vars = data_vars,coords= _coords)
    ds_.to_netcdf(get_high_res_grid_location(),mode = 'w')
    # ds_.to_netcdf(get_high_res_grid_location().replace('.nc','_.nc'),mode = 'w')
    return
def main():
    cat = open_catalog("https://raw.githubusercontent.com/pangeo-data/pangeo-datastore/master/intake-catalogs/ocean/GFDL_CM2.6.yaml")
    # ds = cat["GFDL_CM2_6_one_percent_ocean_3D"]
    # ds = ds.to_dask()
    # ds = cat["GFDL_CM2_6_one_percent_ocean_3D"].to_dask()
    # ds = cat["GFDL_CM2_6_one_percent_ocean_surface"].to_dask()
    ds = cat["GFDL_CM2_6_one_percent_ocean_3D"].to_dask()

    path = '/scratch/cg3306/climate/outputs/data/beneath_surface_1pct_co2_.zarr' #get_filename(1,0,True,'')
    print(path)

    keynames = list(ds.data_vars.keys())
    acceptable_keys = 'usurf vsurf surface_temp u v temp'.split()
    dropnames = [kn for kn in keynames if kn not in acceptable_keys]
    ds = ds.drop(dropnames)
    if 'st_ocean' in ds.coords:
        from data.coords import DEPTHS
        import numpy as np
        available_depths = ds.st_ocean.values
        select_depths = np.array(DEPTHS)
        available_depths = [ad for ad in available_depths if ad > 0]
        available_depths = [ad for ad in available_depths if np.amin(np.abs(select_depths - ad))<1e-2]
        ds = ds.sel(st_ocean = available_depths)
    timer = Timer()
    initial_time = 0
    if os.path.exists(path):
        eds = xr.open_zarr(path)
        initial_time = len(eds.time)
    for i in range(initial_time,len(ds.time)):
        timer.start('download')
        dsi = ds.isel(time = [i,])
        if i==0:
            dsi.to_zarr(path,mode='w')
        else:
            dsi.to_zarr(path, mode = 'a',append_dim = 'time')
        timer.end('download')
        flushed_print(i,'/',len(ds.time),'\t',timer)

if __name__=='__main__':
    main()