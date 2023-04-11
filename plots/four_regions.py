from data.coords import REGIONS
from utils.paths import COARSE_CM2P6_PATH
import matplotlib.pyplot as plt
import os
import xarray as xr
import numpy as np
import itertools
def main():

    path = os.path.join(COARSE_CM2P6_PATH,'coarse_4_surface_gaussian.zarr')
    ds = xr.open_zarr(path).isel(time = 0,depth = 0).drop('time depth'.split())
    
    nlat,nlon = 100,100
    for rowi,(lat0,lat1,lon0,lon1) in enumerate(REGIONS['four_regions']):
        ds1 = ds.sel(lat = slice(lat0,lat1),lon = slice(lon0,lon1))
        nlat_,nlon_ = len(ds1.lat),len(ds1.lon)
        nlat = np.minimum(nlat,nlat_)
        nlon = np.minimum(nlon,nlon_)
    fig,axs = plt.subplots(4,2,figsize = (10,20))
    plot_vars = 'u interior_wet_mask'.split()
    for rowi,(lat0,lat1,lon0,lon1) in enumerate(REGIONS['four_regions']):
        ds1 = ds.sel(lat = slice(lat0,lat1),lon = slice(lon0,lon1))
       

        for coli,pv in enumerate(plot_vars):
            ds1[pv].plot(ax = axs[rowi,coli],vmin = -1,vmax = 1,cmap = 'bwr')     
            axs[rowi,coli].set_xlabel('')
            axs[rowi,coli].set_ylabel('')
            axs[rowi,coli].set_title(f'region={rowi}, {pv}')
            x1,y1 = [0,5],[5,10]
            # x1,y1 = [lon0_,lon1_],[lat0_, lat0_]
            # x2,y2 = [lon0_,lon1_],[lat1_,lat1_]
            # x3,y3 = [lon0_,lon0_],[lat0_,lat1_]
            # x4,y4 = [lon1_,lon1_],[lat0_,lat1_]
            axs[rowi,coli].plot(x1,y1,color = 'black')#,y2,x2,y3,x3,y4,x4,
            
    fig.tight_layout()
    fig.savefig('four_regions.png')
if __name__ == '__main__':
    main()