from copy import deepcopy
import itertools
import logging
import os
from typing import List, Union
from metrics.geomean import WetMaskCollector
from models.load import get_statedict

from utils.arguments import options
from plots.metrics_ import metrics_dataset
from constants.paths import EVALS
from plots.murray import MurrayPlotter, MurrayWithSubplots
import xarray as xr
from utils.slurm import ArgsFinder, read_args
from utils.xarray_oper import drop_unused_coords, sel_available, select_coords_by_extremum
import os
from plots.metrics_ import metrics_dataset
from constants.paths import EVALS
import xarray as xr
from models.load import get_statedict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from utils.slurm import basic_config_logging
from linear.coarse_graining_inverter import CoarseGrainingInverter
class ColorbarTracker:
    def __init__(self,folder:str) -> None:
        self.root = folder
        self.saved = False
    def is_there_any(self,):
        files = os.listdir(self.root)
        files = [file for file in files if 'colorbar' in file]
        return bool(files)
    
def main():
    basic_config_logging()
    
    sigma = 16
    shp = (2700,3600)
    shp1 = tuple(x//sigma for x in shp)
    ds = xr.open_zarr(f'/scratch/cg3306/climate/outputs/data/linear_sgm_{sigma}_dpth_0.zarr')
    latc = 0
    lonc = -225
    clati = np.argmin(np.abs(ds.lat.values - latc))
    cloni = np.argmin(np.abs(ds.lon.values - lonc))
    # ds = ds.isel(lat = slice(clati - 10,clati + 10),lon = slice(cloni -10,cloni + 10))


    x = np.zeros(shp1)
    x[clati,cloni] = 1
    x = x.flatten()
    
    
    
    # ds.isel(time = 0).u.plot()
    # plt.savefig('dummy.png')
    
    matplotlib.rcParams.update({'font.size': 17})
    fds = xr.open_zarr('/scratch/as15415/Data/CM26_Surface_UVT.zarr')
    uval = fds.isel(time = 0).fillna(0).usurf
    uval = uval.rename({'xu_ocean':'lon','yu_ocean':'lat'})
    
    cgi = CoarseGrainingInverter(depth = 0, sigma = sigma)
    cgi.load_parts()
    logging.info(f'cgi.mat.shape = {cgi.mat.shape}')
    logging.info(f'x.shape = {x.shape}')
    
    
    
    f = x @ cgi.mat
    f = f.reshape(shp)
    
    
    
    ind = np.unravel_index(np.argmax(f, axis=None), f.shape)
    c0,c1 = f.shape[0]//2,f.shape[1]//2
    f = f/np.amax(f)
    f = np.where(uval.values == 0,np.nan,f)
    uval.attrs = {}
    uval.data = f
    uval = uval.roll(lat = c0 - ind[0],lon = c1 - ind[1])

    hspan = 30
    slc0,slc1 = tuple(slice( c - hspan,c + hspan+1) for c in [c0,c1])    
    uval = uval.isel(lat = slc0,lon = slc1)
    cmap = matplotlib.cm.bwr
    cmap.set_bad('black',1.)

    kwargs = dict(
        vmin = -1,
        vmax = 1,
        cmap = cmap,
        cbar_kwargs={'label': ""}
    )
    fig,ax = plt.subplots(figsize = (10,8))
    
    
    uval.plot(ax = ax,**kwargs)    
    plt.title('')
    fig.subplots_adjust(
            top=0.981,
            bottom=0.08,
            left=0.08,
            right=0.99,
        )
    plt.xlabel('lon')
    plt.ylabel('lat')
    plt.savefig('forward_filter.png')
    plt.close()
    # return
    m = 0
    lons = [k - m for k in range(2*m+1)]
    kwargs = dict(
        # vmin = -1.5,
        # vmax = 1.5,
        cmap = cmap,
        cbar_kwargs={'label': ""}
    )
    for i,lon in enumerate(lons):
        lat = 0
        xf = np.zeros(shp)
        xf[clati*sigma + lat,cloni*sigma + lon] = 1
        xf = xf.flatten()
        b = (xf @ cgi.mat.T)@cgi.qinvmat
        b = b.reshape(shp1)
        
        
        c0,c1 = b.shape[0]//2,b.shape[1]//2
        
        b = np.roll(b,-1,axis = 0)
        cuval = ds.isel(time = 0,depth = 0).u.fillna(0)
        b = np.where(cuval.values == 0, np.nan,b)
        cuval.data = b
        
        cuval = cuval.roll(lat = c0 - clati,lon = c1 - cloni)
        
        hspan = 7
        slc0,slc1 = tuple(slice( c - hspan,c + hspan+1) for c in [c0,c1])    
        cuval = cuval.isel(lat = slc0,lon = slc1)
        
        

        fig,ax = plt.subplots(figsize=(10,8))
        cuval.plot(ax = ax,add_colorbar = True,**kwargs)
        plt.title('')
        strlat,strlon = tuple( f'p{l}' if l>=0 else f'n{abs(l)}' for l in [lat,lon])
        plt.savefig(f'inverse_filter_lon_{strlon}.png')
        plt.close()
        fig.subplots_adjust(
            top=0.991,
            bottom=0.049,
            left=0.022,
            right=0.991,
        )
    return
   
    
    target_folder = 'paper_images/r2maps_linear'
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    ctrac = ColorbarTracker(target_folder)   
    filtering = 'gcm'
    sigma = 8
    
    args = f'--model lsrp:0 --filtering {filtering} --sigma {sigma}'.split()
    runargs,lsrpid = options(args,key = 'model')
    path =os.path.join(EVALS,lsrpid + '_.nc')
    if not os.path.exists(path):
        return False
    ds = xr.open_dataset(path)
    
    lons = ds.lon.values
    diff = np.abs(lons + 180)
    diff[lons < -180] = np.inf
    loni = np.argmin(diff)
    ds = ds.roll(lon = -loni,roll_coords = True)
    lons = ds.lon.values
    lons = (lons+180)%360 - 180
    ds['lon'] = lons

    # logging.info(ds.lon.values[[0,-1]])
    # return
    ds = metrics_dataset(ds,dim = [])
    ds = ds.sel(filtering = 'gcm')
    keepdims = [f'{key}_r2' for key in 'Su Sv Stemp'.split()]
    dropdims = [key for key in ds.data_vars if key not in keepdims]
    data = ds.drop(dropdims)
    # logging.info(data)
    # return
    
    
    kwargs = dict(
        vmin = 0,
        vmax = 1,
        set_bad_alpha = 0.,
        projection_flag = True,
        sigma = sigma,
        cmap = matplotlib.cm.magma,
        shading = 'gouraud'
    )
    fltr0 = filtering
    fltr1 = filtering
    for co2,depth in itertools.product(range(2),range(len(ds.depth))):
        if co2 == 0:
            co2str = '0p00'
        else:
            co2str = '0p01'            
        
        for source in data.data_vars.keys():
            u = data[source].isel(co2 = co2,depth = depth )
            dp = int(u.depth.values.item())
            mp = MurrayWithSubplots(1,1,xmargs = (0.1,0.,0.02),ymargs = (0.06,0.,0.),figsize = (8,3.5),)
            _,ax,cs = mp.plot(0,0,u,title = None,**kwargs) 
            ax.set_facecolor('black')
            ftype = source.split('_')[0]
            fpng = f'linear_dpth_{dp}_sgm_{sigma}_c2_{co2str}_{ftype}.png'
            path1 = os.path.join(target_folder,fpng)
            logging.info(path1)
            mp.save(path1,transparent=False)        
            # return
        
            if not ctrac.is_there_any():
                mp = MurrayWithSubplots(1,1,xmargs = (0.,0.,0.55),ymargs = (0.06,0.,0.),figsize = (0.75,3.5),)
                mp.plot_colorbar(0,0,cs,)
                cbarfilename = f'colorbar.png'
                path1 = os.path.join(target_folder,cbarfilename)
                logging.info(path1)
                mp.save(path1,transparent=False)

if __name__ == '__main__':
    main()