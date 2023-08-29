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
    
    
    
    target_folder = 'paper_images/r2maps_linear2'
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    ctrac = ColorbarTracker(target_folder)   
    filtering = 'gcm'
    sigma = 4
    
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
    matplotlib.rcParams.update({'font.size': 14})
    
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
            uval = u.values
            uval = np.power(uval,0.6)
            u.data = uval
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