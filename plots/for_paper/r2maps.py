import os
from data.load import load_xr_dataset
from data.paths import get_high_res_grid_location
from models.load import get_statedict
from plots.metrics import metrics_dataset
from constants.paths import EVALS
from plots.murray import MurrayPlotter
from transforms.grids import fix_grid
import xarray as xr
import numpy as np
from utils.slurm import read_args
import cartopy.crs as ccrs
import cartopy as cart
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import os
import matplotlib
import matplotlib.pyplot as plt
from plots.metrics import metrics_dataset
from constants.paths import EVALS
import xarray as xr
from models.load import get_statedict
import numpy as np
import itertools

def load_r2map(linenum:int):
    args = read_args(linenum,filename = 'offline_sweep.txt')
    _,_,_,modelid = get_statedict(args)
    path =  os.path.join(EVALS,modelid + '.nc')
    assert os.path.exists(path)
    ds = xr.open_dataset(path).isel(depth = 0,co2 = 0)
    return metrics_dataset(ds,dim = [])
def main():
    ds0 = load_r2map(49)
    ds1 = load_r2map(52)
    
    target_folder = 'paper_images/r2maps'
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    mp = MurrayPlotter(sigma=4,nrows = 2,ncols = 2,figsize = (10,5))

    path = os.path.join(target_folder,f'Sutemp_r2.png')
    mp.plot(0,0,ds0['Su_r2'],title = '$R^2_u$')
    mp.plot(1,0,ds0['Stemp_r2'],title = '$R^2_{T}$')
    mp.plot(0,1,ds1['Su_r2'],title = '$R^2_u$')
    mp.plot(1,1,ds1['Stemp_r2'],title = '$R^2_{T}$')
    mp.save(path)
    # for svar,rc in itertools.product('Su Sv Stemp'.split(),'r2 corr'.split()):
    #     keyname = f'{svar}_{rc}'
    #     if keyname not in ds0.data_vars:
    #         continue
        
    #     mp.plot(ds0[keyname],path,vmin = 0,vmax = 1)
    #     # print(path)
    #     # return
    #     path = os.path.join(target_folder,f'fglobal_{svar}_{rc}.png')
    #     mp.plot(ds1[keyname],path,vmin = 0,vmax = 1)
    

if __name__ == '__main__':
    main()