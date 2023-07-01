import os
from models.load import get_statedict
from plots.metrics import metrics_dataset
from constants.paths import EVALS
from plots.murray import MurrayPlotter
import xarray as xr
from utils.slurm import read_args
import os
from plots.metrics import metrics_dataset
from constants.paths import EVALS
import xarray as xr
from data.load import load_xr_dataset
from data.coords import DEPTHS
import numpy as np
import matplotlib
def load_dataset(sigma:int,depth:int):
    depth = DEPTHS[depth]
    ds,_ = load_xr_dataset(f'--sigma {np.maximum(sigma,4)} --depth {depth} --lsrp 1 --temperature True'.split(),high_res= sigma == 1)
    return ds.isel(time = 0)
def main():    
    dss = {sigma:load_dataset(sigma,0) for sigma in [1,4,8,12]}
    target_folder = 'paper_images/data_view'
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    mp = MurrayPlotter(sigma=(1,4,8,12),\
        nrows = 4,ncols = len(dss),figsize = (10,8),\
        interxmarg=0.005,ymarg = 0.035)
    path = os.path.join(target_folder,f'snapshot.png')
    
    kwargs = dict(
        vmin = (-1,22,-1e-6,-1e-6),
        vmax = (1,30.5,1e-6,1e-6),
        cbar_label = ('$m/s$','Celsius ($^{\circ}C$)','$m^2/s^4$','$m^2/s^4$')
    )
    name2title = dict(
        u = 'u',
        temp = 'T',
        Su = 'S$_u$',
        Stemp = 'S$_T$',
    )
    for irow,(sigma,ds) in enumerate(dss.items()):
        coords = dict(
            lat = slice(-10,15),lon = slice(100-360,130-360)        
        )
        for icol,key in enumerate('u temp Su Stemp'.split()):
            if key not in ds.data_vars:
                u = ds.u*0
            else:
                u = ds[key]            
            if sigma== 1:
                if 'ulat' in u.coords:
                    u = u.rename(dict(ulat = 'lat',ulon = 'lon'))
                elif 'tlat' in u.coords:
                    u = u.rename(dict(tlat = 'lat',tlon = 'lon'))
            colorbar = (icol,4)
            kwargs_ = {key:val[icol] for key,val in kwargs.items()}
            mp.plot(icol,irow,u,title = f'{name2title[key]}:\u03C3 = {sigma}',coord_sel=coords,\
                sigma = sigma,projection_flag=False,colorbar =colorbar,\
                cmap = matplotlib.cm.bwr,**kwargs_)
    mp.save(path)
if __name__ == '__main__':
    main()