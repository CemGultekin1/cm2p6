import os
from models.load import get_statedict
from plots.metrics_ import metrics_dataset
from constants.paths import EVALS
from plots.murray import MurrayPlotter
import xarray as xr
from utils.slurm import read_args
import os
from plots.metrics_ import metrics_dataset
from constants.paths import EVALS
import xarray as xr
from data.load import load_xr_dataset
from data.coords import DEPTHS,REGIONS
import numpy as np
import matplotlib
def load_dataset(sigma:int,depth:int):
    depth = DEPTHS[depth]
    ds,_ = load_xr_dataset(f'--sigma {np.maximum(sigma,4)} --depth {depth} --lsrp 1 --temperature True'.split(),high_res= sigma == 1)
    return ds.isel(time = 0)
def main():    
    sigma = 1
    ds = load_dataset(sigma,0)
    target_folder = 'paper_images/four_regions'
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    mp = MurrayPlotter(sigma=(sigma,),\
        nrows = 1,ncols = 4,figsize = (10,2.5),\
        interxmarg=0.02,leftxmarg = 0.05,ymarg = 0.04)
    path = os.path.join(target_folder,f'snapshot.png')
    
    kwargs = dict(
        vmin = (-3,-3,-3,-3),
        vmax = (3,3,3,3),
        cbar_label = ('$m/s$','$m/s$','$m/s$','$m/s$','$m/s$')
    )

    four_regions = REGIONS['four_regions']
    for irow,coord in enumerate(four_regions):
        coords = dict(
            lon = slice(*coord[2:]),lat = slice(*coord[:2])        
        )
        u = ds.u
        u = u.rename(dict(ulat = 'lat',ulon = 'lon'))
        colorbar = (0,1)
        kwargs_ = {key:val[irow] for key,val in kwargs.items()}
        mp.plot(0,irow,u,title = f'region #{irow + 1}',coord_sel=coords,\
            sigma = sigma,projection_flag=False,colorbar =colorbar,\
            cmap = matplotlib.cm.bwr,**kwargs_)
    mp.save(path,transparent=False)


if __name__ == '__main__':
    main()