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
        nrows = 1,ncols = 1,figsize = (4.5,3),\
        interxmarg=0.03,leftxmarg = 0.12,ymarg = 0.08,colorbarxmarg=0.2,colorbarwidth=0.025)
    path = os.path.join(target_folder,f'snapshot.png')
    
    kwargs = dict(
        vmin = (-10,),
        vmax = (33,),
        cbar_label = ('Celsius ($^{\circ}C$)',)
    )

    four_regions = REGIONS['four_regions']
    # for irow,coord in enumerate(four_regions):
    #     coords = dict(
    #         lon = slice(*coord[2:]),lat = slice(*coord[:2])        
    #     )
    u = ds.temp
    u = u.rename(dict(tlat = 'lat',tlon = 'lon'))
    colorbar = (0,1)
    irow = 0
    kwargs_ = {key:val[irow] for key,val in kwargs.items()}
    coord_sel = dict(
        lat = slice(-60,60),lon = slice(-180,-10)
    )
    ax = mp.plot(0,0,u,title = '',\
        sigma = sigma,projection_flag=True,colorbar =colorbar,coord_sel=coord_sel,\
        cmap = matplotlib.cm.bwr,\
        set_bad_alpha=0.,\
        grid_lines={'alpha':0.5},\
        **kwargs_)
    ax.set_title('Surface Temperature')
    import matplotlib.patches as patches
    for coords in four_regions:
        ax.add_patch(
        patches.Rectangle(
            (coords[2],coords[0]),
            coords[3] - coords[2],
            coords[1] - coords[0],
            fill=True,color = 'black',edgecolor = None,alpha = 0.5
        ) ) 
    mp.save(path,transparent=True)


if __name__ == '__main__':
    main()