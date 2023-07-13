import os
from models.load import get_statedict
from plots.metrics_ import metrics_dataset
from constants.paths import EVALS
from plots.murray import MurrayPlotter, MurrayWithSubplots
import xarray as xr
from utils.slurm import read_args
import os
from plots.metrics_ import metrics_dataset
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
    sigmas = [4,8,12,16]
    dss = {sigma:load_dataset(sigma,0) for sigma in sigmas}
    target_folder = 'paper_images/data_view'
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    path = os.path.join(target_folder,f'snapshot.png')
    
    kwargs = dict(
        vmin = (-1,22,-1e-6,-1e-6),
        vmax = (1,30.5,1e-6,1e-6),
        
    )
    cbarkwargs = dict(
        cbar_label = ('$m/s$','Celsius ($^{\circ}C$)','$m^2/s^4$','$m^2/s^4$')
    )
    name2title = dict(
        u = 'u',
        temp = 'T',
        Su = 'S$_u$',
        Stemp = 'S$_T$',
    )
    for irow,key in enumerate('u temp Su Stemp'.split()):
    
        coords = dict(
            lat = slice(-10,15),lon = slice(100-360,130-360)        
        )
        for icol,(sigma,ds) in enumerate(dss.items()):
            if key not in ds.data_vars:
                u = ds.u*0
            else:
                u = ds[key]            
            if sigma== 1:
                if 'ulat' in u.coords:
                    u = u.rename(dict(ulat = 'lat',ulon = 'lon'))
                elif 'tlat' in u.coords:
                    u = u.rename(dict(tlat = 'lat',tlon = 'lon'))
            
            matplotlib.rcParams.update({'font.size': 14})
            # if colorbar_flag:
            #     mp = MurrayWithSubplots(1,2,xmargs = (0.08,0.025,0.08),ymargs = (0.05,0.,0.05),figsize = (12,8),sizes =((1,),(10,2)))
            # else:
            mp = MurrayWithSubplots(1,1,xmargs = (0.1,0.,0.05),ymargs = (0.07,0.,0.07),figsize = (5,4),)#sizes =((1,),(10,1)))
            # colorbar = None if icol == 3 else (0,1)
            kwargs_ = {key:val[irow] for key,val in kwargs.items()}
            cbarkwargs_ =  {key:val[irow] for key,val in cbarkwargs.items()}
            _,ax,cs = mp.plot(0,0,u,title = None,coord_sel=coords,\
                sigma = sigma,projection_flag=False,\
                cmap = matplotlib.cm.bwr,**kwargs_)
            # print(f'dims = {dims}')
            
                
                
            path1 = path.replace('.png',f'{name2title[key]}-{sigma}.png').replace('$','')
            print(path1)
            mp.save(path1,transparent=False)
            
            
            colorbar_flag =  icol == 3
            if colorbar_flag:
                mp = MurrayWithSubplots(1,1,xmargs = (0.05,0.,0.5),ymargs = (0.05,0.,0.05),figsize = (1,4),)
                mp.plot_colorbar(0,0,cs,**cbarkwargs_)
                path1 = path.replace('.png',f'{name2title[key]}-{sigma}-colorbar.png').replace('$','')
                print(path1)
                mp.save(path1,transparent=False)
        # break
if __name__ == '__main__':
    main()