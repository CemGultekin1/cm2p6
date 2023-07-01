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
from models.load import get_statedict

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
    mp = MurrayPlotter(sigma=4,nrows = 2,ncols = 2,figsize = (10,5),leftxmarg=0.035)

    path = os.path.join(target_folder,f'Sutemp_r2.png')
    import matplotlib
    ukwargs = dict(
        vmin = 0,
        vmax = 1,
        cbar_label = None,
        set_bad_alpha = 0.,
        colorbar = (0,2),
        grid_lines = {'alpha' : 0.4,'linewidth': 1.5},
        cmap = matplotlib.cm.magma
    )
    
    tkwargs = dict(
        vmin = 0,
        vmax = 1,
        cbar_label = None,
        set_bad_alpha = 0.,
        colorbar = (1,2),
        grid_lines = {'alpha' : 0.4,'linewidth': 1.5},
        cmap = matplotlib.cm.magma
    )
    
    mp.plot(0,0,ds0['Su_r2'],title = '(a) GZ21: R$^2_u$',**ukwargs)    
    mp.plot(0,1,ds1['Su_r2'],title = '(b) Global: R$^2_u$',**ukwargs)
    mp.plot(1,0,ds0['Stemp_r2'],title = '(c) GZ21: R$^2_T$',**tkwargs)
    mp.plot(1,1,ds1['Stemp_r2'],title = '(d) Global: R$^2_T$',**tkwargs)
    mp.save(path,transparent=True)
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