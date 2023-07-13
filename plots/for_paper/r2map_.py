import itertools
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
from models.load import get_statedict

def load_r2map(linenum:int):
    args = read_args(linenum,filename = 'offline_sweep2.txt')
    _,_,_,modelid = get_statedict(args)
    path =  os.path.join(EVALS,modelid + '.nc')
    assert os.path.exists(path)
    ds = xr.open_dataset(path).isel(depth = 0,co2 = 0)
    return metrics_dataset(ds,dim = [])
def load_lsrp_r2map():
    args = '--model lsrp:0 --sigma 4'.split()
    _,_,_,modelid = get_statedict(args)
    path =  os.path.join(EVALS,modelid + '.nc')
    assert os.path.exists(path)
    ds = xr.open_dataset(path).isel(depth = 0,co2 = 0)
    return metrics_dataset(ds,dim = [])
def main():
    ds0 = load_r2map(19)#3)
    ds1 = load_r2map(20)#4)
    lsrp =  load_lsrp_r2map()

    data = {
        'four_regions' : ds0,
        'global' : ds1,
        'linear': lsrp
    }
    
    
    for key,ds in data.items():
        if 'filtering' not in ds.coords:
            continue
        try:
            data[key] = ds.sel(filtering = 'gcm')
        except:
            gcmnum = sum([ord(a) for a in 'gcm'])
            data[key] = ds.sel(filtering = gcmnum)
            
            
    from utils.xarray import select_coords_by_extremum
    for key,val in data.items():
        for key1,val1 in data.items():
            if key1 == key:
                continue
            data[key]=select_coords_by_extremum(val,val1.coords,'lat lon'.split())
    target_folder = 'paper_images/r2maps'
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    sigma = 4

    import matplotlib
    matplotlib.rcParams.update({'font.size': 14})
    kwargs = dict(
        vmin = 0,
        vmax = 1,
        set_bad_alpha = 0.,
        projection_flag = True,
        sigma = sigma,
        cmap = matplotlib.cm.magma
    )
    forcings = 'Su Sv Stemp'.split()
    ftypes = 'r2 corr'.split()
    for forcing,ftype,source in itertools.product(forcings,ftypes,data.keys()):
        forcing = '_'.join([forcing,ftype])
        u = data[source][forcing]
        # u = u.sel(lon = slice(-175,170))
        mp = MurrayWithSubplots(1,1,xmargs = (0.08,0.,0.02),ymargs = (0.06,0.,0.),figsize = (8,3.5),)
        _,ax,cs = mp.plot(0,0,u,title = None,**kwargs) 
        filename = f'{source}-{forcing}.png'
        path1 = os.path.join(target_folder,filename)
        print(path1)
        mp.save(path1,transparent=False)
        
        mp = MurrayWithSubplots(1,1,xmargs = (0.,0.,0.55),ymargs = (0.06,0.,0.),figsize = (0.75,3.5),)
        mp.plot_colorbar(0,0,cs,)
        filename = f'{filename.replace(".png","")}-colorbar.png'
        path1 = os.path.join(target_folder,filename)
        print(path1)
        mp.save(path1,transparent=False)
        return

if __name__ == '__main__':
    main()