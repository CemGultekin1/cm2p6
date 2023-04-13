import itertools
import os
import matplotlib.pyplot as plt
from data.load import pass_geo_grid
from models.load import load_model

from constants.paths import JOBS, VIEW_PLOTS, VIEWS
import xarray as xr
from utils.arguments import options
import numpy as np
from data.coords import REGIONS

def main():
    root = VIEWS
    target = VIEW_PLOTS
    from utils.slurm import read_args
    args = read_args(2,)
    from utils.arguments import replace_params
    args = replace_params(args,'mode','eval','num_workers','1')
    lines = [' '.join(args)]

    title_inc = ['sigma','domain','depth','latitude','lsrp']
    title_nam = ['sigma','train-domain','train-depth','latitude','lsrp']
    subplotkwargs = dict()
    plotkwargs = lambda a : dict()
    for line in lines:
        if line == 'lsrp':
            modelid = 'lsrp'
            title = 'LSRP'
        else:
            modelargs,_ = options(line.split(),key = "model")
            modelid,_,_,_,_,_,_,_=load_model(args)
            print(modelid)
            title = ',   '.join([f"{name}: {modelargs.__getattribute__(key)}" for key,name in zip(title_inc,title_nam)])
        
        ##for old models
        # modelid = line
        # title = line

        snfile = os.path.join(root,modelid + '.nc')
        
        if not os.path.exists(snfile):
            continue
        sn = xr.open_dataset(snfile)
        # runargs,_ = options(line.split(),key = "run")
        sn = pass_geo_grid(sn,4)#runargs.sigma)
 
        depthvals = sn.depth.values
        timevals = sn.time.values
        targetfolder = os.path.join(target,modelid)
        if not os.path.exists(targetfolder):
            os.makedirs(targetfolder)
        for i,(time,depth) in enumerate(itertools.product(timevals,depthvals)):
            s = sn.sel(time = time,depth = depth)
            s = s.drop('time').drop('depth')
            title_ = f"{title}\ntest-depth: {depth},    time: {time}"
            names = 'u v T'.split() + 'Su Sv ST'.split() + 'Su_true Sv_true ST_true'.split() + 'Su_err Sv_err ST_err'.split() + 'Su_std Sv_std ST_std'.split()
            ncol = 3
            names = [n if n in list(s.data_vars) else None for n in names ]
            names = np.array(names)
            names = names.reshape([-1,ncol])
            targetfile = os.path.join(targetfolder,f'snapshot_{i}.png')
            nrow = names.shape[0]
            plt.figure(figsize = (40,25))
            
            sel = dict(lat = slice(REGIONS['custom'][0],REGIONS['custom'][1]),lon = slice(REGIONS['custom'][2],REGIONS['custom'][3]))
            print(sel)
            for ir,ic in itertools.product(range(nrow),range(ncol)):

                ax = plt.subplot(nrow,ncol,ir*ncol + ic + 1,**subplotkwargs)
                if names[ir,ic] is None:
                    continue
                kwargs = plotkwargs(False)
                s[names[ir,ic]].sel(**sel).plot(ax = ax,**kwargs)                
                ax.set_title(names[ir,ic],fontsize=24)

            print(title_)
            plt.suptitle(title_,fontsize=24)
            plt.savefig(targetfile)
            plt.close()
            print(targetfile)
            



if __name__=='__main__':
    main()