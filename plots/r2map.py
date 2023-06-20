import itertools
import os
import matplotlib.pyplot as plt
from plots.metrics import metrics_dataset
from constants.paths import JOBS, R2_PLOTS, EVALS
from utils.xarray import skipna_mean
import xarray as xr
from utils.arguments import options
from utils.slurm import flushed_print
from models.load import get_statedict, load_model
import numpy as np
def main():
    root = EVALS
    models = os.path.join(JOBS,'gz21.txt')
    target = R2_PLOTS
    file1 = open(models, 'r')
    
    lines = file1.readlines()
    lines = lines[51:56]
    suptitles = 'cheng_20230601_1 cheng_20230601_2 cheng_20230601_3 cheng_20230601_4 cheng_20230601_5'.split()
    
    
    file1.close()
    title_inc = ['sigma','domain','depth','latitude','lsrp','lossfun']
    title_name = ['sigma','train-domain','train-depth','latitude','lsrp','lossfun']
    for suptitle,line in zip(suptitles,lines):
        # line = '--lsrp 1 --depth 0 --sigma 4 --temperature True --lossfun MSE --latitude True --domain global --num_workers 16 --disp 50 --widths 5 128 64 32 32 32 32 32 6 --kernels 5 5 3 3 3 3 3 3 --minibatch 2'
        modelargs,modelid = options(line.split(),key = "model")
        _,_,_,modelid = get_statedict(line.split())

        # modelid = 'G-0'
        vals = [modelargs.__getattribute__(key) for key in title_inc]
        vals = [int(val) if isinstance(val,float) else val for val in vals]
        title = ',   '.join([f"{name}: {val}" for name,val in zip(title_name,vals)])
        snfile = os.path.join(root,modelid + '.nc')#'GZ21-subgrid-gz21-temp-global-trained-model.nc')
        # print(f'looking for {snfile}')
        if not os.path.exists(snfile):
            continue
        # print(line)
        sn = xr.open_dataset(snfile).sel(lat = slice(-85,85))#.isel(depth = [0],co2 = 0).drop(['co2'])
        msn = metrics_dataset(sn,dim = [])
        tmsn = metrics_dataset(sn,dim = ['lat','lon'])

        depthvals = msn.depth.values
        targetfolder = os.path.join(target,'20230602')#modelid)
        if not os.path.exists(targetfolder):
            os.makedirs(targetfolder)
        for i in range(len(depthvals)):
            s = msn.isel(depth = i)
            ts = tmsn.isel(depth = i)

            depthval = depthvals[i]

            # suptitle = f"{title}\ntest-depth: {depthval}"
            names = "Su Sv Stemp".split()
            unames = np.unique([n.split('_')[0] for n in list(s.data_vars)])
            names = [n for n in names if n in unames]
            ftypes = ['r2','mse','sc2','corr']
            
            nrows = len(names)
            ncols = len(ftypes)
            _names = np.empty((nrows,ncols),dtype = object)
            for ii,jj in itertools.product(range(nrows),range(ncols)):
                n = f"{names[ii]}_{ftypes[jj]}"
                _names[ii,jj] = n

            targetfile = os.path.join(targetfolder,suptitle.replace(' ','_') + '.png')#f'depth_{i}.png')

            fig,axs = plt.subplots(nrows,ncols,figsize = (ncols*6,nrows*5))
            for ir,ic in itertools.product(range(nrows),range(ncols)):
                name = _names[ir,ic]
                var = s[name]
                pkwargs = dict()
                if 'r2' in name or 'corr' in name:
                    pkwargs = dict(vmin = 0,vmax = 1)
                else:
                    var = np.log10(var)
                var.plot(ax = axs[ir,ic],**pkwargs)
                subtitle = f"{name}:{'{:.2e}'.format(ts[name].values[0])}"
                axs[ir,ic].set_title(subtitle,fontsize=24)
            fig.suptitle(suptitle,fontsize=24)
            fig.savefig(targetfile)
            flushed_print(targetfile)
            plt.close()



if __name__=='__main__':
    main()