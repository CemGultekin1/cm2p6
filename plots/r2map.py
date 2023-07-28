import itertools
import os
import matplotlib.pyplot as plt
from plots.metrics_ import metrics_dataset
from constants.paths import JOBS, R2_PLOTS, EVALS
from utils.xarray_oper import skipna_mean
import xarray as xr
from utils.arguments import options
from utils.slurm import flushed_print
from models.load import get_statedict, load_model
import numpy as np
def main():
    root = EVALS
    models = os.path.join(JOBS,'offline_sweep.txt')
    # models = os.path.join(JOBS,'gz21.txt')
    target = R2_PLOTS
    file1 = open(models, 'r')
    
    lines = file1.readlines()
    # lines = lines[17:18]
    lines = lines[48:49]
    
    
    
    file1.close()
    title_inc = ['sigma','domain','depth','latitude','lsrp','lossfun']
    title_name = ['sigma','train-domain','train-depth','latitude','lsrp','lossfun']
    for line in lines:
        modelargs,modelid = options(line.split(),key = "model")

        # modelid = 'G-0'
        vals = [modelargs.__getattribute__(key) for key in title_inc]
        vals = [int(val) if isinstance(val,float) else val for val in vals]
        title = ',   '.join([f"{name}: {val}" for name,val in zip(title_name,vals)])
        # modelid = 'GZ21--scratch-cg3306-climate-best-trained-model'
        snfile = os.path.join(root,modelid + '.nc')#'GZ21-subgrid-gz21-temp-global-trained-model.nc')
        
        # print(f'looking for {snfile}')
        if not os.path.exists(snfile):
            continue
        
        # if modelargs.lossfun == 'MSE':
        #     continue
        print(line)
        
        sn = xr.open_dataset(snfile).sel(lat = slice(-85,85),)#.isel(depth = [0],co2 = 0).drop(['co2'])
        sn = sn.sel(filtering = 'gaussian',co2 = 0,depth = 0).drop('filtering co2 depth'.split())
        # sn = sn.isel(filtering = 1).drop('filtering')
        msn = metrics_dataset(sn,dim = [])
        tmsn = metrics_dataset(sn,dim = ['lat','lon'])
        phy_keys = {key:range(len(msn[key])) for key in tmsn.coords.keys()}
        targetfolder = os.path.join(target,'20230726')#modelid)
        if not os.path.exists(targetfolder):
            os.makedirs(targetfolder)
        for indices in itertools.product(*phy_keys.values()):
            select_keys = dict((key,i) for key,i in zip(phy_keys.keys(),indices))
            s = msn.isel(**select_keys)
            ts = tmsn.isel(**select_keys)

            suptitle = ' '.join([f'{key} = {ts.coords[key].values.item()}' for key in select_keys.keys()])
            # suptitle = f"{title}\ntest-depth: {depthval}"
            names = "Su Sv Stemp".split()
            unames = np.unique([n.split('_')[0] for n in list(s.data_vars)])
            names = [n for n in names if n in unames]
            ftypes = ['r2',]
            
            nrows = len(names)
            ncols = len(ftypes)
            _names = np.empty((nrows,ncols),dtype = object)
            for ii,jj in itertools.product(range(nrows),range(ncols)):
                n = f"{names[ii]}_{ftypes[jj]}"
                _names[ii,jj] = n
            filename = '_'.join([str(i) for i in indices])
            targetfile = os.path.join(targetfolder,modelid+ '_'+filename + '.png')#f'depth_{i}.png')

            
            for ir,ic in itertools.product(range(nrows),range(ncols)):
                fig,axs = plt.subplots(1,1,figsize = (12,10))
                name = _names[ir,ic]
                var = s[name]
                pkwargs = dict()
                if 'r2' in name or 'corr' in name:
                    pkwargs = dict(vmin = 0,vmax = 1)
                else:
                    var = np.log10(var)
                ax = axs#[ir]
                var.plot(ax = ax,**pkwargs)
                subtitle = f"{name}:{'{:.2e}'.format(ts[name].values.item())}"
                ax.set_title(subtitle,fontsize=24)
                ax.set_ylabel(name)
                fig.suptitle(suptitle,fontsize=12)
                targetfile_ = targetfile.replace('.png',name + '.png')
                fig.savefig(targetfile_)
                flushed_print(targetfile_)
                plt.close()



if __name__=='__main__':
    main()