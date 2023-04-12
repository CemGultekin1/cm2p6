import itertools
import os
import matplotlib.pyplot as plt
from models.load import load_model
from constants.paths import DISTRIBUTION_PLOTS,DISTS
import xarray as xr
from utils.arguments import options
from utils.slurm import flushed_print
import numpy as np

def main():
    root = DISTS
    target = DISTRIBUTION_PLOTS
    if not os.path.exists(target):
        os.makedirs(target)
    
    from utils.slurm import read_args
    from utils.arguments import replace_params
    for arg_index in range(5):
        args = read_args(arg_index + 1)
        args = replace_params(args,'mode','eval','num_workers','1','disp','25','minibatch','1')

        title_feats = 'scheduler min_precision'.split()
        
        lines = [' '.join(args)]
        for line in lines:
            modelid,_,_,_,_,_,_,runargs=load_model(line.split())
            
            snfile = os.path.join(root,modelid + '.nc')
            if not os.path.exists(snfile):
                continue
            sn = xr.open_dataset(snfile)

            names = "Su Sv".split()
            unames = np.unique([n.split('_')[0] for n in list(sn.data_vars)])
            names = [n for n in names if n in unames]

            
            nrows = 1
            ncols = len(names)
            _names = np.empty((nrows,ncols),dtype = object)
            for ii,jj in itertools.product(range(nrows),range(ncols)):
                n = names[jj].split('_')[0]
                _names[ii,jj] = n

            targetfile = os.path.join(target,f'{modelid}.png')
            fig,axs = plt.subplots(nrows,ncols,figsize = (ncols*6,nrows*5))
            for ir,ic in itertools.product(range(nrows),range(ncols)):
                name = _names[ir,ic]
                var = sn[names[ic]+'_density']
                var.plot(ax = axs[ic],)
                dim = var.dims[0]
                bns = var[dim].values
                dx = bns[1] - bns[0]
                gauss_density = np.exp( - bns**2/2 )/np.sqrt(2*np.pi)*dx
                sn['gauss_density'] = ([dim],gauss_density)
                sn['gauss_density'].plot(ax = axs[ic],)
                
                subtitle = name
                axs[ic].set_title(subtitle,fontsize=15)
                axs[ic].set_ylabel('density')
            if runargs.gz21:
                suptitle = 'GZ21'
            else:
                suptitle = ', '.join([f'{feat} = {runargs.__dict__[feat]}' for feat in title_feats])
            fig.suptitle(suptitle,fontsize=15)
            fig.savefig(targetfile)
            flushed_print(targetfile)
            plt.close()



if __name__=='__main__':
    main()