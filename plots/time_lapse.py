import itertools
import os
import matplotlib.pyplot as plt
from utils.paths import JOBS, TIME_LAPSE_PLOTS, TIME_LAPSE
import xarray as xr
from utils.arguments import options
from utils.slurm import flushed_print
import numpy as np
def main():
    root = TIME_LAPSE
    target = TIME_LAPSE_PLOTS  
    #lines = ['G-0']
    
    models = os.path.join(JOBS,'trainjob.txt')
    file1 = open(models, 'r')
    lines = file1.readlines()
    file1.close()

    lines = [lines[i] for i in [0]]
    # lines = ['--lsrp 0 --depth 0 --sigma 4 --filtering gcm --temperature True --latitude False --domain global --num_workers 16 --disp 50 --batchnorm 1 1 1 1 1 1 1 0 --lossfun heteroscedastic --widths 3 128 64 32 32 32 32 32 6 --kernels 5 5 3 3 3 3 3 3 --minibatch 4']

    title_inc = ['sigma','domain','depth','interior','filtering','lossfun']
    title_name = ['sigma','train-domain','train-depth','interior','filtering','lossfun']

    for j,line in enumerate(lines):


        args = line.split()
    
        
        _,modelid = options(args,key = "model")
        runargs,_ = options(args,key = "run")
        vals = [runargs.__getattribute__(key) for key in title_inc]
        vals = [int(val) if isinstance(val,float) else val for val in vals]
        title = [f"{name}: {val}" for name,val in zip(title_name,vals)]
        title = [st + ('\n' if i%3 == 2 else ',   ') for i,st in enumerate(title)]
        title = ''.join(title)
        # if j <3:
        #     title = title.replace('global','four_regions')
            
        snfile = os.path.join(root,modelid + '.nc')
        if not os.path.exists(snfile):
            continue
       
        print(line)
        sn = xr.open_dataset(snfile).isel(co2 = 0,depth = 0)
        print(sn)
        evalfile = snfile.replace('/time_lapse','/evals')
        evalexists =  os.path.exists(evalfile)
        if evalexists:
            evsn = xr.open_dataset(evalfile).isel(co2 = 0,depth = 0)


        if not os.path.exists(target):
            os.makedirs(target)
        s = sn

        lats,lons = s.lat.values, s.lon.values
        
        names = "Su Sv Stemp".split()
        # names = names[:1]
        unames = np.unique([n.split('_')[1] for n in list(s.data_vars) if n not in 'lat lon'.split()])
        names = [n for n in names if n in unames]
        
        nrows = len(lats)
        ncols = len(names)
        targetfile = os.path.join(target,f'{modelid}.png')

        fig,axs = plt.subplots(nrows,ncols,figsize = (ncols*12,nrows*5))
        # print(ncols,nrows)
        interiority = [False, True, True, True, True, False, True]
        for i,(ir,ic) in enumerate(itertools.product(range(nrows),range(ncols))):
            var = s.isel(coord_id = ir,time = range(4,304))
            ax = axs[ir,ic]
            # title_ = f"{title}\n coord: ({lat},{lon})"
            name = names[ic]
            if evalexists:
                t2 = evsn[name+'_true_mom2']
                p2 = evsn[name+'_pred_mom2']
                c2 = evsn[name+'_cross']
                r2 = 1 - (t2 - 2*c2 + p2)/t2
                r2 = r2.sel(lat = var.lat.values.item(),lon = var.lon.values.item(),method = 'nearest').values.item()

            

            true_val = var['true_'+name].values.reshape([-1])
            pred_val = var['pred_'+name+'_mean'].values.reshape([-1])
            if runargs.lossfun == 'heteroscedastic':
                std_val = var['pred_'+name+'_std'].values.reshape([-1])
                pred1 = 1.96*std_val + pred_val
                pred_1 = -1.96*std_val + pred_val

            # for key,val in dict(true_val = true_val,pred_val = pred_val,std_val = std_val).items():
            #     print(key,'is any nan values:\t ',np.any(np.isnan(val)))

            ax.plot(true_val,color = 'tab:blue', label = 'true',linewidth = 2)
            ax.plot(pred_val,color = 'tab:orange', label = 'mean',linewidth = 2)
            if runargs.lossfun == 'heteroscedastic':
                ax.plot(pred1,color = 'tab:green', label = '1.96-std',linestyle = 'dotted',alpha = 0.5)
                ax.plot(pred_1,color = 'tab:green', linestyle = 'dotted',alpha = 0.5)#label = '1-std',
            ax.legend()
            if ic == 0:
                ax.set_ylabel(name)
            if evalexists:
                ax.set_title(f'{name},({lats[ir]},{lons[ir]}), r2 = {"{:.2e}".format(r2)}, {"interior"  if interiority[ir] else "shore"}')
            else:
                ax.set_title(f'{name}, ({lats[ir]},{lons[ir]}), {"interior"  if interiority[ir] else "shore"}')
        fig.suptitle(title,fontsize=24)
        fig.savefig(targetfile)
        flushed_print(targetfile)
        plt.close()



if __name__=='__main__':
    main()