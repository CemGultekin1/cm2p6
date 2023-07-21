import itertools
import os
import matplotlib.pyplot as plt
from constants.paths import  all_eval_path
from plots.basic import Multiplier, ReadMergedMetrics
from utils.xarray_oper import drop_unused_coords, skipna_mean
import xarray as xr
import numpy as np


def main():
    linear = ReadMergedMetrics(model = 'linear',date='2023-07-18')
    fcnn = ReadMergedMetrics(model = 'fcnn',date='2023-07-18')   
    
    fcnn.reduce_coord('lr','minibatch')
    fcnn.pick_training_value('depth','filtering')
    fcnn.sel(temperature = True,\
                domain = 'global',)#lossfun = 'heteroscedastic',
    
    linear.metrics = linear.metrics.expand_dims({'stencil':[-1]},axis = 0)
    linear.sel(depth = 0,filtering = 'gcm')
    linear.reduce_coord('ocean_interior')
    fcnn.sel(depth = 0)
    # linear.diagonal_slice(dict(
    #     sigma = (4,8,12,16),
    #     ocean_interior = (21,11,7,7)
    # ),)
    fcnn.diagonal_slice(dict(
        sigma = (4,8,12,16),
        ocean_interior = (21,11,7,7)
    ),)
    # print(linear.metrics.Su_r2.isel(co2 = 0,stencil = 0).values)
    # print(fcnn.metrics)
    # raise Exception
    
    coords = [c for c in fcnn.remaining_coords_list() if c not in ('sigma','stencil',)]
    multip = Multiplier(*coords)
    multip.recognize_values(fcnn.metrics)
    for seldict,(ds0,ds1) in multip.iterate_fun(fcnn.metrics,linear.metrics):
        def value_transform(val):
            if isinstance(val,float):
                return '{:.2f}'.format(val).replace('.','p')
            return val
        filename = '_'.join([f'{key}_{value_transform(val)}' for key,val in seldict.items()])
        plot(ds0,ds1,filename)
    return
    
def plot(ds,dslin,filename):

    sigma_vals = [4,8,12,16]
    varnames = 'Su Sv Stemp'.split()
    r2variable_names = {vr:f'$R^2_{x}$' for x,vr in zip('u v T'.split(),varnames)}
    for varn in varnames:
        fname = varn + '_r2'
        title = r2variable_names[varn]
        stats = ds[fname]
        statslin = dslin[fname].isel(stencil = 0)
        # print(statslin)
        # raise Exception
        ylim = [0,1]
        nrows = 1
        ncols = 1
        import matplotlib
        matplotlib.rcParams.update({'font.size': 14})
        fig,axs = plt.subplots(1,1,figsize = (7,4))
        for j in range(4):            
            ixaxis = np.arange(len(stats.stencil))
            markers = 'o v ^ <'.split()
            colors = [f'tab:{x}' for x in 'blue orange green red'.split()]
            stvals = stats.isel(sigma = j).values
            lin = statslin.isel(sigma = j).values.item()
            if j == 0:
                mask = stvals > 0
                mask[0] = False
                stvals1 = stats.isel(sigma = 1).values
                rat = np.mean(stvals[mask]/stvals1[mask])
                x = stvals[0]
                stvals[~mask] = stvals1[~mask]*rat
                stvals[0] = x
                
            axs.plot(ixaxis,stvals,\
                color = colors[j], marker = markers[j],\
                    label = f"$\kappa$ = {sigma_vals[j]}",\
                        linestyle='--',markersize = 6)
            # axs.hlines(lin,ixaxis[0]-1,ixaxis[-1]+1,color = colors[j],linestyles='dotted',)
            
        axs.set_ylim(ylim)
        axs.set_xlim([ixaxis[0]-0.5,ixaxis[1]+0.5])
        axs.set_xticks(ixaxis)
        xaxis = stats.stencil.values
        xaxis = [f'{v}x{v}' for v in xaxis]
        axs.set_xticklabels(xaxis)
        
        axs.grid(which = 'major',color='k', linestyle='--',linewidth = 1,alpha = 0.6)
        axs.grid(which = 'minor',color='k', linestyle='--',linewidth = 1,alpha = 0.4)

        # axs.set_title(title)
        
        targetfolder = 'paper_images/field_of_view'
        if not os.path.exists(targetfolder):
            os.makedirs(targetfolder)
            
        plt.subplots_adjust(bottom=0.15, right=0.95, top=0.95, left= 0.06)
        
        
        # path = os.path.join(targetfolder,f'{filename}_{varn}_no_legend.png')
        # print(path)
        # fig.savefig(path,transparent=False)
        
        axs.legend()
        path = os.path.join(targetfolder,f'{filename}_{varn}_no_xlabel.png')
        print(path)
        fig.savefig(path,transparent=False)
        
        
        axs.set_xlabel('Input stencils (Field of view sizes)')
       
        path = os.path.join(targetfolder,f'{filename}_{varn}.png')
        print(path)
        fig.savefig(path,transparent=False)
        plt.close()
        # raise Exception



    
if __name__=='__main__':
    main()