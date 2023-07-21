import itertools
import logging
import os
import matplotlib.pyplot as plt
import matplotlib
from constants.paths import  all_eval_path
from metrics.mcol import ModelMetricsCollection
from utils.xarray_oper import skipna_mean,sel_available,drop_unused_coords
import xarray as xr
import numpy as np
from plots.for_paper.saliency import SubplotAxes

class ReadMergedMetrics(ModelMetricsCollection):
    def pick_matching_ocean_interiority(self,):
        stv = self.metrics.stencil.values
        glv = self.metrics.ocean_interior.values
        glv = [x for x in glv if x in stv]
        seldict = dict(stencil = glv,ocean_interior = glv)
        self.diagonal_slice(seldict)
    def to_integer_depths(self,):
        for key in self.metrics.coords.keys():
            if 'depth' not in key:
                continue
            vals = self.metrics[key].values.astype(int)
            self.metrics = self.metrics.assign_coords({key:vals})
            
    def pick_matching_depths(self,):
        stv = self.metrics.depth.values
        glv = self.metrics.training_depth.values
        
        glvsel = [glv[i] for i,x in enumerate(glv) if int(x) in stv]
        seldict = dict(depth = glvsel,training_depth = glvsel)
        self.sel(**seldict)
    def reduce_usual_coords(self,coords = 'lr temperature minibatch'.split()):
        self.reduce_coord(*coords)
    
class Multiplier:
    def __init__(self,*coords:str):
        self.multip_coords = coords
        self.select_vals = {}
    def receive_value(self,coord:str,*vals):
        self.select_vals[coord] = vals
    def recognize_values(self,ds):
        self.select_vals = {mc:ds.coords[mc].values for mc in self.multip_coords}
    def iterate_selection_dict(self,):
        for vals in itertools.product(*self.select_vals.values()):
            seldict = {mc:val for mc,val in zip(self.select_vals.keys(),vals)}
            yield seldict
    def iterate_fun(self,*ds0:xr.Dataset):
        for seldict in self.iterate_selection_dict():
            ds1 = [sel_available(ds,seldict) for ds in ds0]
            ds1 = [drop_unused_coords(ds) for ds in ds1]
            yield seldict,ds1
            
def plot(stats_ns,filenametag):
    varnames = list(stats_ns.data_vars.keys())
    vartypes = np.unique([vn.split('_')[1] for vn in varnames])
    vartypes = ['r2']
    colors = 'r b g k'.split()
    markers = 'o ^ v < >'.split()
    vnames_dict = dict(
        Su_r2 = '$R^2_u$',
        Sv_r2 = '$R^2_v$',
        Stemp_r2 = '$R^2_T$',
        Su_corr = '$C_u$',
        Sv_corr = '$C_v$',
        Stemp_corr = '$C_T$'           
    )
    model_renames_dict = {
        'linear' : 'Linear',
        'global' : 'CNN (global)',
        'four_regions' : 'CNN (4 regions)',
    }
    modelnames = [model_renames_dict[ns] for ns in stats_ns.model.values]
    stats_ns= stats_ns.assign_coords({'model' : modelnames})

    matplotlib.rcParams.update({'font.size': 14})
    for vtype in vartypes:
        vnselect = [vn for vn in varnames if vn.split('_')[1] == vtype]
        ncols = len(vnselect)
        nrows = 1
        # fig,axs = plt.subplots(nrows,ncols, figsize = (ncols*3,nrows*3))
        fig = plt.figure(figsize = (ncols*5,nrows*5))
        spaxes = SubplotAxes(1,ncols,sizes = ((1,),(3,3,3)),ymargs=(0.18,0.01,0.1),xmargs = (0.05,0.03,0.01))
        for i in range(ncols):
            vname = vnselect[i]
            # ax = axs[i]
            nlossfun = len(stats_ns.lossfun)
            ax = fig.add_axes(spaxes.get_ax_dims(0,i))
            for j,filti in itertools.product(range(len(stats_ns.sigma)),range(nlossfun)):
                vals = stats_ns.isel(sigma = j,lossfun = filti)
                y = vals[vname]
                if np.all(np.isnan(y)):
                    continue
                xticklabels =  [str(x) for x in stats_ns.model.values.tolist()]
                
                xticks = np.array([0,1.5,3])
                if filti == 1:
                    x = xticks[1:]
                    y = y[1:]
                else:
                    x = xticks
                
                y = np.where(y < 0,y/32,y)
                negvalflag = np.any(y < 0) 
                sep = 0.02
                sigma = vals.sigma.values.item()
                
                offset = (10 - sigma )*sep/2
                offwidth = 10*sep/2
                xoeff = x + offset + (filti*2 - 1)*offwidth
                if filti == 0:
                    xoeff[0] = x[0] + offset
                pltkwargs = dict(
                    label = f"$\kappa$ = {sigma}",\
                    color = colors[j],linestyle = 'None',\
                    marker = markers[j],
                    markersize = 8,
                    fillstyle = 'full' if filti == 0 else 'none'
                )
                if filti == 1:
                    pltkwargs.pop('label')
                    
                ax.plot(xoeff,y,**pltkwargs)
                if filti == 0:
                    ax.set_xticks(x)
                    ax.set_xticklabels(xticklabels,rotation=25)
                
                if j == 3 and filti == 1:
                    ax.plot([1],[np.nan],color = 'gray',linestyle = 'none',marker = 's',fillstyle = 'none',label = 'htrsc.')
                    ax.plot([1],[np.nan],color = 'gray',linestyle = 'none',marker = 's',fillstyle = 'full',label = 'MSE',)
            if vtype in ['r2','corr']:
                ymin =np.amin(stats_ns[vname].values)
                ymax = np.amax(stats_ns[vname].values)
                sep = np.mean([ymin,ymax])*0.1
                
                # if negvalflag:                    
                #     yticks = [-1/8,-1/16,0,0.25,0.5,0.75,1]
                #     yticklabels = ['-4','-2','0','0.25','0.5','0.75','1.']
                #     ax.set_yticks(yticks)
                #     ax.set_yticklabels(yticklabels)
                #     ax.set_ylim([-1/8,1.01])
                # else:
                #     ax.set_ylim(ymin - sep,ymax + sep)
                    
                # else:
                ax.set_ylim([-0.4,1])
                #     yticks = [0,0.25,0.5,0.75,1]
                #     yticklabels = ['0','0.25','0.5','0.75','1.']
                #     ax.set_yticks(yticks)
                #     ax.set_yticklabels(yticklabels)
            ax.set_xlim([-0.5,xticks[-1] + 0.5])
            ax.grid( which='major', color='k', linestyle='--',alpha = 0.5)
            ax.grid( which='minor', color='k', linestyle='--',alpha = 0.5)
            if vtype in ['r2','corr']:
                if i==0:
                    ax.legend(loc = 'lower right')
            else:
                ax.legend(loc = 'upper right')
            ax.set_title(vnames_dict[vname])
            # ax.set_ylabel(vtype)
        filename = f"{vtype}_{filenametag}.png"
        
        target_folder = 'paper_images/basic'
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        plt.subplots_adjust(bottom=0.18, right=0.98, top=0.91, left= 0.05)
        fig.savefig(os.path.join(target_folder,filename),transparent=False)
        logging.info(os.path.join(target_folder,filename))
        plt.close()
        # raise Exception



def main():    
    linear = ReadMergedMetrics(model = 'linear',date='2023-07-18')
    fcnn = ReadMergedMetrics(model = 'fcnn',date='2023-07-18')

    linear.reduce_coord('filtering','ocean_interior')
    
    fcnn.reduce_usual_coords()

    fcnn.pick_training_value('filtering','depth',)
    fcnn.pick_matching_ocean_interiority()
    fcnn.sel(depth  = 0)
    linear.sel(depth  = 0)
    fcnn.diagonal_slice(dict(
        sigma = (4,8,12,16),
        stencil = (21,11,7,7)
    ),)
    
    model_names_dict = {
        'linear' : 'Linear',
        'global' : 'CNN (global)',
        'four_regions' : 'CNN (4 regions)',
    }

    def model_names(**kwargs):
        dom = kwargs.get('domain',)
        model_name = model_names_dict[dom]
        return model_name
    def model_names_with_lossfuns(**kwargs):
        lsf = kwargs.get('lossfun','')
        mod = model_names(**kwargs)
        return mod +'_' + lsf
    def model_name_from_coordinate(**kwargs):
        mod = kwargs.get('model')
        mod = '_'.join(mod.split('_')[:-1])
        return mod
    # def model_reindexer(ds):
        
    coords = [c for c in fcnn.remaining_coords_list() if c not in ('sigma','domain','lossfun',)]
    multip = Multiplier(*coords)
    multip.recognize_values(fcnn.metrics)
    for seldict,(ds0,ds1) in multip.iterate_fun(fcnn.metrics,linear.metrics):        
        ds1 = ds1.expand_dims({'domain':['linear']},axis = 0)
        ds0 = xr.merge([ds0,ds1])

        ds = fcnn.init_from_metrics(ds0,)        
        ds.assign_new_variable(model_names_with_lossfuns,'model')
        ds.variable2coordinate('model',resolve_collisions='max',leave_dims='sigma')
        ds.assign_new_variable(model_name_from_coordinate,'model_name')

        ds.metrics = ds.metrics.rename({'domain':'model'})
        ds.assign_new_variable()
        ds.metrics = ds.metrics.
        def value_transform(val):
            if isinstance(val,float):
                return '{:.2f}'.format(val).replace('.','p')
            return val
        
        filenametag = '_'.join([f'{key}_{value_transform(val)}' for key,val in seldict.items()])
        plot(ds.metrics,filenametag)
    
if __name__=='__main__':
    main()
    
    
   
