import itertools
import os
import matplotlib.pyplot as plt
from constants.paths import DISTS
from plots.for_paper.saliency import SubplotAxes
import xarray as xr
from utils.slurm import flushed_print
import numpy as np

class ModelSelection:
    selection_dictionary = {
        'GZ21': dict(domain = 'four_regions',temperature = False,),
        'GZ21-T': dict(domain = 'four_regions',temperature = True,),
        'Glb': dict(domain = 'global',temperature = False,),
        'Glb-T': dict(domain = 'global',temperature = True,)
    }
    def __init__(self,ds:xr.Dataset) -> None:
        self.metrics = ds
    def iterate_variables(self,):
        for key in self.metrics.data_vars.keys():
            yield key, ModelMetric(self.metrics[key])
    
        
class ModelMetric(ModelSelection):
    metrics:xr.DataArray
    def __init__(self,ds:xr.DataArray) -> None:
        super().__init__(ds)
    def separate_dataset(self,ds:xr.Dataset):
        return {key:ds.sel(**val) for key,val in self.selection_dictionary.items()}
    def stack_across(self,skip_na:bool = True):
        byname = {}
        for modelname,select  in self.selection_dictionary.items():
            sel_metric = self.metrics.sel(**select)
            coords = {key:sel_metric[key].values for key in sel_metric.coords.keys() if key not in select }
            metric = sel_metric.values
            metric = np.where(metric == 0,np.nan,metric)
            if skip_na and np.all(np.isnan(metric)):
                continue
            byname[modelname] = metric
        
        return list(byname.keys()),np.stack(list(byname.values()),axis = 0),coords
            
def main():
    root = DISTS
    filename = os.path.join(root,'all.nc')
    ds = xr.open_dataset(filename,mode = 'r')
    ds = ds.isel(depth = 0,training_depth = 0,co2 = 0).drop('depth training_depth co2'.split())
    # print(ds)
    ms = ModelSelection(ds)
    fig = plt.figure(figsize = (3*3,1*3))
    spaxes = SubplotAxes(1,3,sizes = ((1,),(3,3,2)),ymargs=(0.18,0.01,0.1),xmargs = (0.05,0.03,0.01))
    colors = 'r b g k'.split()
    markers = 'o ^ v < >'.split()
    vnames_dict = dict(
        Su_test = '$\mathcal{E}^2_u$',
        Sv_test = '$\mathcal{E}^2_v$',
        Stemp_test = '$\mathcal{E}^2_T$',      
    )
    for icol,(varname,mm) in enumerate(ms.iterate_variables()):
        models,values,coords = mm.stack_across()
        dims = spaxes.get_ax_dims(0,icol)
        ax = fig.add_axes(dims)
        n = len(models)
        for i,sigma in enumerate(coords['sigma']):
            val = values[:,i]
            ax.semilogy(range(n),val,label = f'\u03C3={sigma}',marker = markers[i],color = colors[i],linestyle = 'None',markersize = 4)
        ax.set_xticks(range(n))
        ax.set_xticklabels(models,rotation=45)
        ax.set_xlim([-0.5,n - 0.5])
        ax.set_title(vnames_dict[varname])
        ax.grid( which='major', color='k', linestyle='--',alpha = 0.5)
        ax.grid( which='minor', color='k', linestyle='--',alpha = 0.5)
        if icol == 0:
            ax.legend()
    fig.savefig('paper_images/hierarchy/en_dist.png')

if __name__=='__main__':
    main()