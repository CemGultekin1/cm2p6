from copy import deepcopy
import itertools
import os
from typing import Dict, List, Tuple
from constants.paths import DISTS
from utils.xarray_oper import existing_sel,drop_unused_coords
import xarray as xr
from utils.arguments import options
import numpy as np
from utils.slurm import ArgsReader
from utils.arguments import is_consistent
import matplotlib.pyplot as plt
import matplotlib
from data.coords  import DEPTHS, SIGMAS
def get_args(**kwargs):
    lr = ArgsReader('offline_sweep2.txt')
    kw = deepcopy(kwargs)
    kw.pop('co2')
    consistent_found = False
    for line in lr.iterate_lines():
        args = line.split()
        if is_consistent(args,key = 'model',**kw):
            consistent_found = True
            break
        
    # print(kwargs)
    # print(f'\t\t{args}')
    if consistent_found:
        return args
    else:        
        # print(kw)
        # raise Exception
        return None

def disjoint_windowed_sum(vec:np.ndarray,fact:int):
    return np.array([np.sum(vec[fact*i:fact*(i+1)]) for i in range(len(vec)//fact)])
def simplify_distribution(midpts,dists,fact):
    midpts = disjoint_windowed_sum(midpts,fact)/fact
    dists = disjoint_windowed_sum(dists,fact)
    return midpts,dists

class DistributionData:
    simplify_degree :int = 12
    def __init__(self,selkwargs,ds) -> None:
        self.selkwargs = selkwargs
        self.data_vars = {}
        if ds is None:
            return
        for key,val in ds.data_vars.items():
            dim = val.dims[0]
            coords = ds[dim].values
            npvalues = val.values
            coords,npvalues = simplify_distribution(coords,npvalues,self.simplify_degree)
            self.data_vars[key] = (coords,npvalues)
    def read_feat(self,st:str):
        return self.selkwargs[st]
    def __eq__(self,__o:'DistributionData'):
        for key,val in __o.selkwargs.items():
            if self.selkwargs[key] != val:
                return False
        return True
    def get_inexistents(self,dd:'DistributionData'):
        for varname,val in dd.data_vars.items():
            if varname not in self.data_vars:
                self.data_vars[varname] = val
    def get_variable(self,varname:str):
        for key in self.data_vars:
            if varname in key:
                return self.data_vars[key]
        return None,None
    
def filtering_name_correction(ds:xr.Dataset):
    if 'filtering' not in ds.coords:
        return ds
    legal_filters = 'gcm gaussian'.split()
    fltvs = ds.filtering.values
    non_legal_filters = [flt for flt in fltvs if flt not in legal_filters]
    if not bool(non_legal_filters):
        return ds
    legal_filter_nums = [sum([ord(a) for a in fil]) for fil in legal_filters]
    
    new_filtering = [ legal_filters[legal_filter_nums.index(fltnum)]      for fltnum in fltvs]
    return ds.assign_coords({'filtering':new_filtering})
def co2_value_quick_fix(ds:xr.Dataset):
    ds =  ds.assign_coords({'co2':[False,True]})
    return ds
def depth_quick_fix(ds:xr.Dataset):
    ds =  ds.assign_coords({'depth':[int(d) for d in ds.depth.values]})
    return ds
def get_distribution(**kwargs):
    args = get_args(**kwargs)
    if args is None:
        return None
    _,modelid = options(args,key = 'model')
    path =os.path.join(DISTS,modelid +'.nc')
    if os.path.exists(path):
        ds =xr.open_dataset(path)
        ds = filtering_name_correction(ds)
        ds = co2_value_quick_fix(ds)
        ds = depth_quick_fix(ds)
    else:
        print(kwargs)
        ds = None
    return ds

def simplify_coords(ds,**rootkwargs):
    ds = existing_sel(ds,**rootkwargs)
    ds = drop_unused_coords(ds)    
    return ds

class DistributionCollector:
    def __init__(self, selective_dict:Dict[str,Tuple[str]],**rootkwargs) -> None:
        self.rootkwargs = rootkwargs
        
        self.selective_dict = selective_dict#{key:vals[0] for key,vals in selective_dict.items()}
        self.datasets :List[DistributionData] = []
    def get_rootkwargs(self,**upd):
        rk = deepcopy(self.rootkwargs)
        rk.update(upd)
        return rk
    def load_datasets(self,):      
        keys = tuple(self.selective_dict.keys())  
        for values in itertools.product(*self.selective_dict.values()):
            upd = dict(zip(keys,values))            
            rk = self.get_rootkwargs(**upd)
            ds = get_distribution(**rk)
            if ds is not None:
                ds = simplify_coords(ds,**self.rootkwargs)
            self.datasets.append(DistributionData(rk,ds))
    def find(self,**kwargs):
        i = self.datasets.index(DistributionData(kwargs,None))
        return self.datasets[i]    
    def temperature_cross(self,):
        new_dataset_list = []
        for ds in self.datasets:
            if ds.selkwargs['temperature']:
                continue
            sd = deepcopy(ds.selkwargs)
            sd['temperature'] = True
            dd = self.find(**sd)
            ds.get_inexistents(dd)
            new_dataset_list.append(ds)
        for ds in new_dataset_list:
            ds.selkwargs['temperature'] = True
        self.datasets = new_dataset_list
            
    def get_variable(self,varname:str,featname:str):
        for ds in self.datasets:
            yield ds.get_variable(varname),ds.read_feat(featname)
                
            



def get_domains(sel_dict):
    sel_dict['domain'] = 'four_regions'
    

class ColorFinder:
    def __init__(self,):
        self.colors = 'r g b c m y k'.split()
        self.counter = 0
        self.selected_colors = {}
    def give_color(self,key:str):
        if key not in self.selected_colors:            
            clr = self.colors[self.counter]
            self.counter+=1
            self.selected_colors[key] = clr
        return self.selected_colors[key]
    

def main():
    CO2S = [False,True]
    cf = ColorFinder()
    plt_kwargs = dict(linewidth = 3)
    matplotlib.rcParams.update({'font.size': 20})
    root_folder = 'paper_images/distributions'
    label_conversion = {
        'global': 'CNN(Global)',
        'four_regions': 'CNN(4Regs)'
    }
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)
    filename_params = ['filtering','sigma','co2']
    to_file_look = dict(
        filtering = lambda x: f'filt_{x}',
        sigma = lambda x: f'sig_{x}',
        depth = lambda x:f'dp_{int(x)}',
        co2 = lambda x: f'co2_{int(x)}',
        forcing = lambda x: f'{x}',
    )
    for sigma,co2, in itertools.product(SIGMAS,CO2S):
        root_kwargs = dict(
            lossfun = 'heteroscedastic',
            sigma = sigma,
            filtering = 'gcm',
            co2 = co2,
            depth = 0,#int(depth)
        )
        sel_dict = dict(
            domain = ('four_regions','global'),
            temperature = (False,True),
        )
        distcol = DistributionCollector(
            sel_dict,**root_kwargs
        )
        distcol.load_datasets()
        distcol.temperature_cross()
    
        forcings = 'Su Sv Stemp'.split()
        
        
        
        for forcing in forcings:        
            coords = None
            fig,ax = plt.subplots(1,1,figsize = (5,5))
            for (coords,vals),feat in distcol.get_variable(forcing,'domain'):
                if coords is None:
                    continue
                vals = vals/np.sum(vals)
                ax.semilogy(coords,vals,color = cf.give_color(feat),label = label_conversion[feat],**plt_kwargs)
            if coords is None:
                coords = np.linspace(-5.5,5,100)
            gauss = np.exp(-coords**2/2)
            gauss = gauss/np.sum(gauss)        
            feat = '$\mathcal{N}(0,1)$'
            ax.plot(coords,gauss,linestyle = '--',color = cf.give_color(feat),label = feat,alpha=0.5,**plt_kwargs)
            
            ax.set_ylim([1e-6,5e-2])
            ax.grid( which='major', color='k', linestyle='--',alpha = 0.5)
            ax.grid( which='minor', color='k', linestyle='--',alpha = 0.5)
            filename = [to_file_look[key](root_kwargs[key]) for key in filename_params] 
            filename += [to_file_look['forcing'](forcing)]
            filename = '_'.join(filename) + '.png'
            path = os.path.join(root_folder,filename)
            fig.tight_layout()
            
            
            fig.savefig(path.replace('.png','_no_legend.png'))
            ax.legend()
            fig.savefig(path)
            print(path)
            plt.close()
    
    return
    
    
    return
    x = ds[cooname].values
    y = ds[varname].values
    ax.plot(x,y,label = name)
    gauss = np.exp( - x**2/2)
    gauss = gauss/np.sum(gauss)
    ax.plot(x,gauss,label = '$\mathcal{N}(0,1)$')
    ax.legend()
    fig.savefig('dummy.png')
    
    

if __name__=='__main__':
    main()