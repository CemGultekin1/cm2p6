import itertools
import os
from typing import Dict, List, Tuple, Union
from constants.paths import EVALS, all_eval_path
from options.params import MODEL_PARAMS
from options.reduce import ScalarArguments
from utils.arguments import options
from utils.slurm import flushed_print
from utils.xarray_oper import drop_unused_coords,skipna_mean,cat
import xarray as xr
import numpy as np
from metrics.modmet import ModelResultsCollection
from metrics.gather_evals1 import ModelMetricsGathering

class ModelMetricsCollection(ModelMetricsGathering):
    def __init__(self,date:str = '2023-07-14', model:str = ''):        
        if len(model) == 0:
            super().__init__('', 0, 0,)
        else:
            super().__init__('', 0, 0,model = model)
        self.date = date
        self.metrics = self.read(self.merged_path)
        
    def reduce_coord(self,*coords:str):
        for coord in coords:
            if coord not in self.metrics.coords.keys():
                continue
            self.metrics = self.metrics.max(dim = coord,skipna = True)
    def pick_training_value(self,*coords:str):
        ckeys = self.metrics.coords.keys()
        for coord in coords:
            colcoord = ModelResultsCollection.collision_naming(coord)
            if colcoord not in ckeys:
                continue
            self.metrics = xr.where(self.metrics[coord] == self.metrics[colcoord],self.metrics,0).sum(dim = colcoord)
    def sel(self,**kwargs):
        self.metrics = self.metrics.sel(**kwargs)
        self.metrics = drop_unused_coords(self.metrics)
    def isel(self,**kwargs):
        self.metrics = self.metrics.isel(**kwargs)
        self.metrics = drop_unused_coords(self.metrics)
    def assign_new_variable(self,fun:callable,varname:str):
        coordvals = tuple(self.metrics.coords.values())
        shp = [len(cd) for cd in coordvals]
        names = []
        coordkeys = tuple(self.metrics.coords.keys())
        for x in itertools.product(*coordvals):            
            names.append(fun(**dict(zip(coordkeys,x))))
        names = np.array(names).reshape(shp)
        self.metrics[varname] = (coordkeys,names)
    def variable2coordinate(self,varname:str,resolve_collisions:str = 'none',leave_dims:List[str] = []):
        if varname  not in self.metrics.data_vars:
            raise Exception
        metrics = self.metrics
        x = metrics[varname]
        xval:np.ndarray = x.values
        ux = np.unique(xval.flatten())
        collision_flag =  len(ux) < xval.size
        if resolve_collisions == 'none' and collision_flag:
            raise Exception
        reduc_coords = [key for key in metrics.coords.keys() if key not in leave_dims] #+ [varname]
        newmetric = xr.Dataset()
        metrics = metrics.drop(varname)
        metrics_dict = {}
        for ux_ in ux:
            uxx = xr.where(x == ux_,metrics,np.nan)
            if resolve_collisions == 'max':
                uxx = uxx.max(dim = reduc_coords,skipna = True)
            elif resolve_collisions == 'mean':
                uxx = skipna_mean(uxx,reduc_coords)
            metrics_dict[ux_] = uxx
        newmetric = cat(metrics_dict,varname)
        self.metrics = newmetric
    def diagonal_slice(self,seldict:Dict[str,Tuple[Union[int,float,str]]]):
        metrics = self.metrics
        selnames = list(seldict.keys())
        keyname = selnames[0]
        keyvals = seldict[keyname]
        newmetric = {}
        for vals in zip(*seldict.values()):
            newmetric[vals[0]] = metrics.sel(dict(zip(selnames,vals))).drop(selnames)
        newmetric = cat(newmetric,'keyname')
        self.metrics = newmetric
            
def main():
    mr = ModelMetricsCollection(date ='2023-07-14',model = 'linear')
    metrics = mr.metrics
    print(metrics)
    return
    metrics = metrics.isel(co2 = 1,sigma = 0).sel(stencil = [1,3],depth = 0)
    
    # metrics = metrics.isel(co2 = 0,lr = 1,lossfun= 0,temperature = 0,training_filtering=1,training_depth=0,depth=0,minibatch = 0,filtering = 1 ,)
    # metrics = metrics.sel(stencil = 21)
    # metrics = metrics.sel(sigma = 4)
    for key in metrics.data_vars:
        print(
            f'{key}:\t\t\t{metrics[key].values}'
        )
    
    return
    mr = ModelMetricsCollection('20230712_')
    metrics = mr.metrics
    mr.reduce_coord('lr','temperature','stencil','minibatch')
    mr.pick_training_value('filtering',)
    mr.sel(depth = 0,training_depth = 0,co2 = 0)
    def name_fun(**kwargs):
        contains = 'filtering lossfun domain'.split()
        values = [str(kwargs[k].item()) for k in contains]
        return '-'.join(values)
    mr.assign_new_variable(name_fun,'name')
    mr.variable2coordinate('name',resolve_collisions='mean',leave_dims=['sigma'])
    metrics = mr.metrics.sel(sigma = 4)
    sur2 = metrics.Su_r2.values
    names = metrics.name.values
    st = '\n' + '-'*64 + '\n'
    print(
        st.join([f'{name}:\t\t\t\t{su}' for name,su in zip(names,sur2)])
    )
if __name__ == '__main__':
    main()
            
            