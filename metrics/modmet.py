import itertools
import os
from typing import Dict, List, Tuple
from constants.paths import EVALS
from options.reduce import ScalarArguments
from utils.arguments import options
from utils.slurm import flushed_print
import xarray as xr
import numpy as np

class ModelCoordinates(ScalarArguments):
    def __init__(self,modelargs:List[str]) -> None:
        super().__init__()
        self.modelargs = modelargs
        self.coord_dict = self.transform_arguments(*modelargs)
        _,modelid = options(modelargs,key = 'model')
        self.modelid = modelid
    def get_coords_dict(self,tag:str = ''):
        return {tag + key:val for key,val in self.coord_dict.items() if val is not None}

class ModelMetric(ModelCoordinates):
    metrics :xr.Dataset
    def __init__(self, modelargs: List[str],metrics:xr.Dataset) -> None:
        super().__init__(modelargs)
        self.metrics :xr.Dataset= metrics
    def get_coords_dict(self,):#model_tag:str,metric_tag:str):
        return super().get_coords_dict(),{key : self.metrics[key].values.tolist() for key in self.metrics.coords.keys()}
    def transform_feature_coord_names(self,rename:Dict[str,str]):
        return self.metrics.rename(rename)
    def past_coords_to_metric(self,coords:Tuple[str]):
        for coord in coords:
            if coord not in self.coord_dict:
                raise Exception
            val = self.coord_dict[coord]
            ckeys = list(self.metrics.coords.keys())
            if coord in ckeys:
                continue
            if isinstance(val,str):
                val = hash(val)
            self.metrics = self.metrics.expand_dims({coord:[val]},axis = 0)

class MergeMetrics(ModelMetric):
    metrics :xr.Dataset
    def __init__(self, modelargs: List[str],) -> None:
        super().__init__(modelargs,xr.Dataset())
    def merge(self,metrics:xr.Dataset):
        self.metrics = xr.merge([self.metrics,metrics])
    @property
    def filename(self,):
        return os.path.join(EVALS,self.modelid+'.nc')
    def save(self,):
        print(self.filename)
        self.metrics.to_netcdf(self.filename,mode = 'w')
    def load(self,):
        self.metrics = xr.open_dataset(self.filename,)
        return self
    
# class FaultyFilteringNamingResolution(MergeMetrics):
#     def __init__(self,x:MergeMetrics) -> None:
#         self.__dict__.update(x.__dict__)
#         self.merged_metrics = x
#     def find_best_fitting_filter(self,):
#         self.merged_metrics.filtering
    
class ModelResultsCollection:
    collision_tag:str = 'training_'
    def __init__(self) -> None:
        self.models : List[MergeMetrics] = []
    def add_metrics(self,mm:MergeMetrics):
        self.models.append(mm)
    def merged_dataset(self,):
        model = self.models[0]
        model_coord,feat_coord = model.get_coords_dict()
        
        
        model_coord = {key:[val] for key,val in model_coord.items()}
        feat_coord :Dict[str,list]= {key:np.array(val).tolist() for key,val in feat_coord.items()}
        
        for model in self.models:
            mdict,fdict = model.get_coords_dict()
            for key,val in mdict.items():
                assert np.isscalar(val)
                model_coord[key].append(val)
            for key,val in fdict.items():
                if np.isscalar(val):
                    val = np.array([val])
                feat_coord[key].extend(val)
        model_coord = {key:np.unique(val).tolist() for key,val in model_coord.items()}
        feat_coord = {key:np.unique(val).tolist() for key,val in feat_coord.items()}
        
        
        model_keys = list(model_coord.keys())
        for key in model_keys:
            val = model_coord[key]
            if len(val) > 1:
                continue
            model_coord.pop(key)
            
        feat_keys = list(feat_coord.keys())
        for key in feat_keys:
            val = feat_coord[key]
            if len(val) > 1:
                continue
            feat_coord.pop(key)
        
        
        
        rename_dict = {mkey:self.collision_tag + mkey if mkey in feat_coord else mkey for mkey in model_coord.keys() }
        
        def rename_dict_fun(dc:dict):
            dc_ = {}
            for key in dc:
                if key in rename_dict:
                    dc_[rename_dict[key]] = dc[key]
                else:
                    dc_[key] = dc[key]
            return dc_
        model_coord = rename_dict_fun(model_coord,)
        merged_coord = dict(model_coord,**feat_coord)
        
        shape = [len(v) for v in merged_coord.values()]
        print(
            ' '.join([f'{key}:{n}' for key,n in zip(merged_coord,shape)])
        )
        def empty_arr():
            return np.ones(np.prod(shape))*np.nan
        
        data_vars = {}
        flushed_print(f'total number of models = {len(self.models)}')
        for model in self.models:
            mdict,_ = model.get_coords_dict()
            mdict = rename_dict_fun(mdict)
            # print(mdict)
            # print(fdict)
            inds = [v.index(mdict[k]) for k,v in model_coord.items()]
            lent = np.prod(shape[len(inds):])
            inds = inds + [0]*(len(shape) - len(inds))            
            alpha0 = np.ravel_multi_index(inds,shape)
            alpha1 = alpha0 + lent
            for key in model.metrics.data_vars.keys():
                if key not in data_vars:
                    data_vars[key] = empty_arr()
                datarr = model.metrics[key]
                datarr = datarr.reindex(indexers = feat_coord, fill_value = np.nan)
                values = datarr.values
                data_vars[key][alpha0:alpha1] = values.flatten()

        for key,val in data_vars.items():
            data_vars[key] = (list(merged_coord.keys()),val.reshape(shape))
        ds = xr.Dataset(data_vars = data_vars,coords = merged_coord)
        return ds