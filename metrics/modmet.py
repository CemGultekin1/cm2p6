import itertools
import os
from typing import Dict, List, Tuple, Union
from constants.paths import EVALS, all_eval_path
from options.reduce import ScalarArguments
from utils.arguments import options
from utils.slurm import flushed_print
from utils.xarray_oper import drop_unused_coords,skipna_mean,cat
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
            # if isinstance(val,str):
            #     val = hash(val)
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
    def file_exists(self,):
        return os.path.exists(self.filename)
    def load(self,):
        self.metrics = xr.open_dataset(self.filename,)
        # met = self.metrics.isel(co2 = 0,depth = 0)
        # from utils.xarray_oper import plot_ds
        # keys = met.data_vars.keys()
        # met_data = {key:met[key] for key in keys if 'Su_true' in key}
        # plot_ds(met_data,f'metrue-{self.coord_dict["sigma"]}-{self.coord_dict["filtering"]}.png',ncols = 3)
        # raise Exception
        self.metrics = drop_unused_coords(self.metrics)
        return self
    
    
class ModelResultsCollection:
    collision_tag:str = 'training_'
    def __init__(self) -> None:
        self.models : List[MergeMetrics] = []
    def add_metrics(self,mm:MergeMetrics):
        self.models.append(mm)
    @staticmethod
    def collision_naming(mkey:str):
        return ModelResultsCollection.collision_tag + mkey
    
    def merged_dataset(self,):
        if not self.models:
            print('no metrics to merge')
            return xr.Dataset()
        model = self.models[0]
        model_coord,feat_coord = model.get_coords_dict()
        
        
        model_coord = {key:[val] for key,val in model_coord.items()}
        feat_coord :Dict[str,list]= {key:np.array(val).flatten().tolist() for key,val in feat_coord.items()}
        
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
        
        
        
        rename_dict = {mkey:self.collision_naming(mkey) if mkey in feat_coord else mkey for mkey in model_coord.keys() }
        def rename_dict_fun(dc:dict):
            dc_ = {}
            for key in dc:
                if key in rename_dict:
                    dc_[rename_dict[key]] = dc[key]
                else:
                    dc_[key] = dc[key]
            return dc_
        model_coord = rename_dict_fun(model_coord,)
        attrs = dict()
        model_keys = list(model_coord.keys())
        for key in model_keys:
            val = model_coord[key]
            if len(val) > 1 or len(val)==0:
                continue
            attrs[key] = model_coord.pop(key)[0]
            
        feat_keys = list(feat_coord.keys())
        for key in feat_keys:
            val = feat_coord[key]
            if len(val) > 1 or len(val)==0:
                continue
            attrs[key] = feat_coord.pop(key)[0]
        
        
        
        merged_coord = dict(model_coord,**feat_coord)
       
        for key in merged_coord:
            if key in attrs:
                attrs.pop(key)
        shape = [len(v) for v in merged_coord.values()]
        print(
            ' '.join([f'{key}:{n}' for key,n in zip(merged_coord,shape)])
        )
        
        # data_set = start_nan_dataset(list(model.metrics.data_vars.keys()),merged_coord)
        # print(data_set)
        # raise Exception
        datasets = []
        flushed_print(f'total number of models = {len(self.models)}')
        for model in self.models:
            mdict,_ = model.get_coords_dict()
            mdict = rename_dict_fun(mdict)
            mdict = {key:[val] for key,val in mdict.items() if key in merged_coord}
            new_metrics = model.metrics.copy()
            new_metrics = new_metrics.expand_dims(**mdict)
            datasets.append(new_metrics)
        ds = xr.merge(datasets,fill_value = np.nan)
        ds.attrs = attrs
        # for key,val in ds.attrs.items():
        #     print(f'{key} \t- {val} \t- {type(val)}')
        # raise Exception
        # for key,val in data_vars.items():
        #     data_vars[key] = (list(merged_coord.keys()),val.reshape(shape))
        # ds = xr.Dataset(data_vars = data_vars,coords = merged_coord)
        return ds
    
    
class ModelResults:
    def __init__(self,tag:str,path:str  = '') -> None:
        if len(path) == 0:
            self.path = all_eval_path().replace('.nc',f'{tag}.nc')
        else:
            self.path = path
        self.metrics = xr.open_dataset(self.path)
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
            # slc = slc.expand_dims({keyname:vals[:1]},axis = 0).reindex(indexers = {keyname:keyvals},fill_value = 0)
            # if is_empty_xr(newmetric):
            #     newmetric = slc
            # else:
            #     newmetric += slc
        newmetric = cat(newmetric,'keyname')
        self.metrics = newmetric

            
def main():
    mr = ModelResults('20230712_linear')
    metrics = mr.metrics
    # print(metrics)
    # return
    metrics = metrics.isel(co2 = 1,sigma = 0).sel(stencil = [1,3],depth = 0)
    print(metrics)
    # metrics = metrics.isel(co2 = 0,lr = 1,lossfun= 0,temperature = 0,training_filtering=1,training_depth=0,depth=0,minibatch = 0,filtering = 1 ,)
    # metrics = metrics.sel(stencil = 21)
    # metrics = metrics.sel(sigma = 4)
    for key in metrics.data_vars:
        print(
            f'{key}:\t\t\t{metrics[key].values}'
        )
    
    return
    mr = ModelResults('20230712_')
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
            
            