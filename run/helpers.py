import torch
import numpy as np
import xarray as xr
from utils.arguments import options
import time

class Timer:
    def __init__(self,):
        self.times = {}
    def start(self,label):
        if label not in self.times:
            self.times[label] = []
        self.times[label].append(time.time())
    def end(self,label):
        assert label in self.times
        t1 = self.times[label][-1]
        self.times[label][-1] = time.time() - t1
    def __repr__(self) -> str:
        keys = [f"\t{lbl} : {np.mean(vals[-30:-1])}" for lbl, vals in self.times.items()]
        return "\n".join(keys)
    def reset(self,):
        self.times = {}
        
class PrecisionToStandardDeviation:
    def __init__(self,args) -> None:
        modelargs,_ = options(args,key = "model")
        self.square_root_flag = True
        if modelargs.lossfun == 'heteroscedastic_v2':
            self.square_root_flag = False    
    def __call__(self,x:torch.Tensor):
        if self.square_root_flag:
            return torch.sqrt(1/x)
        else:
            return 1/x
class AdaptiveHistogram:
    def __init__(self,ds:xr.Dataset,num_bins:int) -> None:
        '''
        neglects nan values
        '''
        num_bins = (num_bins//2)*2
        self.hnbins = num_bins//2
        dss = {key: ds[key].values for key in ds.data_vars}
        # self.coords = ds.coords
        # key = list(ds.data_vars.keys())[0]
        # self.dims =ds[key].dims
        nan_neglected = {key: x[x==x] for key,x in dss.items()}
        self.extremums = {key: 5.5 for key,x in nan_neglected.items()}
        self.counts = {key:np.zeros((num_bins,),dtype = np.int64) for key in self.extremums}
    def update(self,ds:xr.Dataset):
        for key in ds.data_vars:
            self.update_by_key(ds,key)
    def update_by_key(self,ds:xr.Dataset,key:str):
        x = ds[key].values
        x = x[x==x]
        extm = self.extremums[key]
        dx =extm/self.hnbins
        xhat = x/dx
        xhat = np.floor(xhat).astype(np.int64)
        mask = (xhat < self.hnbins)*(-self.hnbins<=xhat)
        xhat = xhat[mask]
        xhat = xhat + self.hnbins
        values, cts = np.unique(xhat, return_counts=True)
        values = values.astype(np.int64)
        ckey = self.counts[key]
        ckey[values] += cts
        self.counts[key] = ckey
    def normalized_name(self,key):
        return f'{key}_normalized'#f'({key} - mean({key}))/std({key})'
    def get_density_xarray(self,):
        densities = {key:val/val.sum() for key,val in self.counts.items()}
        data_vars = {key + '_density': (self.normalized_name(key),val) for key,val in densities.items()}
        coords = {self.normalized_name(key):np.linspace(-self.extremums[key],self.extremums[key],self.hnbins*2+1)
                  for key in densities}
        coords = {key:(axis[1:] + axis[:-1])/2 for key,axis in coords.items()}
        return xr.Dataset(data_vars = data_vars,coords = coords)