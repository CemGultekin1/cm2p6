import itertools
from typing import List, Tuple
from data.load import get_data
from data.low_res import SingleDomain
from metrics.modmet import MergeMetrics
from data.coords import DEPTHS,SIGMAS
from metrics.moments import moments_metrics_reduction
import xarray as xr
from utils.xarray import is_xarray_empty
import numpy as np

class CoarseGridInteriorOceanWetMask(SingleDomain):
    def __init__(self, sd:SingleDomain):
        self.__dict__.update(sd.__dict__)
    def get_mask(self,stencil:int):
        hspan = stencil//2
        self.set_half_spread(hspan)
        self._wetmask = None
        self._forcingmask = None
        self.half_spread = hspan
        fwm = self.forcing_wet_mask     
        return fwm 

class WetMask:
    def __init__(self,sigma:int,stencil:int) -> None:
        self.sigma = sigma
        self.stencil = stencil
        self.wet_mask = xr.DataArray()
    def is_empty(self,):
        return  is_xarray_empty(self.wet_mask)
    def __eq__(self, __o: 'WetMask') -> bool:
        return __o.sigma == self.sigma and  self.stencil == __o.stencil

class WetMaskCollector:
    def __init__(self,) -> None:
        self.datasets = dict()
        self.masks = []
    def get_datargs(self,sigma,depth):
        return f'--sigma {sigma} --depth {depth} --domain global'
    def get_dataset(self,sigma,depth):
        datargs = self.get_datargs(sigma,depth)
        if datargs in self.datasets:
            return self.datasets[datargs]
        ds, = get_data(datargs.split(),torch_flag=False,data_loaders=False,groups = ('train',))
        return CoarseGridInteriorOceanWetMask(ds)
    def get_wet_mask(self,sigma,stencil):
        wetmask = WetMask(sigma,stencil)
        if wetmask in self.masks:
            return self.masks[self.masks.index(wetmask)]       
        wms = xr.DataArray()
        for depth in DEPTHS: 
            ds = self.get_dataset(sigma,depth)
            wm= ds.get_mask(wetmask.stencil)
            from utils.xarray import plot_ds
            plot_ds(dict(wetmask = wm),f'wm-{int(depth)}.png')
            wm = wm.expand_dims({'depth':[depth]},axis = 0).reindex(indexers = {'depth':DEPTHS},fill_value = 0)
            if is_xarray_empty(wms):
                wms = wm
            else:
                wms += wm
        wetmask.wet_mask = wms
        self.masks.append(wetmask)
        return wetmask

class WetMaskedMetrics(MergeMetrics):
    def __init__(self, modelargs: List[str],wc:WetMaskCollector) -> None:
        super().__init__(modelargs)
        self.wet_mask_collector = wc
    def get_mask(self,stencil:int = 0,):
        model_coords,metric_coords = self.get_coords_dict()
        model_coords.update(metric_coords)
        stencil = model_coords['stencil'] if stencil == 0 else stencil
        sigma = model_coords['sigma']
        return self.wet_mask_collector.get_wet_mask(sigma,stencil)
    def latlon_reduct(self,):
        wetmask = self.get_mask(stencil=1)
        return wetmask.wet_mask
        metrics = xr.where(wetmask,np.nan,self.metrics)
        mmr = moments_metrics_reduction(metrics,dim = 'lat lon'.split())
        return mmr
    
        
# def co2_nan_expansion(sn:xr.Dataset):
#     if 'co2' not in sn.coords:
#         return sn
#     snco2slcs = []
#     for i in range(len(sn.co2)):
#         snco2slcs.append(sn.isel(co2 = i).drop('co2'))
#     mask = np.isnan(snco2slcs[0])*0
#     for snco2 in snco2slcs:
#         mask = mask + np.isnan(snco2)
#     mask = mask>0
#     for i,snco2 in enumerate(snco2slcs):
#         snco2 = xr.where(mask,np.nan,snco2)
#         snco2 = snco2.expand_dims({'co2':[sn.co2.values[i]]})
#         snco2slcs[i] = snco2
#     snco2slcs = xr.merge(snco2slcs)
#     return snco2slcs
    

