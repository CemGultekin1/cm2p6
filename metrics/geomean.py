import itertools
from typing import List, Tuple
from data.load import get_data
from data.low_res import SingleDomain
from metrics.modmet import MergeMetrics
from data.coords import DEPTHS,SIGMAS
from metrics.moments import moments_metrics_reduction
import xarray as xr
from utils.xarray import is_empty_xr, select_coords_by_extremum, select_coords_by_value,cat, shape_dict,plot_ds,drop_unused_coords
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
        return  is_empty_xr(self.wet_mask)
    def __eq__(self, __o: 'WetMask') -> bool:
        return __o.sigma == self.sigma and  self.stencil == __o.stencil

class WetMaskCollector:
    def __init__(self,) -> None:
        self.datasets = dict()
        self.masks :List[WetMask]= []
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
        # #---------------------------For Debugging---------------------------#
        # if len(self.masks) > 0:
        #     return self.masks[0].wet_mask
        # #-------------------------------------------------------------------#
        if wetmask in self.masks:
            return self.masks[self.masks.index(wetmask)].wet_mask      
        wms = {}
        for depth in DEPTHS: 
            ds = self.get_dataset(sigma,depth)
            wms[depth]= ds.get_mask(wetmask.stencil)
        wms = cat(wms,'depth')
        wetmask.wet_mask = wms
        self.masks.append(wetmask)
        return wetmask.wet_mask


    
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
    def reduce_moments_metrics(self,stencil :int= 0):
        metrics = self.metrics.copy()
        wetmask = self.get_mask(stencil = stencil)
        wetmask = select_coords_by_extremum(wetmask,metrics.coords,'lat lon'.split())
        wetmask = select_coords_by_value(wetmask,metrics.coords,'depth')        
        metrics = xr.where(wetmask,metrics,np.nan)    
        metrics = moments_metrics_reduction(metrics,dim = 'lat lon'.split())        
        return metrics
    def latlon_reduct(self,stencil :int= 0):
        self.metrics = self.reduce_moments_metrics(stencil=stencil)
    def filtering_name_fix(self,):
        if 'filtering' not in self.metrics.coords:
            return 
        fls = self.metrics['filtering'].values
        legal_terms = 'gaussian gcm'.split()
        legal_terms_in_num = [sum([ord(s) for s in filtering]) for filtering in legal_terms]
        nonlegal_terms = [fl for fl in fls if fl not in legal_terms]
        if not bool(nonlegal_terms):
            return
        
        nonlegal_terms = [fl for fl in fls if fl not in legal_terms_in_num]
        if bool(nonlegal_terms):
            model_coords,_ = self.get_coords_dict()
            depth = model_coords['depth']
            sur2 = self.metrics.sel(co2 = 0,depth = depth,method='nearest').Su_r2.values
            true_filtering = model_coords['filtering']                
            legal_terms = np.array([true_filtering,legal_terms[1 - legal_terms.index(true_filtering)]])
            suri = np.argsort(sur2)[::-1]
            legal_terms = legal_terms[suri]
            self.metrics = self.metrics.assign_coords(filtering = legal_terms)
            print(f'\t\t{fls}->{legal_terms}')
        else:
            legal_terms = np.array([legal_terms[legal_terms_in_num.index(fl)] for fl in fls])
            self.metrics = self.metrics.assign_coords(filtering = legal_terms)
            print(f'\t\t{fls}->{legal_terms}')
            
class VariableWetMaskedMetrics(WetMaskedMetrics):
    def __init__(self, modelargs: List[str], wc: WetMaskCollector,stencils:List[int]) -> None:
        super().__init__(modelargs, wc)
        self.stencils = stencils
    def latlon_reduct(self,):
        metrics = {}
        for stencil in self.stencils:
            metrics[stencil] = self.reduce_moments_metrics(stencil=stencil)
        self.metrics = cat(metrics,'stencil')        
        self.metrics = drop_unused_coords(self.metrics)
