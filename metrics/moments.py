from typing import List
from metrics.modmet import MergeMetrics, ModelMetric
import numpy as np
from utils.xarray import is_xarray_empty
import xarray as xr



moment_names = {
    (1,0): 'true_mom1',
    (0,1): 'pred_mom1',
    (2,0): 'true_mom2',
    (0,2): 'pred_mom2',
    (1,1): 'cross'
}

class MomentMetrics(ModelMetric):
    def __init__(self,modelargs) -> None:
        super().__init__(modelargs, None)
        self.ntime = 0
    def add2metrics(self,mm:xr.Dataset):
        if is_xarray_empty(self.metrics):
            self.metrics = mm
        else:
            self.metrics += mm
        self.ntime += 1
    @staticmethod
    def generate_moments(pred:xr.Dataset,tr:xr.Dataset):
        evals = {}
        evals[(1,0)] = pred.copy()
        evals[(2,0)] = np.square(pred).copy()
        evals[(0,1)] = tr.copy()
        evals[(0,2)] = np.square(tr).copy()
        evals[(1,1)] = (pred*tr).copy()
        evals = {moment_names[key]:val for key,val in evals.items()}
        for ev,val in evals.items():
            evals[ev] = val.rename({key:f'{key}_{ev}' for key in val.data_vars})
        return  xr.merge(list(evals.values()))



def moments_metrics_reduction(sn, dim = ['lat','lon']):
    reduckwargs = dict(dim = dim)
    dvn = list(sn.data_vars)
    dvn = np.unique([dn.split('_')[0] for dn in dvn])
    xarrs = []
    for key in dvn:
        mms = {}
        for mtuple,mname in moment_names.items():
            mms[mtuple] = sn[f"{key}_{mname}"].copy()
        if len(reduckwargs) > 0:
            nonan = xr.where(np.isnan(mms[(1,0)]),0,1)
            def skipna_average(st):
                return xr.where(np.isnan(st),0,st).inplace_sum(**reduckwargs)/nonan.inplace_sum(**reduckwargs)
            mms = {key:skipna_average(val) for key,val in mms.items()}


        mse = mms[(2,0)] + mms[(0,2)] - 2*mms[(1,1)]
        sc2 = mms[(0,2)] 
        
        pvar = mms[(2,0)] - np.square(mms[(1,0)])
        tvar = mms[(0,2)] - np.square(mms[(0,1)])
        
        pvar = xr.where(pvar <0,np.nan,pvar)
        tvar = xr.where(tvar <0,np.nan,tvar)
        mse = xr.where(mse <0,np.nan,mse)
         
        r2 = 1 - mse/sc2
        correlation = (mms[(1,1)] - mms[(1,0)]*mms[(0,1)])/np.sqrt(tvar*pvar)
        r2.name = f"{key}_r2"
        mse.name = f"{key}_mse"
        sc2.name = f"{key}_sc2"
        correlation.name = f"{key}_corr"
        xarrs.extend([r2,correlation,mse,sc2])
    return xr.merge(xarrs)
