import numpy as np
import xarray as xr
from utils.xarray import plot_ds

moment_names = {
    (1,0): 'true_mom1',
    (0,1): 'pred_mom1',
    (2,0): 'true_mom2',
    (0,2): 'pred_mom2',
    (1,1): 'cross'
}

def moments_dataset(prd,tr):
    evals = {}
    evals[(1,0)] = prd.copy()
    evals[(2,0)] = np.square(prd).copy()
    evals[(0,1)] = tr.copy()
    evals[(0,2)] = np.square(tr).copy()
    evals[(1,1)] = (prd*tr).copy()
    evals = {moment_names[key]:val for key,val in evals.items()}
    for ev,val in evals.items():
        evals[ev] = val.rename({key:f'{key}_{ev}' for key in val.data_vars})
    return xr.merge(list(evals.values()))
def co2_nan_expansion(sn:xr.Dataset):
    if 'co2' not in sn.coords:
        return sn
    snco2slcs = []
    for i in range(len(sn.co2)):
        snco2slcs.append(sn.isel(co2 = i).drop('co2'))
    mask = np.isnan(snco2slcs[0])*0
    for snco2 in snco2slcs:
        mask = mask + np.isnan(snco2)
    mask = mask>0
    for i,snco2 in enumerate(snco2slcs):
        snco2 = xr.where(mask,np.nan,snco2)
        snco2 = snco2.expand_dims({'co2':[sn.co2.values[i]]})
        snco2slcs[i] = snco2
    snco2slcs = xr.merge(snco2slcs)
    return snco2slcs
    
def metrics_dataset(sn, dim = ['lat','lon']):
    sn = co2_nan_expansion(sn)
    reduckwargs = dict(dim = dim)
    dvn = list(sn.data_vars)
    dvn = np.unique([dn.split('_')[0] for dn in dvn])
    xarrs = []
    for key in dvn:
        moms = {}
        for mtuple,mname in moment_names.items():
            moms[mtuple] = sn[f"{key}_{mname}"].copy()
        if len(reduckwargs) > 0:
            nonan = xr.where(np.isnan(moms[(1,0)]),0,1)
            def skipna_average(st):
                return xr.where(np.isnan(st),0,st).sum(**reduckwargs)/nonan.sum(**reduckwargs)
            moms = {key:skipna_average(val) for key,val in moms.items()}


        mse = moms[(2,0)] + moms[(0,2)] - 2*moms[(1,1)]
        sc2 = moms[(0,2)] 
        
        pvar = moms[(2,0)] - np.square(moms[(1,0)])
        tvar = moms[(0,2)] - np.square(moms[(0,1)])
        
        pvar = xr.where(pvar <0,np.nan,pvar)
        tvar = xr.where(tvar <0,np.nan,tvar)
        mse = xr.where(mse <0,np.nan,mse)
         
        r2 = 1 - mse/sc2
        correlation = (moms[(1,1)] - moms[(1,0)]*moms[(0,1)])/np.sqrt(tvar*pvar)
        r2.name = f"{key}_r2"
        mse.name = f"{key}_mse"
        sc2.name = f"{key}_sc2"
        correlation.name = f"{key}_corr"
        xarrs.extend([r2,correlation,mse,sc2])
    return xr.merge(xarrs)
