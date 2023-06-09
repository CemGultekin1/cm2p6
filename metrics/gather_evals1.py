import itertools
import os
from models.nets.cnn import kernels2spread
from plots.metrics import metrics_dataset
from constants.paths import JOBS, EVALS, all_eval_path
from metrics.mmcoords import ModelMetricCoords, ModelResultsCollection
from utils.xarray import skipna_mean
import xarray as xr
from utils.arguments import args2dict, options
import numpy as np

def get_lsrp_modelid(args):
    runargs,_ = options(args,key = "model")
    lsrp_flag = runargs.lsrp > 0 and runargs.temperature
    if not lsrp_flag:
        return False, None,None
    keys = ['model','sigma']
    vals = [runargs.__getattribute__(key) for key in keys]
    lsrpid = runargs.lsrp - 1
    vals[0] = f'lsrp:{lsrpid}'
    line =' '.join([f'--{k} {v}' for k,v in zip(keys,vals)])
    _,lsrpid = options(line.split(),key = "model")
    return True, lsrpid,line
def turn_to_lsrp_models(lines):
    lsrplines = []
    for i in range(len(lines)):
        line = lines[i]
        lsrp_flag,_,lsrpline = get_lsrp_modelid(line.split())
        if lsrp_flag:
            lsrplines.append(lsrpline)
    lsrplines = np.unique(lsrplines).tolist()
    return lsrplines 

def append_statistics(sn:xr.Dataset,):#coordvals):
    modelev = metrics_dataset(sn.sel(lat = slice(-85,85)),dim = [])
    # print(modelev)
    # raise Exception
    modelev = skipna_mean(modelev,dim = ['lat','lon'])
    # for c,v in coordvals.items():
    #     if c not in modelev.coords:
    #         modelev = modelev.expand_dims(dim = {c:v})
    for key in 'Su Sv Stemp'.split():
        r2key = f"{key}_r2"
        msekey = f"{key}_mse"
        sc2key = f"{key}_sc2"
        if r2key not in modelev.data_vars:
            continue
        modelev[r2key] = 1 - modelev[msekey]/modelev[sc2key]
    return modelev
    # print(modelev.Su_r2.values.item(),1- modelev.Su_mse.values.item()/modelev.Su_sc2.values.item())
    # return modelev
def merge_and_save(stats):
    xr.merge(list(stats.values())).to_netcdf(all_eval_path(),mode = 'w')

def kernel_size_fun(kernels):    
    ks = kernels2spread(kernels)*2 + 1
    return ks
def main():
    root = EVALS
    models = os.path.join(JOBS,'offline_sweep2.txt')
    file1 = open(models, 'r')
    lines = file1.readlines()
    file1.close()
    
    # lines = [lines[0],lines[16],lines[32],lines[48]]


    # lines = turn_to_lsrp_models(lines) + lines  
    
    mrc = ModelResultsCollection()
    for i,line in enumerate(lines):
        _,modelid = options(line.split(),key = 'model')
        snfile = os.path.join(root,modelid + '.nc')
        if not os.path.exists(snfile):
            continue
        try:
            sn = xr.open_dataset(snfile)
        except:
            continue
        print(f'{i}/{len(lines)}')
        # print(sn.isel(co2 = 0,depth = 0,lat = slice(100,103),lon = slice(100,103)))
        metrics = append_statistics(sn,)
        mm = ModelMetricCoords(line.split(),metrics)
        mm.past_coords_to_metric(('filtering',))
        mrc.add_metrics(mm)
    ds = mrc.merged_dataset()
    filename = all_eval_path().replace('.nc','20230710.nc')
    ds.to_netcdf(filename,mode = 'w')
    
def filtering_correction():
    filename = '/scratch/cg3306/climate/outputs/evals/all20230710.nc' #all_eval_path()
    stats = xr.open_dataset(filename)
    # lsrp = stats.isel(model = 1)
    # skipna_mean(lsrp,)
    stats_ = []
    training_filterings = stats.training_filtering.values
    for i in range(len(training_filterings)):
        stats1 = stats.isel(training_filtering = [i])
        maxstats = stats1.max(dim ='filtering',skipna = True)
        minstats = stats1.min(dim ='filtering',skipna = True)
        other_test_filter = xr.where(maxstats==minstats,np.nan,minstats)
        other_filter = training_filterings[1-i]
        this_filter = training_filterings[i]
        other_test_filter = other_test_filter.expand_dims(dim = {'filtering': [other_filter]})
        maxstats = maxstats.expand_dims(dim = {'filtering': [this_filter]})
        stats_.append(maxstats)
        stats_.append(other_test_filter)
    stats = xr.merge(stats_)
    stats.to_netcdf(filename.replace('.nc','_.nc'),mode = 'w')
if __name__=='__main__':
    main()