import itertools
import os
import sys
from data.exceptions import RequestDoesntExist
from plots.metrics import metrics_dataset, moments_dataset
from run.train import Timer
import torch
from data.load import get_data
from data.vars import get_var_mask_name
from models.load import load_model, load_old_model
import matplotlib.pyplot as plt
from utils.arguments import options, populate_data_options
from params import replace_params
from utils.parallel import get_device
from utils.paths import LEGACY
from utils.slurm import flushed_print
import numpy as np
from utils.xarray import fromtensor, fromtorchdict, fromtorchdict2tensor, plot_ds
import xarray as xr


def torch_stack(*dicts):
    dicts = list(dicts)
    groups = [list(d.keys()) for d in dicts]
    cdict = dicts[0]
    for d in dicts[1:]:
        cdict = dict(cdict,**d)
    newdicts = []
    for g in groups:
        newdicts.append(torch.stack([cdict[key]['val'] for key in g],dim = 1))
    return tuple(newdicts)

def mask(outputs,masks):
    for key in outputs:
        m = masks[get_var_mask_name(key)]['val']
        vec = outputs[key]['val']
        mask = m<0.5
        vec[mask] = np.nan
        outputs[key]['val'] = vec
    return outputs

def match(outputs,forcings,):
    outputs = torch.unbind(outputs, dim = 1)
    keys = list(forcings.keys())
    outputdict = {}
    for i,(out,key) in enumerate(zip(outputs,keys)):
        outputdict[key] = {}
        outputdict[key]['val'] = out
    outputdict = pass_other_keys(outputdict,forcings)
    return outputdict

def concat_datasets(x,y):
    for key in y:
        for key_ in y[key]:
            v0 = x[key][key_]
            v1 = y[key][key_]
            x[key][key_] = torch.cat((v0,v1),dim = 0)
    return x
def separate_linsupres(forcings):
    keys = list(forcings.keys())
    nk = len(keys)//2
    true_forcings = {key:forcings[key] for key in keys[:nk]}
    lsrp_res_forcings = {key:forcings[key] for key in keys[nk:]}
    return true_forcings,lsrp_res_forcings

def override_names(d1,d2):
    newdict = {}
    keys1 = list(d1.keys())
    keys2 = list(d2.keys())
    for key1,key2 in zip(keys1,keys2):
        newdict[key2] = d1[key1]
    return newdict
def pass_other_keys(d1,d2,exceptions = ['val']):
    for key in d2:
        for k in d2[key]:
            if k in exceptions:
                continue
            d1[key][k] = d2[key][k]
    return d1

def to_xarray(torchdict,depth):
    data_vars = {
        key : (["depth","lat","lon"] ,torchdict[key]['val'].numpy()) for key in torchdict
    }
    for key in torchdict:
        lat = torchdict[key]["lat"][0,:].numpy()
        lon = torchdict[key]["lon"][0,:].numpy()
        break
    coords = dict(lat = (["lat"],lat),lon = (["lon"],lon),depth = (["depth"],depth))
    return xr.Dataset(data_vars = data_vars,coords = coords)

def err_scale_dataset(mean,truef):
    err = np.square(truef - mean)
    sc2 = np.square(truef)
    names = list(err.data_vars)
    for name in names:
        err = err.rename({name : name+'_mse'})
        sc2 = sc2.rename({name : name+'_sc2'})
    return xr.merge([err,sc2])

def expand_depth(evs,depthval):
    return evs.expand_dims(dims = dict(depth = depthval),axis=0)
def lsrp_pred(respred,tr):
    keys= list(respred.data_vars.keys())
    data_vars = {}
    coords = {key:val for key,val in tr.coords.items()}
    for key in  keys:
        trkey = key.replace('_res','')
        trval = tr[trkey] - tr[key] # true - (true - lsrp) = lsrp
        data_vars[trkey] = (trval.dims,trval.values)
        respred[key] = trval + respred[key]
        respred = respred.rename({key:trkey})
        tr = tr.drop(key)
    lsrp = xr.Dataset(data_vars =data_vars,coords = coords)
    return (respred,lsrp),tr
def update_stats(stats,prd,tr,key):
    stats_ = moments_dataset(prd,tr)
    if key not in stats:
        stats[key] = stats_
    else:
        stats[key] = stats[key] + stats_
    return stats
def get_lsrp_modelid(args):
    runargs,_ = options(args,key = "model")
    lsrp_flag = runargs.lsrp > 0 and runargs.temperature
    if not lsrp_flag:
        return False, None
    keys = ['model','sigma']
    vals = [runargs.__getattribute__(key) for key in keys]
    lsrpid = runargs.lsrp - 1
    vals[0] = f'lsrp:{lsrpid}'
    line =' '.join([f'--{k} {v}' for k,v in zip(keys,vals)])
    _,lsrpid = options(line.split(),key = "model")
    return True, lsrpid

def get_legacy_args(args):
    leg_args = args.copy()
    leg_args = replace_params(leg_args,'gz21','True')
    return leg_args

def main():
    args = sys.argv[1:]
    # args = '--filtering gaussian --num_workers 1 --disp 1 --min_precision 0.024 --interior False --domain four_regions --batchnorm 1 1 1 1 1 1 1 0 --widths 2 128 64 32 32 32 32 32 4 --kernels 5 5 3 3 3 3 3 3 --minibatch 4 --mode eval'.split()
    # from utils.slurm import read_args
    # from params import replace_params
    # args = read_args(1)
    # args = replace_params(args,'mode','eval','num_workers','1','disp','25','minibatch','1')
    args_legacy = get_legacy_args(args)
    runargs,_ = options(args,key = "run")

    modelid,_,net,_,_,_,_,runargs=load_model(args)
    _,_,gz21,_,_,_,_,_=load_model(args_legacy)
    # modelid,net=load_old_model('0')
    device = get_device()
    net.to(device)
    gz21.to(device)
    lsrp_flag, lsrpid = get_lsrp_modelid(args)
    
    kwargs = dict(contained = '' if not lsrp_flag else 'res')
    assert runargs.mode == "eval"
    net.eval()
    gz21.eval()
    multidatargs = populate_data_options(args,non_static_params=[],domain = 'global',interior = False)
    multi_datargs_legacy = populate_data_options(args_legacy,non_static_params=[],domain = 'global',interior = False)
    allstats = {}
    for datargs,datargs_legacy in zip(multidatargs,multi_datargs_legacy):
        try:
            test_generator, = get_data(datargs,half_spread = net.spread, torch_flag = False, data_loaders = True,groups = ('test',))
            test_generator_legacy, = get_data(datargs_legacy,half_spread = net.spread, torch_flag = False, data_loaders = True,groups = ('test',))
        except RequestDoesntExist:
            print('data not found!')
            test_generator = None
        if test_generator is None:
            continue
        nt = 0
        time_ranking = None
        averaged_fields = None
        for (fields,forcings,forcing_mask,_,forcing_coords),(fields_legacy,forcings_legacy,forcing_mask,_,forcing_coords_legacy) in zip(test_generator,test_generator_legacy):            
            fields_tensor = fromtorchdict2tensor(fields).type(torch.float32)
            
            fields_legacy_tensor = fromtorchdict2tensor(fields_legacy).type(torch.float32)
            depth = forcing_coords['depth'].item()
            co2 = forcing_coords['co2'].item()
            kwargs = dict(contained = '' if not lsrp_flag else 'res', \
                expand_dims = {'co2':[co2],'depth':[depth],},\
                drop_normalization = True,
                masking = False,
                )
            if nt ==  0:
                flushed_print(depth,co2)

            with torch.set_grad_enabled(False):
                mean,prec =  net.forward(fields_tensor.to(device))
                mean = mean.to("cpu")
                prec = prec.to("cpu")
                mean_legacy,prec_legacy = gz21.forward(fields_legacy_tensor.to(device))
                mean_legacy = mean_legacy.to("cpu")
                prec_legacy = prec_legacy.to("cpu")
                # fields_legacy_tensor.to("cpu")


            predicted_forcings_mean = fromtensor(mean,forcings,forcing_coords, forcing_mask,denormalize = True,**kwargs)
            predicted_forcings_std = fromtensor(torch.sqrt(1/prec),forcings,forcing_coords, forcing_mask,denormalize = True,**kwargs)
            true_forcings = fromtorchdict(forcings,forcing_coords,forcing_mask,denormalize = True,**kwargs)
            
            
            predicted_forcings_mean_legacy = fromtensor(mean_legacy,forcings_legacy,forcing_coords_legacy, forcing_mask,denormalize = True,**kwargs)
            predicted_forcings_std_legacy = fromtensor(torch.sqrt(1/prec_legacy),forcings_legacy,forcing_coords, forcing_mask,denormalize = True,**kwargs)
            
            if lsrp_flag:
                predicted_forcings_mean,_ = lsrp_pred(predicted_forcings_mean,true_forcings)
                predicted_forcings_mean,_ = predicted_forcings_mean
 
            mean_err = np.square(predicted_forcings_mean_legacy - predicted_forcings_mean).rename({'Su':'Su_mean_mse','Sv':'Sv_mean_mse'})
            std_err = np.square(predicted_forcings_std - predicted_forcings_std_legacy).rename({'Su':'Su_std_mse','Sv':'Sv_std_mse'})
            mean_s2 = np.square(predicted_forcings_mean_legacy).rename({'Su':'Su_mean_sc2','Sv':'Sv_mean_sc2'})
            std_s2 = np.square(predicted_forcings_std_legacy).rename({'Su':'Su_std_sc2','Sv':'Sv_std_sc2'})
            cur_fields = xr.merge([mean_err,std_err,mean_s2,std_s2])
            
            max_track_keys = [key for key in cur_fields.data_vars.keys() if 'sc2' not in key]
            
            if averaged_fields is None:
                averaged_fields = cur_fields.copy()
                
                data_vars = {key + '_max' : averaged_fields[key] for key in max_track_keys}
                data_vars.update(
                    {key + '_argmax' : (averaged_fields[key]*0).astype(np.int64) for key in averaged_fields.data_vars.keys()}
                )
                time_ranking = xr.Dataset(
                    data_vars = data_vars,
                    coords = mean_err.coords,
                )
                
            else:
                averaged_fields += cur_fields
                for key in max_track_keys:
                    _amax = f'{key}_argmax'
                    _max = f'{key}_max'
                    time_ranking[_amax] = xr.where(time_ranking[_max] >= cur_fields[key],time_ranking[_amax],nt)
                    time_ranking[_max] = xr.where(time_ranking[_max] >= cur_fields[key],time_ranking[_max],cur_fields[key])
            
            nt += 1
                            
            if runargs.disp > 0 and nt%runargs.disp==0:
                if nt == 1:
                    plot_ds(cur_fields,'cur_fields.png',ncols = 2)
                avgf = averaged_fields/nt
                savefields = xr.merge([avgf,time_ranking])
                flushed_print(nt)
                filename = os.path.join(LEGACY,modelid+'_.nc')
                if not os.path.exists(LEGACY):
                    os.makedirs(LEGACY)
                savefields.to_netcdf(filename,mode = 'w')
                
        avgf = averaged_fields/nt
        savefields = xr.merge([avgf,time_ranking])
        flushed_print(nt)
        filename = os.path.join(LEGACY,modelid+'_.nc')
        if not os.path.exists(LEGACY):
            os.makedirs(LEGACY)
        savefields.to_netcdf(filename,mode = 'w')


            

            






if __name__=='__main__':
    main()
