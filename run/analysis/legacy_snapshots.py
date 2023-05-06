import itertools
import os
import matplotlib.pyplot as plt
from models.load import load_model
from run.analysis.legacy_comparison import get_legacy_args
from constants.paths import LEGACY_PLOTS,LEGACY
from utils.xarray import fromtorchdict2tensor, fromtorchdict,fromtensor
import xarray as xr
from utils.arguments import options
import numpy as np
from utils.arguments import replace_params
from data.load import load_lowres_dataset
import torch
import matplotlib
import sys
def multi_dim_nparray_sort(arr):
    args = np.argsort(arr.flatten())[::-1]
    x = np.unravel_index(args,arr.shape)
    locs = np.stack([np.array(c) for c in zip(*x)],axis = 0)
    return locs
def data_at_time_and_point(time,ilat,ilon,mdd,net,ftype,name):
    fields,forcings,forcing_mask,field_coords,forcing_coords = mdd[time]
    fields_tensor = fromtorchdict2tensor(fields).type(torch.float32)
    bounds = fields_tensor.shape[2:]
    latlon = [ilat,ilon]
    psread = 4*net.spread
    slices = [[x - psread,x + psread] for x in latlon]
    slices = [[np.maximum(0,x[0]), np.minimum(bound,x[1])] for x,bound in zip(slices,bounds)]
    slices = [[x[1] - psread*2 if x[1] == bound else x[0],x[0] + psread*2 if x[0] == 0 else x[1]]  for x,bound in zip(slices,bounds)]
    forcing_slices = [[x[0],x[1] - 2*net.spread] for x in slices]
    slices = [slice(*x) for x in slices]
    forcing_slices = [slice(*x) for x in forcing_slices]
    bound_fields_tensor = fields_tensor[:,:,slices[0],slices[1]]
    with torch.set_grad_enabled(False):
        mean,prec =  net.forward(bound_fields_tensor)
    std = torch.sqrt(1/prec)
    forcings_shape = np.array(fields_tensor.shape[2:]) - 2*net.spread
    
    zmean = torch.zeros(1,2,forcings_shape[0],forcings_shape[1])
    zmean[:,:,forcing_slices[0],forcing_slices[1]] = mean
    
    zstd = torch.zeros(1,2,forcings_shape[0],forcings_shape[1])
    zstd[:,:,forcing_slices[0],forcing_slices[1]] = std
            
    kwargs = dict(contained = '', drop_normalization = True,masking = False,)
    predicted_forcings_mean = fromtensor(zmean,forcings,forcing_coords, forcing_mask,denormalize = True,**kwargs)
    predicted_forcings_std = fromtensor(zstd,forcings,forcing_coords, forcing_mask,denormalize = True,**kwargs)
    true_forcings = fromtorchdict(forcings,forcing_coords,forcing_mask,denormalize = True,**kwargs)
    fwetmask = mdd.fieldwetmask#.isel(lat = slice(net.spread,-net.spread),lon = slice(net.spread,-net.spread))
    fwetmask = fwetmask.sel(
        lat = slice(true_forcings.lat.values[0],true_forcings.lat.values[-1]),
        lon = slice(true_forcings.lon.values[0],true_forcings.lon.values[-1])
    )
    # print('-'*64)
    # print(f'net.spread = {net.spread}')
    # print('-'*64)
    # print(fwetmask)
    # print(true_forcings)
    # print('-'*64)
    # print(fwetmask.lat.values[[0,-1]])
    # print(true_forcings.lat.values[[0,-1]])
    # print('-'*64)
    true_forcings = xr.where(fwetmask,true_forcings,np.nan)
    true_forcings = true_forcings.isel(lat = forcing_slices[0],lon = forcing_slices[1])
    predicted_forcings_mean = predicted_forcings_mean.isel(lat = forcing_slices[0],lon = forcing_slices[1])
    predicted_forcings_std = predicted_forcings_std.isel(lat = forcing_slices[0],lon = forcing_slices[1])
    if ftype == 'std':
        return true_forcings[name],predicted_forcings_std[name]
    return true_forcings[name],predicted_forcings_mean[name]
def get_vmax_vmins(*args,absolute:bool = True):
    vmin,vmax = np.inf,-np.inf
    for arg_ in args:
        if arg_ is None:
            continue
        arg = np.where(np.isnan(arg_),0,arg_)
        vmax_,vmin_ = np.amax(arg).item(),np.amin(arg).item()
        vmax,vmin = np.maximum(vmax,vmax_),np.minimum(vmin,vmin_)
    if absolute:
        vmax = np.maximum(np.abs(vmax),np.abs(vmin))
        vmin = -vmax
    return dict(vmax = vmax,vmin = vmin)
def main():
    args = sys.argv[1:]
    
    # from utils.slurm import read_args
    # from utils.arguments import replace_params
    # args = read_args(22,filename = 'temp_trainjob.txt')
    
    args = replace_params(args,'mode','eval','num_workers','1','disp','25','minibatch','1')
    args_legacy = get_legacy_args(args)
    runargs,_ = options(args,key = "run")

    modelid,_,net,_,_,_,_,runargs=load_model(args)
    _,_,gz21,_,_,_,_,_=load_model(args_legacy)
    gz21.spread = 10
    net.eval()
    gz21.eval()
    net.to("cpu")
    gz21.to("cpu")
    args = replace_params(args,'mode','eval','domain','global','wet_mask_threshold','0.5','interior','False')
    mdd = load_lowres_dataset(args,half_spread = net.spread, apply_mask = False)[0]
    
    args = replace_params(args_legacy,'mode','eval','domain','global','wet_mask_threshold','0.5','interior','False')
    mdd_legacy = load_lowres_dataset(args,half_spread = gz21.spread,apply_mask = False )[0]
    
    
    snfile = os.path.join(LEGACY,modelid + '_.nc')
    if not os.path.exists(snfile):
        print(f'{snfile} not found !')
        return
    sn = xr.open_dataset(snfile).sel(lat = slice(-85,85)).load()
    sn = sn.isel(co2 = 0,depth= 0).drop('co2 depth'.split())
    
    names = "Su Sv".split()
    unames = np.unique([n.split('_')[0] for n in list(sn.data_vars)])
    names = [n for n in names if n in unames]
    ftypes = ['mean','std']
    
    for name,ftype in itertools.product(names,ftypes):
        stdflag = ftype == 'std'
        fn = name + '_' + ftype + '_mse'
        print(fn)
        fnmax = fn + '_max'
        fnargmax = fn + '_argmax'
        args = multi_dim_nparray_sort(sn[fnmax].values)
        nrows = 16
        # print(args[:10])
        indexes = np.unique(args//30,axis = 0,return_index = True)[1]
        indexes = np.sort(indexes)
        args = args[indexes]
        # print(args[:10])
        select_args = args[:nrows]#slice(0,len(args),len(args)//nrows)]
        nrows = select_args.shape[0]
        # select_args = select_args[:nrows]
        ncols = 4
        fig,axs = plt.subplots(nrows,ncols,figsize = (ncols*7,nrows*7))
        fig.tight_layout()
        for ri,(lat,lon) in enumerate(select_args):
            time = sn[fnargmax].isel(lat = lat,lon = lon).values.item()
            trf,netout = data_at_time_and_point(time,lat,lon,mdd,net,ftype,name)
            _,legacy = data_at_time_and_point(time,lat,lon,mdd_legacy,gz21,ftype,name)
            vms = get_vmax_vmins(netout,legacy,None if stdflag else trf,absolute=ftype == 'mean')
            cmap = matplotlib.cm.bwr
            cmap.set_bad(color = 'gray')
            vms.update(dict(cmap = cmap))
            legacy.plot(ax = axs[ri,0],**vms)
            netout.plot(ax = axs[ri,1],**vms)
            diff = np.abs(legacy-netout)
            if stdflag:
                vms.pop('vmin')
                vms.pop('vmax')
            trf.plot(ax =axs[ri,3],**vms)   
            vms.pop('vmin',None)
            vms.pop('vmax',None)   
            diff.plot(ax = axs[ri,2],**vms)
            axs[ri,0].set_title(f'GZ21 {name}-{ftype}')
            axs[ri,1].set_title(f'R4 {name}-{ftype}')
            axs[ri,2].set_title(f'|GZ21-R2| {name}-{ftype}')
            axs[ri,3].set_title(f'true {name}-{ftype}')
            
            for j in range(4):
                axs[ri,j].set_xlabel('')         
                axs[ri,j].set_ylabel('')         
            
        target = os.path.join(LEGACY_PLOTS,modelid + f'_{name}_{ftype}_snapshots.png')
        fig.savefig(target)
        plt.close()
        print(target)



if __name__=='__main__':
    main()