import os
from typing import List, Tuple
from data.exceptions import RequestDoesntExist
from data.low_res_dataset import MultiDomainDataset
from data.high_res_dataset import  HighResCm2p6
from data.paths import get_filter_weights_location, get_high_res_data_location, get_high_res_grid_location, get_learned_deconvolution_location, get_low_res_data_location
import copy
from data.vars import FIELD_NAMES, FORCING_NAMES, LATITUDE_NAMES,LSRP_RES_NAMES, get_var_mask_name, rename
from data.scalars import load_scalars
from transforms.gcm_filter_weights import GcmFilterWeights
from transforms.learned_deconv import SectionedDeconvolutionFeatures
from transforms.grids import get_grid_vars
import xarray as xr
from data.coords import  DEPTHS, REGIONS, TIMES
from utils.arguments import options, replace_params
import numpy as np
import torch

def flattened_grid_spacing(ds:xr.Dataset):
    xu = ds.xu_ocean.values
    yu = ds.yu_ocean.values
    earth_radius_in_meters = 6.3710072e6
    
    dxu = xu[1:] - xu[:-1]
    dxu = np.median(dxu)*np.ones(xu.shape)
    dxu = dxu*earth_radius_in_meters/180*np.pi
    dxu = dxu.reshape([1,-1])*np.ones((len(yu),1))
    
    dyu = yu[1:] - yu[:-1]
    dyu = np.concatenate([dyu,dyu[-1:]])
    dyu = dyu*earth_radius_in_meters/180*np.pi
    dyu = dyu.reshape([-1,1])*np.ones((1,len(xu)))
    ds['dxu'] = (('yu_ocean','xu_ocean'),dxu)
    ds['dyu'] = (('yu_ocean','xu_ocean'),dyu)
    ds['dxt'] = (('yt_ocean','xt_ocean'),dxu)
    ds['dyt'] = (('yt_ocean','xt_ocean'),dyu)
    return ds
    
    


def load_grid(ds:xr.Dataset,spacing = 'asis'):
    if spacing != 'asis':
        grid_loc =  flattened_grid_spacing(ds)
    else:
        grid_loc = xr.open_dataset(get_high_res_grid_location())
    passkeys = ['dxu','dyu','dxt','dyt']#,'area_t',]
    st_oceans = ds.st_ocean.values
    for key in passkeys:
        val = grid_loc[key]
        val = val.expand_dims(dim = {'st_ocean': st_oceans},axis = 0)
        ds[key] = val
    return ds

def load_filter_weights(args,utgrid='u',svd0213 = False):
    path = get_filter_weights_location(args,preliminary=False,utgrid=utgrid,svd0213=svd0213)
    fw = xr.open_dataset(path)
    return fw

def load_learned_deconv(args,):
    path = get_learned_deconvolution_location(args,preliminary = False)
    fw = xr.open_dataset(path)
    return fw



def pass_geo_grid(ds,sigma):
    grid = xr.open_dataset(get_high_res_grid_location())
    lon = grid.xt_ocean.values
    lat = grid.yt_ocean.values
    ilon = np.argsort(((lon + 180 )%360 - 180))
    lon = lon[ilon]
    geolon =grid.geolon_t.values[:,ilon]
    geolat = grid.geolat_t.values[:,ilon]
    geoc = xr.Dataset(
        data_vars = dict(
            geolon = (['lat','lon'],geolon),
            geolat = (['lat','lon'],geolat),
        ),
        coords = dict(
            lat = lat,lon = lon
        )
    )
    if sigma > 1:
        geoc = geoc.coarsen(dim = {'lat':sigma,'lon':sigma},boundary = 'trim').mean().load()
    rlat = ds['lat'].values
    rlon = ds['lon'].values
    # geoc = geoc.sel(lat = slice(rlat[0],rlat[-1]),lon = slice(rlon[0],rlon[-1]))
    geolon = geoc.geolon.values
    geolat = geoc.geolat.values
    def match_shape(g,r,axis):
        df = len(r) - g.shape[axis]
        ldf = df//2
        rdf = df - ldf
        if df>0:
            padding_ = (rdf,ldf)
            padding = [(0,0),(0,0)]
            padding[axis] = padding_
            return np.pad(g,padding,'wrap')
        elif df<0:
            ldf = -ldf
            rdf = -rdf
            return g[ldf:-rdf]
        else:
            return g

    ds = ds.assign_coords(
        dict(
            geolon = (['lat','lon'],(geolon + 180 ) % 360 - 180),
            geolat = (['lat','lon'],geolat)))    
    return ds
        


def load_xr_dataset(args,high_res = None):
    runargs,_ = options(args,'run')
    high_res = high_res if high_res is not None else runargs.mode == 'data'
    if high_res:
        data_address = get_high_res_data_location(args)
    else:
        data_address = get_low_res_data_location(args)
    if not os.path.exists(data_address):
        print('RequestDoesntExist\t',data_address)
        raise RequestDoesntExist
    ds_zarr= xr.open_zarr(data_address,consolidated=False )
    if high_res:  
        if runargs.depth == 0:
            ds_zarr = ds_zarr.expand_dims(dim = {'st_ocean':[0]},axis = 1)
        ds_zarr = load_grid(ds_zarr,spacing = runargs.spacing)
    if runargs.sanity:
        ds_zarr = ds_zarr.isel(time = slice(0,1))
    ds_zarr,scs=  preprocess_dataset(args,ds_zarr,high_res)
    return ds_zarr,scs

def get_var_grouping(args)-> Tuple[Tuple[List[str],...],Tuple[List[str],...]]:
    runprms,_=options(args,key = "run")
    fields,forcings = FIELD_NAMES.copy(),FORCING_NAMES.copy()
    lsrp_res = LSRP_RES_NAMES.copy()
    
    if not runprms.temperature and not runprms.mode == 'scalars':
        fields = fields[:2]
        forcings = forcings[:2]
        lsrp_res = lsrp_res[:2]
    if runprms.latitude:
        fields.extend(LATITUDE_NAMES)
    varnames = [fields]
    forcingmask_names = []
   
    fieldmasks = [get_var_mask_name(f) for f in fields]
    fieldmask_names = [fieldmasks]

    forcingmasks = [get_var_mask_name(f) for f in forcings]
    lsrpforcingmasks = [get_var_mask_name(f) for f in lsrp_res] 
    if runprms.mode == 'scalars':
        varnames[0].extend(forcings + lsrp_res)
        forcingmask_names.append(fieldmasks)
        forcingmask_names[0].extend(forcingmasks + lsrpforcingmasks)
    elif runprms.lsrp>0:
        if runprms.mode != 'train':
            varnames.append(forcings + lsrp_res)
            forcingmask_names.append(forcingmasks + lsrpforcingmasks)
        else:
            varnames.append(lsrp_res)
            forcingmask_names.append(lsrpforcingmasks)
    else:
        varnames.append(forcings)
        forcingmask_names.append(forcingmasks)
    if runprms.mode == 'view':
        varnames.extend(fieldmask_names)
    varnames.extend(forcingmask_names)
    # print('len(varnames)',len(varnames))
    

    for i in range(len(varnames)):
        varnames[i] = tuple(varnames[i])
    varnames = tuple(varnames)
    return varnames

def dataset_arguments(args,**kwargs_):
    

    prms,_=options(args,key = "data")
    runprms,_=options(args,key = "run")
    
    
    ds_zarr,scalars = load_xr_dataset(args)
    if runprms.mode == 'train':
        boundaries = REGIONS[prms.domain]
    else:
        boundaries = REGIONS['global']
        
    kwargs = ['lsrp','latitude','temperature','section','interior','filtering','wet_mask_threshold']
    kwargs = {key:runprms.__dict__[key] for key in kwargs}
    kwargs['boundaries'] = boundaries
    kwargs['scalars'] = scalars
    kwargs['coarse_grain_needed'] = runprms.mode == "data"

    for key,val in kwargs_.items():
        kwargs[key] = val
    def isarray(x):
        return isinstance(x,list) or isinstance(x,tuple)
    for key,var in kwargs.items():
        if isarray(var):
            if isarray(var[0]):
                for i in range(len(var)):
                    var[i] = tuple(var[i])
                kwargs[key] = list(var)
            else:
                kwargs[key] = tuple(var)
    kwargs['var_grouping'] = get_var_grouping(args)
    args = (ds_zarr,prms.sigma)
    return args,kwargs

class Dataset(torch.utils.data.Dataset):
    mdm:MultiDomainDataset
    def __init__(self,mdm:MultiDomainDataset):
        self.mdm = mdm
    def __len__(self,):
        return len(self.mdm)
    def __getitem__(self,i):
        outs =  self.mdm[i]
        return self.mdm.outs2numpydict(outs)

def load_lowres_dataset(args,**kwargs)->List[MultiDomainDataset]:
    _args,_kwargs = dataset_arguments(args,**kwargs)
    ds = MultiDomainDataset(*_args, **_kwargs)
    dsets = populate_dataset(ds,**kwargs)
    return dsets

def load_highres_dataset(args,**kwargs)->HighResCm2p6:
    _args,_kwargs = dataset_arguments(args,**kwargs)
    ds = HighResCm2p6(*_args, **_kwargs)
    return (ds,)

class TorchDatasetWrap(torch.utils.data.Dataset):
    def __init__(self,mdm):
        self.mdm = mdm
    def __len__(self,):
        return self.mdm.__len__()
    def __getitem__(self,i):
        return self.mdm[i]

def get_data(args,torch_flag = False,data_loaders = True,**kwargs):
    ns,_ = options(args,key = "run")
    if ns.mode != "data":
        dsets = load_lowres_dataset(args,torch_flag = torch_flag,**kwargs)
    else:
        dsets = load_highres_dataset(args,torch_flag = torch_flag,**kwargs)

    if data_loaders:
        minibatch = ns.minibatch//len(REGIONS[ns.domain])
        if ns.mode != "train":
            minibatch = None
        params={'batch_size':minibatch,\
            'shuffle': ns.mode in ["train","view"] or kwargs.get('shuffle',False),\
            'num_workers':ns.num_workers,\
            'prefetch_factor':ns.prefetch_factor,\
            'collate_fn':collate_fn,\
            'pin_memory':True}
        if not torch_flag:
            params.pop('collate_fn')
        torchdsets = (TorchDatasetWrap(dset_) for dset_ in dsets)
        return [torch.utils.data.DataLoader(tset_, **params) for tset_ in torchdsets]
    else:
        return dsets
    


def get_data_(args,torch_flag = False,data_loaders = True,**kwargs):
    ns,_ = options(args,key = "run")
    if ns.mode != "data":
        dsets = load_lowres_dataset(args,torch_flag = torch_flag,**kwargs)
    else:
        dsets = load_highres_dataset(args,torch_flag = torch_flag,**kwargs)

    if data_loaders:
        minibatch = ns.minibatch//len(REGIONS[ns.domain])
        if ns.mode != "train":
            minibatch = None
        params={'batch_size':minibatch,\
            'shuffle': False,\
            'num_workers':ns.num_workers,\
            'prefetch_factor':ns.prefetch_factor,\
            'collate_fn':collate_fn,\
            'pin_memory':True}
        if not torch_flag:
            params.pop('collate_fn')
        torchdsets = (TorchDatasetWrap(dset_) for dset_ in dsets)
        return [torch.utils.data.DataLoader(tset_, **params) for tset_ in torchdsets]
    else:
        return dsets

def collate_fn(samples):
    nb = len(samples)
    nt = len(samples[0])
    chns = [[] for _ in range(nt)]
    for i in range(nb):
        smpl = samples[i]
        for j in range(nt):
            chns[j].append(smpl[j])
    for j in range(nt):
        chns[j] = torch.concatenate(chns[j],dim = 0)
    return chns
def get_filter_weights_generator(args,data_loaders = True,):
    ns,_ = options(args,key = "run")
    assert ns.filtering == 'gcm'
    assert ns.mode == 'data'
    ds_zarr,_ = load_xr_dataset(args)
    grids = get_grid_vars(ds_zarr.isel(time = 0))

    dsets =[GcmFilterWeights(ns.sigma,grid,section = ns.section) for grid in grids]
    if data_loaders:
        minibatch = None
        params={'batch_size':minibatch,\
            'shuffle': False,\
            'num_workers':ns.num_workers,\
            'prefetch_factor':ns.prefetch_factor,\
            'collate_fn':collate_fn}
        torchdsets = [TorchDatasetWrap(dset_)  for dset_ in dsets]
        return [torch.utils.data.DataLoader(torchdset, **params) for torchdset in torchdsets]
    else:
        return dsets
    
def get_deconvolution_generator(args,data_loaders = True,):
    ns,_ = options(args,key = "run")
    assert ns.filtering == 'gcm'
    assert ns.mode == 'data'
    fds,_ = load_xr_dataset(args,high_res=True)
    cds,_ = load_xr_dataset(args,high_res=False)
    dsets = [SectionedDeconvolutionFeatures(ns.sigma,cds.copy(),fds.copy(),section = ns.section,\
                spatial_encoding_degree=5,coarse_spread=10,\
                correlation_spread=3,correlation_distance=2,\
                correlation_spatial_encoding_degree=2)]
    if data_loaders:
        params={'batch_size':None,\
            'shuffle': True,
            'num_workers':ns.num_workers,\
            'prefetch_factor':ns.prefetch_factor}
        torchdsets = [TorchDatasetWrap(dset_)  for dset_ in dsets]
        return [torch.utils.data.DataLoader(torchdset, **params) for torchdset in torchdsets]
    else:
        return dsets

def get_wet_mask_location(datargs):
    datargs = replace_params(datargs,'filtering','gaussian','co2','False')
    data_address = get_low_res_data_location(datargs,silent=True)    
    return data_address.replace('gaussian','wet_mask')
    
def load_wet_mask(datargs):
    wet_mask_location = get_wet_mask_location(datargs)
    # print(wet_mask_location)
    return xr.open_zarr(wet_mask_location)

def populate_dataset(dataset:MultiDomainDataset,groups = ("train","validation"),**kwargs):
    datasets = []
    for group in groups:
        t0,t1 = TIMES[group]
        datasets.append(dataset.set_time_constraint(t0,t1))
    return tuple(datasets)
def get_time_values(deep):
    if deep:
        return load_xr_dataset('--mode train --depth 5'.split())[0].time.values
    return load_xr_dataset('--mode train --depth 0'.split())[0].time.values

def preprocess_dataset(args,ds:xr.Dataset,high_res_flag:bool ):
    prms,_ = options(args,key = "run")
    if high_res_flag:
        ds = rename(ds)
    coord_names = list(ds.coords.keys())

    def add_co2(ds,prms):
        if prms.co2:
            ds['co2'] = [0.01]
        else:
            ds['co2'] = [0.]
        ds = ds.isel(co2 = 0)
        return ds
    ds = add_co2(ds,prms)
    scs = load_scalars(args)
    
    if prms.depth > 1e-3:
        if 'depth' not in coord_names:
            raise RequestDoesntExist
        if high_res_flag:
            depthvals_=ds.coords['depth'].values
            inds = [np.argmin(np.abs(depthvals_ - d )) for d in DEPTHS if d>1e-3]
            ds = ds.isel(depth = inds)
        else:
            depthvals_=ds.coords['depth'].values
            ind = np.argmin(np.abs(depthvals_ - prms.depth ))
            ds = ds.isel(depth = ind)
            if prms.mode != 'scalars':
                scs = scs.sel(depth = prms.depth,method = 'nearest')
            if np.abs(ds.depth.values-prms.depth)>1:
                print(f'requested depth {prms.depth},\t existing depth = {ds.depth.values}')
                raise RequestDoesntExist
    else:        
        if prms.mode != 'data':
            ds['depth'] = [0]
            ds = ds.isel(depth = 0)
            if prms.mode != 'scalars':
                scs = scs.isel(depth = 0)
    
    

    if prms.mode in ['train','eval','view'] and 'tr_depth' in ds.coords:
        depthval = ds.depth.values
        trd = ds.tr_depth.values
        tr_ind = np.argmin(np.abs(depthval - trd))
        if np.abs(trd[tr_ind] - depthval)>1:
            raise RequestDoesntExist
        ds = ds.isel(tr_depth = tr_ind)
    
    if prms.mode != 'scalars' and scs is not None and not high_res_flag :
        if 'tr_depth' in scs:
            depthval = ds.depth.values
            trd = scs.tr_depth.values
            tr_ind = np.argmin(np.abs(depthval - trd))
            if np.abs(trd[tr_ind] - depthval)>1:
                raise RequestDoesntExist
            scs = scs.isel(tr_depth = tr_ind)
    if not high_res_flag:
        wet_mask = load_wet_mask(args)        
        depth = ds.depth.values.item()
        wet_mask = wet_mask.sel(depth = depth)
        existing_masks = 'interior_wet_mask wet_density'.split()
        for emask in existing_masks:
            if emask not in ds.data_vars:
                continue
            ds = ds.drop(emask)                
        ds = xr.merge([ds,wet_mask])
    return ds,scs