import copy
from typing import Dict, List, Tuple
import torch
from data.low_res import SingleDomain
from data.geography import frequency_encoded_latitude
import numpy as np
from data.vars import FIELD_MASK, FIELD_NAMES, FORCING_MASK, FORCING_NAMES, get_var_mask_name
import xarray as xr
from utils.xarray_oper import plot_ds, tonumpydict
def determine_ndoms(*args,**kwargs):
    arglens = [1]
    for i in range(len(args)):
        if isinstance(args[i],list):
            arglens.append(len(args[i]))
    for key,_ in kwargs.items():
        if isinstance(kwargs[key],list):
            arglens.append(len(kwargs[key]))
    return  int(np.amax(arglens))
class MultiDomain(SingleDomain):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.var_grouping = kwargs.pop('var_grouping')
    def get_lat_features(self,lats):
        posdict = self.locate(lats[0],lats[-1],lat = True)
        (n0,_),n = posdict['locs'],posdict["len"]
        slc = slice(n0,len(lats)+n0)
        abslat,signlat = frequency_encoded_latitude(n,self.half_spread*2+1)
        return np.cos(abslat[slc]),np.cos(signlat[slc])
    def append_lat_features(self,outs):
        key = list(outs.keys())[0]
        lats = outs[key].u.lat.values
        abslat,signlat = self.get_lat_features(lats)
        n = len(outs[key].u.lon)
        abslat = abslat.reshape(-1,1)@np.ones((1,n))
        signlat = signlat.reshape(-1,1)@np.ones((1,n))
        latfeats = xr.Dataset(
            data_vars = dict(
                abslat = (["lat","lon"],abslat),
                signlat = (["lat","lon"],signlat),
            ),
            coords = dict(
                lon = outs[key].u.lon,
                lat = outs[key].u.lat
            )
        )
        outs['lat_feats'] = latfeats
        return outs
    


class MultiDomainDataset(MultiDomain):
    def __init__(self,*args,scalars = None,latitude = False,temperature = False,torch_flag = False, **kwargs):
        self.scalars = scalars
        self.latitude = latitude
        self.temperature = temperature
        self.torch_flag = torch_flag
        self.input_kwargs = kwargs
        super().__init__(*args,**kwargs)


    @property
    def sslice(self,):
        return slice(self.half_spread,-self.half_spread)

    def pad(self,data_vars:dict,coords:dict):
        forcing_mask_names = [get_var_mask_name(fn) for fn in  self.forcing_names]
        for name in data_vars.keys():
            dims,vals = data_vars[name]
            if 'lat' not in dims or 'lon' not in dims:
                continue
            pad = (0,0)
            if name in self.forcing_names + forcing_mask_names and self.half_spread>0:
                vrshp = list(vals.shape)
                vals = vals.reshape([-1]+ vrshp[-2:])
                vals =  vals[:,self.sslice,self.sslice]
                vals = vals.reshape(vrshp[:-2] + list(vals.shape[-2:]))
                # print(f'{vrshp}->{vals.shape}' )
            padtuple = (len(vals.shape)-2)*[(0,0)] + [(0,pad[0]),(0,pad[1])]
            vals = np.pad(vals,pad_width = tuple(padtuple),constant_values = np.nan)
            data_vars[name] = (dims,vals)
        
        def pad_coords(coords,slice_flag = False):
            lat = coords['lat']
            pad = 0
            coords['lat_pad'] = pad
            lat = np.pad(lat,pad_width = ((0,pad),),constant_values = 0)
            if slice_flag:
                lat = lat[self.sslice]
            coords['lat'] = lat

            lon = coords['lon']
            pad = 0
            coords['lon_pad'] = pad
            lon = np.pad(lon,pad_width = ((0,pad),),constant_values = 0)
            if slice_flag:
                lon = lon[self.sslice]
            coords['lon'] = lon
            return coords
        
        forcing_coords = pad_coords(copy.deepcopy(coords),slice_flag=self.half_spread>0)
        coords = pad_coords(coords,slice_flag=False)
        
        return data_vars,coords,forcing_coords

    def add_lat_features(self,data_vars,coords):
        lats = coords['lat']
        lons = coords['lon']
        abslat,signlat = self.get_lat_features(lats)
        data_vars['abs_lat'] = (['lat','lon'], abslat.reshape([-1,1]) @ np.ones((1,len(lons))))
        data_vars['sign_lat'] = (['lat','lon'],signlat.reshape([-1,1]) @ np.ones((1,len(lons))))
        return data_vars
    def group_variables(self,data_vars):
        groups = []
        for vargroup in self.var_grouping:
            valdict = {}
            for varname in vargroup:
                if varname not in data_vars:
                    continue
                valdict[varname] = data_vars[varname]
                # for suff in '_mean _std'.split():
                for suff in '_scale '.split():
                    nvarname = varname + suff
                    if nvarname in data_vars:
                        valdict[nvarname] = data_vars[nvarname]
            groups.append(valdict)
        return tuple(groups)

    def group_np_stack(self,vargroups):
        return tuple([self._np_stack(vars) for vars in vargroups])
    def _np_stack(self,vals:Dict[str,Tuple[List[str],np.ndarray]]):
        v = []
        for _,val in vals.values():
            v.append(val)
        if len(v) == 0:
            return np.empty(0)
        else:
            return np.stack(v,axis =0)
    def group_to_torch(self,vargroups):
        return tuple([self._to_torch(vars) for vars in vargroups])
    def _to_torch(self,vals:np.array,dtype = torch.float32):
        # vals = vals[:,300:-280,300:-280]
        return torch.from_numpy(vals).type(dtype)
    def normalize(self,data_vars,coords):
        keys_list = tuple(data_vars.keys())
        for key in keys_list:
            dims,vals = data_vars[key]
            if 'lat' not in dims or 'lon' not in dims:
                continue
            shp = {d:len(coords[d]) for d in dims}
            newdims = {key:None for key in shp}
            if 'lon' in shp:
                shp['lon'] = 1
                newdims.pop('lon')
            if 'lat' in shp:
                shp['lat'] = 1
                newdims.pop('lat')
            shp0 = [shp[key] for key in newdims]
            shp1 = list(shp.values())
            newdims = list(newdims.keys())        
            a = np.ones(shp0)
            if self.scalars is not None:
                if f"{key}_scale" in self.scalars:
                    a = self.scalars[f"{key}_scale"].values
                    a = a.reshape(shp0)

            
            if not self.torch_flag:
                data_vars[f"{key}_scale"] = (newdims,a)
                # data_vars[f"{key}_mean"] = (newdims,a)
                # data_vars[f"{key}_std"] = (newdims,b)
            # vals = (vals - a.reshape(shp1))/b.reshape(shp1)
            vals = vals/a.reshape(shp1)
            data_vars[key] = (dims,vals)
        return data_vars,coords

    def mask(self,data_vars):
        keys_list = tuple(data_vars.keys())
        for key in keys_list:
            dims,f = data_vars[key]
            if not ('lat' in dims and 'lon' in dims):
                continue
            mask = f==f
            f[~mask] = 0
            mask_found = False
            
            for group,group_mask in zip([self.field_names,self.forcing_names],[FIELD_MASK,FORCING_MASK]):
                if key in group:
                    mask = data_vars[group_mask][1]
                    mask_found =True
                    break
            if mask_found:
                varmask = get_var_mask_name(key)
                data_vars[varmask] = (dims,mask)
                if not self.torch_flag:
                    data_vars[f"{varmask}_normalization"] = (['normalization'],np.array([0,1]))
        return data_vars
    def __getitem__(self, i):
        ds = super().__getitem__(i)
        # print(f'MultiDomainDataset - {[f"{key}-{val.shape}" for key,val in ds.coords.items()]}')
        per_region = []
        requested_boundaries = ([None]*4,) if self.requested_boundaries is None else self.requested_boundaries
        # print(f'requested_boundaries = {requested_boundaries}')
        for lat0,lat1,lon0,lon1 in requested_boundaries:            
            if lat0 is not None:
                subds = ds.sel(lat = slice(lat0,lat1),lon= slice(lon0,lon1))
            else:
                subds = ds
            single_dom_out = self.single_domain(subds)
            if not self.torch_flag:
                return single_dom_out
            per_region.append(single_dom_out)
        cropped_per_region = []
        def get_slice(length: int, length_to: int):
            d_left = max(0, (length - length_to) // 2)
            d_right = d_left + max(0, (length - length_to)) % 2
            return slice(d_left, length - d_right)
        for var_inputs in zip(*per_region):
            shps = []
            for var_in in var_inputs:
                shps.append(np.array(var_in.shape))
            shps = np.stack(shps,axis = 0)
            shps = np.amin(shps,axis =0)
            # shps = np.amax(shps,axis =0)
            group = []
            for var_in in var_inputs:
                slcs = [get_slice(shp,_shp) for shp,_shp in zip(var_in.shape,shps)]
                var_in = var_in[slcs[0],slcs[1],slcs[2]]
                # var_in = var_in[:shps[0],:shps[1],:shps[2]]
                group.append(var_in)                
                # zer =torch.zeros(*shps)
                # shps_ = var_in.shape
                # zer[:shps_[0],:shps_[1],:shps_[2]] = var_in                
                # group.append(zer)
            group = torch.stack(group,dim = 0)
            cropped_per_region.append(group)
        min_gpu_reject_size = 200
        max_shape = np.stack([np.array(group.shape[2:]) for group in cropped_per_region],axis = 0)
        max_shape = np.amax(max_shape,axis = 0)
        pad_shape = np.maximum(min_gpu_reject_size - max_shape,0)
        if True:#np.all(pad_shape == 0) or not torch.cuda.is_available():
            return tuple(cropped_per_region)
        cropped_per_region_ = []
        for group in cropped_per_region:
            shp = group.shape
            padded_shape = np.array(shp)
            padded_shape[2:] += pad_shape
            z = torch.zeros(*padded_shape)
            z[:,:,:shp[2],:shp[3]] = group
            cropped_per_region_.append(z)
        return tuple(cropped_per_region_)
        
    def single_domain(self,outs):
        data_vars,coords = tonumpydict(outs)
        # for key,(dim,val) in data_vars.items():
        #     print(f'{key}-{dim}: {val.shape}')
        for ik,iv in self.input_kwargs.items():
            if ik not in coords:
                if np.isscalar(iv) or isinstance(iv,str):
                    coords[ik] = np.array([iv])
        # print('\n'.join([f'{key} : {type(coords[key])}' for key in coords]))
        # print('\n'.join([f'{key} : {data_vars[key][1].shape}' for key in data_vars]))
        # raise Exception
        if self.latitude:
            data_vars = self.add_lat_features(data_vars,coords)

        data_vars,coords = self.normalize(data_vars,coords)
        data_vars = self.mask(data_vars)
        
        
        data_vars,coords,forcing_coords = self.pad(data_vars,coords)
        
        
        # dropkeys = []
        # for key in data_vars:
        #     if 'normalization' in key or 'scale' in key:
        #         dropkeys.append(key)
        #         continue
        #     if 'S' not in key:
        #         dropkeys.append(key)
        #         continue
        # for dk in dropkeys:
        #     data_vars.pop(dk)
        # selkeys = 'Su Sv Stemp'.split()
        # data_vars = {key:data_vars[key] for key in selkeys}
        # ds = xr.Dataset(data_vars,forcing_coords)        
        # ds = np.log10(np.abs(ds))
        # print(ds)
        # plot_ds(ds,'ds.png',ncols = 1)
        # raise Exception
        
        
        grouped_vars = self.group_variables(data_vars)
        
        if self.torch_flag:
            grouped_vars = self.group_np_stack(grouped_vars)
            return self.group_to_torch(grouped_vars)
        else:
            grouped_vars = list(grouped_vars)
            grouped_vars.append(coords)
            grouped_vars.append(forcing_coords)
            return tuple(grouped_vars)
