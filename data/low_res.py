import copy
from typing import Tuple
from data.vars import FIELD_MASK, FORCING_MASK
from utils.xarray_oper import no_nan_input_mask, plot_ds
import xarray as xr
import numpy as np
from transforms.grids import bound_grid, fix_grid, larger_longitude_grid


class CM2p6Dataset:
    ds : xr.Dataset
    sigma : int
    half_spread : int
    preboundaries : Tuple[int,...]
    global_hres_coords : Tuple[np.ndarray,...] #3 periods of longitude
    global_lres_coords: Tuple[np.ndarray,...] #3 periods of longitude
    def __init__(self,ds:xr.Dataset,sigma,*args,boundaries = None,half_spread = 0,**kwargs):
        self.ds = ds.copy()
        self.sigma = sigma
        self.half_spread = half_spread
        self.global_hres_coords = [None]*4
        def flatten_tuple_list(l_):
            l = []
            for n in l_:
                l.append(n)
            return l
        self.requested_boundaries = None if isinstance(boundaries,tuple) else boundaries
    
        varnames = kwargs.get('var_grouping',None)
        if varnames is not None:
            self.field_names = flatten_tuple_list(varnames[0])
            self.forcing_names = flatten_tuple_list(varnames[1])
        else:
            self.field_names = None
            self.forcing_names  = None
        
        self.global_lres_coords = self.ds.lat.values,larger_longitude_grid(self.ds.lon.values)
        self.preboundaries = (-90,90,-180,180)
        

    @property
    def depth(self,):
        return self.ds.depth.values
    def set_time_constraint(self,t0,t1):
        x = copy.deepcopy(self)
        nt = len(self.ds.time)
        t0,t1 = np.floor(nt*t0),np.ceil(nt*t1)
        t0 = int(np.maximum(t0,0))
        t1 = int(np.minimum(t1,len(self.ds.time)))
        x.ds = self.ds.isel(time = slice(t0,t1))
        return x
    def ntimes(self,):
        return len(self.ds.time)
    def __len__(self,):
        return len(self.ds.time)

    @property
    def lres_spread(self,):
        return self.half_spread

    def locate(self,*args,lat = True,):
        clat,clon = self.global_lres_coords
        if lat:
            cc = clat
        else:
            cc = clon
        locs = []
        for lat in args:
            locs.append(np.argmin(np.abs(cc - lat)))
        return dict(locs = locs,len = len(cc))

   
class SingleDomain(CM2p6Dataset):
    local_lres_coords :Tuple[np.ndarray,...] # larger lres grid
    final_local_lres_coords : Tuple[np.ndarray,...] # smaller lres grid
    initiated : bool
    all_land : bool
    def __init__(self,*args,apply_mask:bool = True,**kwargs):
        super().__init__(*args,**kwargs)
        self.confine(*self.preboundaries)
        self.initiated = False
        self.all_land = None
        self._wetmask = None
        self._forcingmask = None
        self.interior = kwargs.get('interior')
        self.wet_mask_threshold = kwargs.get('wet_mask_threshold')
        self.apply_mask = apply_mask
        # print(f'self.wet_mask_threshold = {self.wet_mask_threshold}')
        # print(f'self.interior = {self.interior}')

    @property
    def shape(self,):
        clat,clon = self.final_local_lres_coords
        return len(clat), len(clon)
    def set_half_spread(self,addsp):
        self.half_spread = addsp
        self.confine(*self.preboundaries)

    def confine(self,latmin,latmax,lonmin,lonmax):
        ulat,ulon = self.global_lres_coords
        bulat,bulon,lat0,lat1,lon0,lon1 = bound_grid(ulat,ulon,latmin,latmax,lonmin,lonmax,self.lres_spread)
        self.boundaries = {"lat":slice(lat0,lat1),"lon": slice(lon0,lon1)}
        self.final_boundaries = {"lat":slice(lat0,lat1),"lon": slice(lon0,lon1)}
        self.local_lres_coords = bulat,bulon
        self.final_local_lres_coords = bulat,bulon
        
    def fix_grid(self,u, ):
        latlon = self.local_lres_coords
        dims = u.dims
        if 'lat' in dims and 'lon' in dims:
            return fix_grid(u,latlon)#at_name = "ulat",lon_name = "ulon")
        else:
            return u
    @property
    def field_wet_mask(self,):
        if self._wetmask is None:
            ds = self.ds.isel(time =0).load()
            ds = self.get_grid_fixed_lres(ds)
            if self.interior:
                landmask = 1 - ds.interior_wet_mask
            else:
                if 'wet_density' in ds.data_vars:
                    landmask = xr.where(ds.wet_density > self.wet_mask_threshold,0,1)
                else:
                    landmask = None
            for key in ds.data_vars.keys():
                mask_ = np.isnan(ds[key])
                if landmask is None:
                    landmask  = mask_
                else:
                    landmask += mask_
            wetmask = xr.where(landmask > 0,0,1)
            # plot_ds({'wetmask':wetmask},'wetmask1.png',ncols = 1)
            wetmask.name = 'wet_mask'
            if self.requested_boundaries is not None:
                wmask = wetmask.values
                bmask = wmask*0
                lat = wetmask.lat.values
                lon = wetmask.lon.values
                for lat0,lat1,lon0,lon1 in self.requested_boundaries:
                    latmask = (lat >= lat0)*(lat <= lat1)
                    lonmask = (lon >= lon0)*(lon <= lon1)
                    mask = latmask.reshape([-1,1])@lonmask.reshape([1,-1])
                    bmask = mask  + bmask
                bmask = (bmask > 0).astype(float)
                wmask = wmask*bmask
                wetmask = xr.DataArray(
                    data = wmask,
                    dims = ['lat','lon'],
                    coords = {'lat':lat,'lon':lon},
                    name = 'wet_mask'
                )
            
            self._wetmask = wetmask
        return self._wetmask
    @property
    def forcing_wet_mask(self,):
        if self._forcingmask is None:
            if self.half_spread > 0:
                forcing_mask = no_nan_input_mask(self.field_wet_mask,self.half_spread,lambda x: x==0,same_size = True)
                self._forcingmask =  xr.where(forcing_mask==0,1,0)
            else:
                self._forcingmask = self.field_wet_mask
        return self._forcingmask
            
    def __getitem__(self,t):
        ds = self.ds.isel(time =t).load()
        ds = self.get_grid_fixed_lres(ds)
        for key in 'interior_wet_mask wet_mask'.split():
            if key in ds.data_vars.keys():
                ds = ds.drop(key)
        ds[FIELD_MASK] = (['lat','lon'],self.field_wet_mask.data.copy())
        ds[FORCING_MASK] = (['lat','lon'],self.forcing_wet_mask.data.copy())
        return ds

    def get_grid_fixed_lres(self,ds):
        fields = list(ds.data_vars.keys())
        var = []
        for field in fields:
            v = self.fix_grid(ds[field])
            v.name = field
            var.append(v)
        U = xr.merge(var)
        return  U

