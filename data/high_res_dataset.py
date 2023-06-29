from typing import Callable, List
from utils.xarray import concat,  tonumpydict
import xarray as xr
from transforms.grids import get_grid_vars, ugrid2tgrid_interpolation
from transforms.subgrid_forcing import BaseLSRPSubgridForcing, filtering_classes
import numpy as np

class HighResCm2p6:
    def __init__(self,ds:xr.Dataset,sigma,*args,section = [0,1],**kwargs):
        self.ndepth = len(ds.depth)
        self.ntime = len(ds.time)
        self.per_depth :List[HighResCm2p6perDepth]= []
        for i in range(self.ndepth):
            self.per_depth.append(HighResCm2p6perDepth(ds.isel(depth = [i],),sigma,*args,section=section,**kwargs))
        
        self.ds = ds
    def __len__(self,):
        return self.ntime    
    def __getitem__(self,itime:int):
        data_vars_ = {}
        coords_ = {}
        def cat_datavars(d,new_d):
            for key,(dims,val) in new_d.items():
                if key not in d:
                    d[key] = new_d[key]
                    continue
                i = dims.index('depth')
                dims0,val0 = d[key]
                val0 = np.concatenate([val0,val],axis= i)
                d[key] = dims0,val0
            return d
        def cat_coords(d,new_d,sel:List[str] = ['depth']):
            for key,val in new_d.items():
                if key not in d:
                    d[key] = val
                    continue
                if key not in sel:
                    continue
                d[key] = np.unique(np.concatenate([d[key],val]))
            return d
        for idepth in range(self.ndepth):
            data_vars,coords = self.per_depth[idepth][itime]
            data_vars_ = cat_datavars(data_vars_,data_vars)
            coords_ = cat_coords(coords_,coords)
        return  data_vars_,coords_
    
class HighResCm2p6perDepth:
    ds : xr.Dataset
    sigma : int
    half_spread : int
    coarse_grain : Callable
    initiated : bool
    def __init__(self,ds:xr.Dataset,sigma,*args,section = [0,1],**kwargs):
        self.ds = ds.copy()#.isel({f"{prefix}{direction}":slice(1500,1800) for prefix in 'u t'.split() for direction in 'lat lon'.split()})#
        self.sigma = sigma
        self.initiated = False
        self.wet_mask = None
        self._ugrid_subgrid_forcing = None
        self._tgrid_subgrid_forcing = None
        self._grid_interpolation = None
        self.forcing_class = filtering_classes[kwargs.get('filtering')]
        a,b = section
        nt = len(self.ds.time)
        time_secs = np.linspace(0,nt,b+1).astype(int)
        t0 = int(time_secs[a])
        t1 = int(time_secs[a+1])        
        self.ds = self.ds.isel(time = slice(t0,t1))
        self.wet_mask_compute_flag = a == 0

    @property
    def depth(self,):
        return self.ds.depth

    def is_deep(self,):
        return self.depth[0] > 1e-3
    def __len__(self,):
        return len(self.ds.time)*len(self.ds.depth)
    def time_depth_indices(self,i):
        return i
        di = i%len(self.ds.depth)
        ti = i//len(self.ds.depth)
        return ti,di

    def get_hres_dataset(self,i):
        # ti,di = self.time_depth_indices(i)
        # ds = self.ds.isel(time = ti,depth = di) 
        ds = self.ds.isel(time = i)
        # ds = ds.isel(**{f"{k0}{k1}":slice(1000,1960) for k0 in 'u t'.split() for k1 in 'lat lon'.split()})
        return ds

    @property
    def ugrid(self,):
        ds = self.get_hres_dataset(0)
        ugrid,_ = get_grid_vars(ds)
        return ugrid
    @property
    def tgrid(self,):
        ds = self.get_hres_dataset(0)
        _,tgrid = get_grid_vars(ds)
        return tgrid
    @property
    def ugrid_subgrid_forcing(self,)-> BaseLSRPSubgridForcing:
        if self._ugrid_subgrid_forcing is None:
            self._ugrid_subgrid_forcing : BaseLSRPSubgridForcing = self.forcing_class(self.sigma,self.ugrid)
        return self._ugrid_subgrid_forcing
    
    @property
    def tgrid_subgrid_forcing(self,)-> BaseLSRPSubgridForcing:
        if self._tgrid_subgrid_forcing is None:
            self._tgrid_subgrid_forcing :BaseLSRPSubgridForcing =self.forcing_class(self.sigma,self.tgrid)
        return self._tgrid_subgrid_forcing

    @property
    def grid_interpolation(self,):
        if self._grid_interpolation is None:
            self._grid_interpolation = ugrid2tgrid_interpolation(self.ugrid,self.tgrid)
        return self._grid_interpolation

    def _base_get_hres(self,i,):
        ds = self.get_hres_dataset(i)
        u,v,temp = ds.u.load(),ds.v.load(),ds.temp.load()
        u = u.rename(ulat = "lat",ulon = "lon")
        v = v.rename(ulat = "lat",ulon = "lon")
        temp = temp.rename(tlat = "lat",tlon = "lon")
        u.name = 'u'
        v.name = 'v'
        temp.name = 'temp'
        return u.fillna(0),v.fillna(0),temp.fillna(0)
   

    def join_wet_mask(self,mask):
        def drop_time(mask):
            if 'time' in mask.dims:
                mask = mask.isel(time = 0)
            if 'time' in mask.coords:
                mask = mask.drop('time')
            return mask
        mask = drop_time(mask)
        if self.wet_mask is None:
            self.wet_mask = mask
        else:
            self.wet_mask = xr.merge([self.wet_mask,mask])#.wet_mask
    def get_mask(self,):
        def pass_gridvals(tgridval,ugridval):
            for key_ in 'lat lon'.split():
                tgridval[key_] = ugridval[key_]
            return tgridval
        
        ucoarse_wet_density = self.ugrid_subgrid_forcing.wet_mask_generator.coarse_wet_density
        tcoarse_wet_density = self.tgrid_subgrid_forcing.wet_mask_generator.coarse_wet_density
        tcoarse_wet_density = pass_gridvals(tcoarse_wet_density,ucoarse_wet_density)
        coarse_wet_density = (ucoarse_wet_density + tcoarse_wet_density)/2
         
        ucoarse_interior_wet_mask = self.ugrid_subgrid_forcing.wet_mask_generator.coarse_interior_wet_mask
        tcoarse_interior_wet_mask = self.tgrid_subgrid_forcing.wet_mask_generator.coarse_interior_wet_mask
        tcoarse_wet_density = pass_gridvals(tcoarse_interior_wet_mask,ucoarse_interior_wet_mask)
        coarse_interior_wet_mask = ucoarse_interior_wet_mask*tcoarse_interior_wet_mask
        
        coarse_wet_density.name = 'wet_density'
        coarse_interior_wet_mask.name = 'interior_wet_mask'
        return coarse_wet_density.compute(),coarse_interior_wet_mask.compute()
    def get_forcings(self,i):
        u,v,temp = self._base_get_hres(i)
        # print(u)
        # plot_ds(dict(u=u,v=v,),f'get_forcings_uv.png',ncols = 1)
        # plot_ds(dict(temp = temp),f'get_forcings_temp.png',ncols = 1)
        # raise Exception
        ff =  self.fields2forcings(i,u,v,temp)
        return ff
    def fields2forcings(self,i,u,v,temp,ScipyFiltering = False):
        u_t,v_t = self.grid_interpolation(u,v)
        uvars = dict(u=u,v=v)
        tvars = dict(u = u_t, v = v_t,temp = temp,)
        def switch_grid_on_dictionary(ulres):
            ulres['u'],ulres['v'] = self.grid_interpolation(ulres['u'],ulres['v'])

        uforcings,(ucres,ulres),(_,ulres0,uhres0) = self.ugrid_subgrid_forcing(uvars,'u v'.split(),'Su Sv'.split())
        switch_grid_on_dictionary(ulres)
        switch_grid_on_dictionary(ulres0)
        switch_grid_on_dictionary(uhres0)
        tforcings,(tcres,_),_ = self.tgrid_subgrid_forcing(tvars,'temp '.split(),'Stemp '.split(),hres0 = uhres0,lres0 = ulres0,)
        tcres.pop('u')
        tcres.pop('v')
        
        
        uvars = dict(uforcings,**ucres)
        tvars = dict(tforcings,**tcres)
        def pass_gridvals(tgridvaldict,ugridvaldict):
            assert len(ugridvaldict) > 0
            ugridval = list(ugridvaldict.values())[0]
            for key,tgridval in tgridvaldict.items():
                for key_ in 'lat lon'.split():
                    tgridval[key_] = ugridval[key_]
                tgridvaldict[key] = tgridval
            return tgridvaldict
       
        tvars = pass_gridvals(tvars,uvars)
        fvars =  dict(uvars,**tvars)
        fvars = self.expand_dims(i,fvars,time = True)#,depth = True)
        return concat(**fvars)
    
        
    def expand_dims(self,i,fields,time = True):#,depth = True):
        ti = i
        _time = self.ds.time.values[ti]
        dd = dict()
        if time:
            dd['time'] = [_time]
        if isinstance(fields,dict):
            fields = {key:val.expand_dims(dd).compute() for key,val in fields.items()}
        else:
            fields = fields.expand_dims(dd)
        return fields
    def append_mask(self,ds):
        wetmasks = self.get_mask()
        ds = xr.merge([ds] + list(wetmasks))
        return ds
    def __getitem__(self,i):
        ds = self.get_forcings(i,)
        ds = self.append_mask(ds)
        ds = ds.drop('co2')
        nds =  tonumpydict(ds)
        return nds