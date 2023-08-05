import numpy as np
import gcm_filters as gcm
import xarray as xr
from scipy.ndimage import gaussian_filter
class BaseTransform:
    def __init__(self,sigma,grid,*args,dims = 'lat lon'.split(),**kwargs):
        self.sigma = sigma
        self.grid = grid
        self.dims = dims

class PlainCoarseGrain(BaseTransform):
    def __init__(self, *args, **kwargs):
        super().__init__( *args, **kwargs)
        self._coarse_specs = dict({axis : self.sigma for axis in self.dims},boundary = 'trim')
        self.coarse_wet_density = self.grid.wet_mask.coarsen(**self._coarse_specs).mean()
        self.coarse_wet_mask = xr.where(self.coarse_wet_density>0,1,0).compute()
    def __call__(self,x):
        forcing_coarse = x.coarsen(**self._coarse_specs).mean()
        # forcing_coarse = xr.where(self.coarse_wet_mask,forcing_coarse,np.nan)
        return  forcing_coarse

class GreedyCoarseGrain(PlainCoarseGrain):
    def __call__(self,x,greedy = True):
        if greedy:
            forcing_coarse = x.coarsen(**self._coarse_specs).mean()
            landpts = self.coarse_wet_density== 0
            cwd = xr.where(landpts,1, self.coarse_wet_density)
            grd = forcing_coarse/cwd
            grd = xr.where(landpts,np.nan,grd)
            return grd
        else:
            return super().__call__(x)

class Filtering(BaseTransform):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self._norm = None
        self._wet_mask = None
    @property
    def norm(self,):
        if self._norm is None:
            self._norm =  self.get_norm()
        return self._norm
    def get_norm(self,):
        return self.base_filter(self.grid.area)
    def base_filter(self,x):
        return None
    def filter(self,x):
        return None
    def __call__(self,x):
        norm = self.norm 
        return self.filter(x)/norm

class GcmFiltering(Filtering):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        n_steps = kwargs.get('n_steps',None)
        self._gcm = gcm.Filter(**filter_specs(self.sigma,self.grid,area_weighted=False,wet_masked=True,n_steps = n_steps))
    def base_filter(self,x):
        return self._gcm.apply(x,dims = self.dims)
    def filter(self,x):
        return self._gcm.apply(self.grid.area*x,dims =  self.dims)



def stacked_gaussian_filter(data,*args,**kwargs):
    result = np.zeros_like(data)
    for i,data_ in enumerate(data):
        result[i] = gaussian_filter(data_, *args,**kwargs)
    return result

class ScipyFiltering(Filtering):
    def filter(self,x):
        output = xr.apply_ufunc(\
            lambda data: stacked_gaussian_filter(data, self.sigma/2, mode='wrap'),\
            x*self.grid.area,dask='parallelized', output_dtypes=[float, ])
        return output
        
    def base_filter(self,x):
        return xr.apply_ufunc(\
            lambda data: stacked_gaussian_filter(data, self.sigma/2, mode='wrap'),\
            x,dask='parallelized', output_dtypes=[float, ])

class GreedyScipyFiltering(Filtering):
    def filter(self,x):
        # wet_mask = xr.where(np.isnan(x),0,1)
        output = xr.apply_ufunc(\
            lambda data: stacked_gaussian_filter(data, self.sigma/2, mode='wrap'),\
            x*self.grid.area*self.grid.wet_mask,dask='parallelized', output_dtypes=[float, ])
        # output = xr.where(wet_mask,output,np.nan)
        return output
    def base_filter(self,x):
        return xr.apply_ufunc(\
            lambda data: stacked_gaussian_filter(data, self.sigma/2, mode='wrap'),\
            x*self.grid.wet_mask,dask='parallelized', output_dtypes=[float, ])


class WetMask(PlainCoarseGrain):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coarse_interior_wet_mask = self.generate_interior_mask().compute()
    def generate_interior_mask(self,):
        wet_mask = self.grid.wet_mask #xr.where(self.grid.wet_mask == 0,1,0)
        wet_mask =  xr.apply_ufunc(\
            lambda data: stacked_gaussian_filter(data, self.sigma/2, mode='wrap'),\
            wet_mask,dask='parallelized', output_dtypes=[float, ])
        wet_mask = wet_mask*wet_mask.roll({'lat' : 1})*wet_mask.roll({'lon' : 1})
        wet_mask = self(wet_mask)
        wet_mask = wet_mask*wet_mask.roll({'lat' : 1})*wet_mask.roll({'lon' : 1})
        return xr.where(wet_mask<1,0,1)#forcing_land_mask#
        


def filter_specs(sigma,grid, area_weighted = False, wet_masked= False,tripolar = False,n_steps = 16):
    wetmask,area = grid.wet_mask.copy(),grid.area.copy()
    filter_scale = sigma/2*np.sqrt(12)
    dx_min = 1
    grid_vars = dict(area = area,wet_mask = wetmask)
    if tripolar:
        grid_type = gcm.GridType.TRIPOLAR_REGULAR_WITH_LAND_AREA_WEIGHTED
    else:
        grid_type = gcm.GridType.REGULAR_WITH_LAND_AREA_WEIGHTED
    if area_weighted and not wet_masked:
        grid_type = gcm.GridType.REGULAR_WITH_LAND_AREA_WEIGHTED
        grid_vars['wet_mask'] = wetmask*0 + 1
    elif not area_weighted and wet_masked:
        grid_type = gcm.GridType.REGULAR_WITH_LAND
        grid_vars.pop('area')
    elif not area_weighted and not wet_masked:
        grid_type = gcm.GridType.REGULAR
        grid_vars = dict()
    specs = {
        'filter_scale': filter_scale,
        'dx_min': dx_min,
        'filter_shape':gcm.FilterShape.GAUSSIAN,
        'grid_type':grid_type,
        'grid_vars':grid_vars,
    }
    if n_steps is not None:
        specs['n_steps'] = n_steps
    return specs