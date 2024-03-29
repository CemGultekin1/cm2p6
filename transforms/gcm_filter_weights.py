from utils.xarray_oper import plot_ds
import xarray as xr
import numpy as np
from transforms.coarse_graining import BaseTransform, GcmFiltering, GreedyCoarseGrain
class FilterWeightsBase(BaseTransform):
    def __init__(self, sigma, grid, *args, dims=..., **kwargs):
        super().__init__(sigma, grid, *args, dims=dims, **kwargs)
        self.left_spacing = 2*sigma
        self.right_spacing = 3*sigma + 1
        self.span = self.left_spacing + self.right_spacing
        if grid is not None:
            self.fine_shape = tuple(
                [len(grid.lat),len(grid.lon)]
            )
            self.coarse_shape = tuple(
                [c//self.sigma for c in self.fine_shape]
            )
def continue_values(arr):
    diff = arr[1:]-arr[:-1]
    i = np.argmax(np.abs(diff))+1
    left = arr[:i]
    right = arr[i:]
    if len(right) == 0:
        return arr  - np.mean(arr)
    avgdiff= np.mean(np.concatenate([diff[:i-1],diff[i:]]))
    right = right - arr[i] + arr[i-1]  + avgdiff
    arr = np.concatenate([left,right])
    assert len(np.unique(arr)) == len(arr)
    arr =  arr - np.mean(arr)
    return arr
class GcmFilterWeightsBase(FilterWeightsBase):
    def __init__(self,sigma,grid,*args,dims = 'lat lon'.split(),**kwargs):
        super().__init__(sigma,grid,*args,dims = dims,**kwargs)
        self.coarse_wet_mask = GreedyCoarseGrain(sigma,grid).coarse_wet_mask 
        self.ndepth = len(self.grid.depth)
        self.nlon = len(self.coarse_wet_mask.lon)
        self.nlat = len(self.coarse_wet_mask.lat)
    def get_subgrid(self,idepth,ilat,ilon):
        fine_index = [i*self.sigma for i in (ilat,ilon)]
        coords = [self.grid[dim].data[ic] for dim,ic in zip(self.dims,fine_index)]
        mcs = {dim :  len(self.grid[dim])//2  - ic  for  dim,ic in zip(self.dims,fine_index)}
        subgrid = self.grid.roll(shifts = mcs,roll_coords=True)
        fine_index = [np.argmin(np.abs(subgrid[dim].values - c)) for dim,c in zip(self.dims,coords)]
        isel_dict = {
            dim : slice(ic - self.left_spacing,ic + self.right_spacing) for dim,ic in zip(self.dims,fine_index)
        }
        subgrid = subgrid.isel(**isel_dict,depth = idepth)
        for dim in self.dims:
            subgrid[dim] = continue_values(subgrid[dim].values)
        return subgrid
    def generate_filter(self,subgrid):
        filtering = GcmFiltering(self.sigma,subgrid)
        coarse_graining = GreedyCoarseGrain(self.sigma,subgrid)
        span = self.span
        eye_field = xr.DataArray(
            data = np.eye(span**2).reshape([span,span,span,span]),
            dims = 'c0 c1 lat lon'.split(),
            coords = dict(
                c0 = np.arange(span),c1 = np.arange(span),lat = subgrid.lat,lon = subgrid.lon
            )
        )
        f_eye = filtering(eye_field)
        c_eye = coarse_graining(f_eye)
        mc = len(c_eye.lat)//2
        c_eye = c_eye.isel(lat = mc,lon =  mc)
        c_eye = c_eye.values.reshape([span,span])
        return c_eye
class GcmFilterWeights(GcmFilterWeightsBase):
    def __init__(self,sigma,grid,*args,dims = 'lat lon'.split(),section = (0,1),**kwargs):
        super().__init__(sigma,grid,*args,dims = dims,**kwargs)
        print(f'section = {section}')
        M = self.nlon*self.nlat*self.ndepth
        i,N = section
        n = M//N
        n0 = n*i
        n1 = np.minimum(M,n*(i+1))
        self.indexes = (n0,n1)
        print(f'len(self) = {len(self)}')
    def __len__(self,):
        n0,n1 = self.indexes
        return n1 - n0
    def get_index(self,i:int):
        i = i + self.indexes[0]
        depthi,i1 = i%self.ndepth,i//self.ndepth
        ilat,i2 = i1%self.nlat,i1//self.nlat
        ilon,_ = i2%self.nlon,i2//self.nlon
        return depthi,ilat,ilon
    def __getitem__(self,i):
        # i = np.random.randint(self.indexes[1] - self.indexes[0])
        depth,lat,lon = self.get_index(i)
        subgrid =  self.get_subgrid(depth,lat,lon)
        wet_mask = subgrid.wet_mask.values
        ocean_flag = np.any(wet_mask > 0)
        if ocean_flag:
            #can still have nan values bc the filter doesnt fill the square
            filter_weights = self.generate_filter(subgrid)
            infmap = filter_weights == np.inf
            filter_weights[infmap] = np.nan
            filter_weights[wet_mask == 0] = np.nan
        else:
            filter_weights = np.ones((self.span,self.span))*np.nan
        if np.any(np.isnan(filter_weights)):
            subgrid['wet_mask'] = subgrid.wet_mask.copy()*0 + 1
            wet_filter_weights = self.generate_filter(subgrid)
            assert not np.any(np.isnan(wet_filter_weights))
            assert not np.any(wet_filter_weights == np.inf)
            land_mask = filter_weights != filter_weights
            filter_weights[land_mask] = wet_filter_weights[land_mask]
            # filter_weights = wet_filter_weights
        locs = [depth,lat,lon]
        coords = dict(depth = self.grid.depth.values,\
            **{dim:self.coarse_wet_mask[dim].values for dim in self.dims},\
            **{
            'rel_lat':np.arange(self.span) - self.left_spacing,
            'rel_lon': np.arange(self.span) - self.left_spacing,
        })
        return coords,locs,filter_weights,wet_mask
    
