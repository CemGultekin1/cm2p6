import xarray as xr
import numpy as np
from transforms.coarse_graining import base_transform, gcm_filtering, greedy_coarse_grain
class FilterWeightsBase(base_transform):
    def __init__(self, sigma, grid, *args, dims=..., **kwargs):
        super().__init__(sigma, grid, *args, dims=dims, **kwargs)
        self.left_spacing = 2*sigma
        self.right_spacing = 3*sigma + 1
        self.span = self.left_spacing + self.right_spacing
class GcmFilterWeights(FilterWeightsBase):
    def __init__(self,sigma,grid,*args,dims = 'lat lon'.split(),section = (0,1),**kwargs):
        super().__init__(sigma,grid,*args,dims = dims,**kwargs)
        print(f'section = {section}')
        self.coarse_wet_mask = greedy_coarse_grain(sigma,grid).coarse_wet_mask 
        ilats,ilons = np.where(self.coarse_wet_mask.data>0)
        
        i,N = section
        
        n = len(ilats)//N
        n0 = n*i
        n1 = np.minimum(len(ilats),n*(i+1))
        ilats = ilats[n0:n1]
        ilons = ilons[n0:n1]
        self._len = len(ilats)
        print(f'self._len = {self._len}')
        self.indexes = {
            dim : ic for dim,ic in zip(dims,[ilats,ilons])
        }
    def __len__(self,):
        return self._len
    def __getitem__(self,i):
        icoords = [self.indexes[dim][i]*self.sigma for dim in self.dims]
        coords = [self.grid[dim].data[ic] for dim,ic in zip(self.dims,icoords)]
        mcs = {dim :  len(self.grid[dim])//2  - ic  for  dim,ic in zip(self.dims,icoords)}
        subgrid = self.grid.roll(shifts = mcs,roll_coords=True)
        icoords = [np.argmin(np.abs(subgrid[dim].values - c)) for dim,c in zip(self.dims,coords)]

        isel_dict = {
            dim : slice(ic - self.left_spacing,ic + self.right_spacing) for dim,ic in zip(self.dims,icoords)
        }
        subgrid = subgrid.isel(**isel_dict)
        filtering = gcm_filtering(self.sigma,subgrid)
        coarse_graining = greedy_coarse_grain(self.sigma,subgrid)
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
        
        # import matplotlib.pyplot as plt
        # import itertools
        # fig,axs = plt.subplots(5,5,figsize = (60,60))
        # for ii,jj in itertools.product(range(5),range(5)):            
        #     np.log10(c_eye).isel(lat = ii, lon = jj).plot(ax = axs[ii,jj])
        # fig.savefig('comparison.png')
        # raise Exception
        c_eye = c_eye.isel(lat = mc,lon =  mc)
        locs = [self.indexes[dim][i] for dim in self.dims]
        coords = {dim:self.coarse_wet_mask[dim].values for dim in self.dims}
        coords = dict(coords,**{
            'rel_lat':np.arange(span) - self.left_spacing,'rel_lon': np.arange(span) - self.left_spacing
        })
        return coords,locs,c_eye.data.reshape([span,span])
    
