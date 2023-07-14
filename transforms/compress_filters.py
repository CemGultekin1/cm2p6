
from data.load import load_filter_weights, load_xr_dataset
from transforms.gcm_compression_spatially_variant import FilterWeightSpaceVariantCompression,Variant2DMatmult
from transforms.gcm_filter_weights import GcmFilterWeightsBase
import numpy as np
from utils.xarray_oper import plot_ds
import xarray as xr 
from transforms.grids import get_grid_vars
from transforms.coarse_graining import BaseTransform
from utils.arguments import options
import matplotlib.pyplot as plt

class FilterFiller(GcmFilterWeightsBase):
    def __init__(self, sigma, grid, filter_weights,*args, dims=..., **kwargs):
        super().__init__(sigma, grid, *args, dims=dims, **kwargs)
        self.filter_weights = filter_weights.copy()
        self.full_ocean_filters = {}
    def generate_full_ocean_filter(self,lat_index:int,lon_index:int):
        subgrid = self.center((lat_index,lon_index))
        subgrid.wet_mask = subgrid.wet_mask*0 + 1.
        fw = self.generate_filter(subgrid)
        self.full_ocean_filters[lon_index] = fw
        return fw
            
    def __call__(self,i:int,j:int):
        subgrid = self.center((i,j))
        wetm = subgrid.wet_mask
        x = self.filter_weights.isel(lat = i,lon = j)
        x.data = np.where(wetm.data == 0,np.nan,x.data)
    def minimization(self,x:np.ndarray,cost_mat:np.ndarray):
        if not np.any(np.isnan(x)):
            return x
def main():
    sigma = 4
    args = f'--sigma {sigma} --filtering gcm --lsrp 1 --mode data'.split()
    dfw = load_filter_weights(args,utgrid='u').load()
    m = 64
    lonvalue = 110
    
    lonslice = dict(lon = slice(lonvalue*sigma - 2*sigma,lonvalue*sigma + 3*sigma + 1))
    latslice = lambda latv: dict(lat = slice(latv*sigma - 2*sigma,latv*sigma + 3*sigma + 1))
    ds,_ = load_xr_dataset(args,high_res = True)
    ds = ds.isel(**{ut+dim:val for dim,val in lonslice.items() for ut in 'u t'.split()})
    ugrid,_ = get_grid_vars(ds.isel(time = 0))
    filters = {}
    for j in range(m):
        latv = 50 + 5*j
        flt = dfw.filters.isel(lat = latv,lon = lonvalue).drop('lat').drop('lon')
        wm = ugrid.wet_mask.isel(**latslice(latv))
        flt.data = np.where(wm.data == 0,np.nan,flt.data)
        filters[str(j)] = flt
    
    plot_ds(filters,'filters.png',ncols = int(np.sqrt(m)),dims = 'rel_lat rel_lon'.split())

if __name__ == '__main__':
    main()