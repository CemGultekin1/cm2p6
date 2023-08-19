from copy import deepcopy
import logging
from transforms.subgrid_forcing import BaseSubgridForcing
from linear.coarse_graining_inversion import CoarseGrainingInverter,RemoveZeroRows,RowPick,ColPick
from linear.coarse_graining_operators import get_grid
import xarray as xr
import scipy.sparse as sp
import numpy as np
class InverseGcmFiltering(CoarseGrainingInverter):
    def __init__(self, filtering: str = 'gcm', depth: int = 0, sigma: int = 16) -> None:
        super().__init__(filtering, depth, sigma)
        self.grid = get_grid(sigma,depth)
        self.fine_coords = self.grid.coords
        
    def correct_shrinkage(self,):
        if self.rzr is None: # currently mat is not shrunk
            if self.mat.shape[0] > self.qinvmat.shape[1]: # qinv is shrunk                
                rzr = RemoveZeroRows(self.mat)        
                self.qinvmat = rzr.expand_with_zero_rows(self.qinvmat)
                self.qinvmat = rzr.expand_with_zero_rows(self.qinvmat.T).T
            elif self.mat.shape[0] < self.qinvmat.shape[1]:
                raise Exception
        else:
            if self.mat.shape[0] < self.qinvmat.shape[1]: 
                self.mat = self.rzr.expand_with_zero_rows(self.mat)
            elif self.mat.shape[0] == self.qinvmat.shape[1]: 
                self.mat = self.rzr.expand_with_zero_rows(self.mat)
                self.qinvmat = self.rzr.expand_with_zero_rows(self.qinvmat)
                self.qinvmat = self.rzr.expand_with_zero_rows(self.qinvmat.T).T
            else:
                raise Exception
    def __call__(self, u:xr.DataArray,inverse = False) -> xr.DataArray:
        if not inverse:
            return self.forward_model(u)
        else:
            return self.inverse_model(u,self.fine_coords)
    @property
    def nlat(self,):
        return self.grid.lat.size

    @property
    def nlon(self,):
        return self.grid.lon.size
    
    @property
    def cnlat(self,):
        return self.grid.lat.size//self.sigma

    @property
    def cnlon(self,):
        return self.grid.lon.size//self.sigma
    
def from_rectangle_to_matrix_bounds(lats,lons,nlat,nlon):
    lats = lats % nlat
    lons = lons % nlon
    return lats*nlon + lons
    
    
class LocalizableInverseGcmFiltering(InverseGcmFiltering):
    half_span_factor:int = 16
    interior_half_span_factor:int = 10
    def localize_grid(self,cilat:int,cilon:int):
        ilat = cilat*self.sigma        
        ilon = cilon*self.sigma
        clat,clon = self.nlat//2,self.nlon//2
        rolldict = dict(lat = -ilat + clat,\
                lon = -ilon + clon)
        

        
        hspan = self.half_span_factor
        
        bnds = {t:slice(c - self.sigma*hspan,c + self.sigma*hspan )\
            for t,c in zip('lat lon'.split(),[clat,clon])}
        grid = self.grid.roll(**rolldict)
        grid = grid.isel(**bnds)
        return grid
    
    def localize(self,cilat:int,cilon:int):
        self.correct_shrinkage()
        bounds = tuple( np.arange((ci-self.half_span_factor)*self.sigma,\
                    (ci+self.half_span_factor)*self.sigma) \
                        for ci in (cilat,cilon))
        finds = from_rectangle_to_matrix_bounds(\
                    bounds[0].reshape([-1,1]),bounds[1].reshape([1,-1]),\
                    self.nlat,self.nlon).flatten()
        cbounds = tuple(np.arange(ci-self.half_span_factor,\
                        ci+self.half_span_factor)\
                                for ci in (cilat,cilon))
        cinds = from_rectangle_to_matrix_bounds(\
                    cbounds[0].reshape([-1,1]),cbounds[1].reshape([1,-1]),\
                    self.cnlat,self.cnlon).flatten()
        rowp = RowPick(cinds,self.cnlat*self.cnlon)
        colp = ColPick(cinds,self.cnlat*self.cnlon)
        
        fcolp = ColPick(finds,self.nlat*self.nlon)
        
        qinvmat = colp.pick_cols(rowp.pick_rows(self.qinvmat))
        mat = rowp.pick_rows(mat)
        mat = fcolp.pick_cols(mat)
        grid = self.localize_grid(cilat,cilon)
        return LocalInverseGcmFiltering(mat,qinvmat,grid,filtering = self.filtering,depth=self.depth,sigma = self.sigma)
    
class LocalInverseGcmFiltering(LocalizableInverseGcmFiltering):
    def __init__(self,mat:sp.coo_matrix,qinvmat:sp.coo_matrix, grid:xr.Dataset,filtering: str = 'gcm', depth: int = 0, sigma: int = 16) -> None:
        CoarseGrainingInverter.__init__(self,filtering,depth,sigma)
        self.mat = mat
        self.qinvmat = qinvmat
        self.grid = grid
        self.subgrid_forcing = BaseSubgridForcing(self.sigma,self.grid,)
    # def 
        
def main():
    logging.basicConfig(level=logging.INFO,format = '%(asctime)s %(message)s',)
    ligf = LocalizableInverseGcmFiltering(depth = 0,sigma = 16)
    ligf.load()
    ligf.load_quad_inverse()
    
    logging.info(f'ligf.mat.shape = {ligf.mat.shape}')
    logging.info(f'ligf.qinvmat.shape = {ligf.qinvmat.shape}')
    
    ligf.localize(100,103)
    
    logging.info(f'ligf.mat.shape = {ligf.mat.shape}')
    logging.info(f'ligf.qinvmat.shape = {ligf.qinvmat.shape}')
    


if __name__ == '__main__':
    main()
        
        

# class GcmSubgridForcing(BaseLSRPSubgridForcing):
#     filtering_class = GcmFiltering
#     coarse_grain_class = GreedyCoarseGrain
#     inv_filtering = InverseGcmFiltering