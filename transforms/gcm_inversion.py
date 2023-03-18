from typing import List
from data.load import load_filter_weights, load_xr_dataset
from transforms.gcm_filter_weights import FilterWeightsBase
from transforms.grids import get_grid_vars
from utils.arguments import options
import numpy as np
from utils.xarray import plot_ds
import xarray as xr 
from scipy.ndimage import convolve1d
from transforms.multi_gmres import MultiGmresForFiltering,PseudoInvertibleMatmultBase


class FilterWeightCompression(FilterWeightsBase):
    def __init__(self,sigma,filter_weights) -> None:
        super().__init__(sigma,None,)
        self.filter_weights = filter_weights
        self.ranked_matmult_filter = []
    def svd_components(self,weights_as_column:np.ndarray):
        _,_,vh = np.linalg.svd(weights_as_column,full_matrices=False)
        # weight_map = u@np.diag(s)
        separated_filters = vh.T
        return separated_filters # weight_map.reshape(self.filter_weights.shape),
    def get_separable_components(self,):
        weightmat = self.filter_weights.data.transpose([0,1,3,2]).reshape([-1,self.span])
        
        spfil_lat = self.svd_components(weightmat,)
        weightmat = weightmat @ spfil_lat
        weightmat = weightmat.reshape(self.filter_weights.shape).transpose([0,1,3,2]).reshape([-1,self.span])
        
        spfil_lon = self.svd_components(weightmat)
        weightmat = weightmat @ spfil_lon

        wmap =  weightmat.reshape(self.filter_weights.shape)
        
        coords = {c:self.filter_weights[c] for c in 'lat lon'.split()}
        rank = np.arange(self.span)
        rel_ind = rank - self.left_spacing
        ds = xr.Dataset(
            data_vars = dict(
                    weight_map = (('lat','lon','lat_degree','lon_degree'),wmap),
                    latitude_filters = (('rel_ind','lat_degree'),spfil_lat),
                    longitude_filters = (('rel_ind','lon_degree'),spfil_lon),
            ),
            coords = dict(coords,**{
                'rel_ind' : rel_ind,
                'lat_degree' : rank,
                'lon_degree' : rank,                
            })
        )
        return ds
    
class GcmInversion(FilterWeightsBase):
    def __init__(self,sigma,grid,filter_weights,rank) -> None:
        super().__init__(sigma,grid,)
        self.filter_weights = filter_weights
        clat,clon = len(filter_weights.lat),len(filter_weights.lon)
        lat_filters =[ Matmult1DFilter( 0,\
            filter_weights.latitude_filters.isel(lat_degree = i,).data,sigma,clat)
                     for i in range(self.span) if i < rank]
        lon_filters =[ Matmult1DFilter( 1,\
            filter_weights.longitude_filters.isel(lon_degree = i,).data,sigma,clon)
                     for i in range(self.span) if i < rank]
        energy = np.square(filter_weights.weight_map).sum(dim = 'lat lon'.split()).values
        sorted_degrees = np.argsort(-energy.flatten())
        sorted_degrees = sorted_degrees[:rank]
        lati,loni = np.unravel_index(sorted_degrees,energy.shape)
        self.ranked_matmult_filter = [
            Matmult2DFilter(filter_weights.weight_map.isel(lat_degree = i,lon_degree = j).data,
                [lat_filters[i],lon_filters[j]],self.sigma
            ) for i,j in zip(lati,loni)
        ]
    def solve(self,cu:xr.DataArray,maxiter :int = 2):
        rhs = cu.fillna(0).data.flatten()
        gmres = MultiGmresForFiltering(self.ranked_matmult_filter,rhs,maxiter = maxiter)
        uopt,_ = gmres.solve()
        uopt = uopt.reshape((len(self.grid.lat),len(self.grid.lon)))
        uopt = xr.DataArray(
            data = uopt,
            dims = self.grid.dims,
            coords = self.grid.coords
        )
        uopt = xr.where(self.grid.wet_mask == 0,np.nan,uopt)
        return uopt
        
class Matmult2DFilter(FilterWeightsBase,PseudoInvertibleMatmultBase):
    def __init__(self,map_weights,matmult_filts:List['Matmult1DFilter'],sigma:int,) -> None:
        FilterWeightsBase.__init__(self,sigma,None)
        PseudoInvertibleMatmultBase.__init__(self,)
        self.matmult_filts = matmult_filts
        self.map_weights = map_weights
    def reshaper(self,x):
        if x.size == self.map_weights.size:
            return x.reshape(self.map_weights.shape)
        else:
            shp =  [s*self.sigma for s in self.map_weights.shape]
            return x.reshape(shp)
    def __call__(self,x,inverse = False):
        
        if inverse:
            mw = np.where(np.abs(self.map_weights) > 1e-5, self.map_weights,np.inf)            
            x = x/mw
        
        x = self.matmult_filts[0](x,inverse = inverse)
        x = self.matmult_filts[1](x,inverse = inverse)
        if not inverse:
            x = self.map_weights*x
        return x
        
class Convolutional1DFilter(FilterWeightsBase):
    def __init__(self,axis:int,filter_weights:np.ndarray,sigma:int,ncoarse:int,filterid:int) -> None:
        super().__init__(sigma,None)
        self.axis = axis
        n = len(filter_weights)
        self.filterid = filterid
        self.forward_filter = filter_weights
        nfine = ncoarse*sigma
        filter_mat = np.zeros((ncoarse,nfine))
        self.nfine = nfine
        self.ncoarse = ncoarse
        empt = np.zeros(nfine)
        empt[:n] = filter_weights
        empt = np.roll(empt,-self.left_spacing)
        
        for i in range(ncoarse):
            filter_mat[i,:] = empt
            empt = np.roll(empt,self.sigma)
        self.filter_mat = filter_mat
        self.filter_mat_inverse = None
        q,r = np.linalg.qr(self.filter_mat.T)
        inverse_mat = (np.linalg.inv(r)@q.T).T
        final_filters = []
        origins = []
        spans = []
        tol = 1e-7
        for f in range(sigma):
            single_filter = inverse_mat[range(f,inverse_mat.shape[0],sigma),:]  
            for i in range(single_filter.shape[0]):
                single_filter[i] = np.roll(single_filter[i],-i)
                
            avgfilt = np.mean(single_filter,axis= 0)
            amax = np.amax(np.abs(avgfilt)) 
            sig = np.abs(avgfilt)/amax> tol
            leftside = np.sum(sig[len(sig)//2:])
            rightside = np.sum(sig[:len(sig)//2])
            span = leftside + rightside
            avgfilt = np.concatenate([avgfilt[-leftside:],avgfilt[:rightside]])
            
            origins.append(leftside)
            final_filters.append(avgfilt)
            spans.append(span)
        self.inverse_filters = final_filters
        self.inverse_filter_origins = origins
        self.inverse_filter_spans = spans
    def __call__(self,x,inverse =False):
        # print(f'\t filter(axis = {self.axis}, id = {self.filterid},inverse = {inverse})')
        if not inverse:
            fx = convolve1d(x, self.forward_filter,axis= self.axis,origin = -((5*self.sigma + 1)//2) + 2*self.sigma, mode='wrap')
            if self.axis == 0:
                return fx[::self.sigma,:]
            else:
                return fx[:,::self.sigma]
        shp = np.array(x.shape,dtype = int)
        shp[self.axis]= shp[self.axis]*self.sigma
            
        hu = np.empty(shp)
        if self.axis == 0:
            for i,(filt,orig) in enumerate(zip(self.inverse_filters,self.inverse_filter_origins)):
                hu[range(i,shp[0],self.sigma),:] = convolve1d(x, filt,axis= self.axis,origin = -(len(filt)//2) + orig , mode='wrap')
        else:
            for i,(filt,orig) in enumerate(zip(self.inverse_filters,self.inverse_filter_origins)):
                hu[:,range(i,shp[1],self.sigma)] = convolve1d(x, filt,axis= self.axis,origin = -(len(filt)//2) + orig , mode='wrap')
        return hu
        
class Matmult1DFilter(FilterWeightsBase):
    def __init__(self,axis:int,filter_weights:np.ndarray,sigma:int,ncoarse:int) -> None:
        super().__init__(sigma,None)
        self.axis = axis
        n = len(filter_weights)
        nfine = ncoarse*sigma
        filter_mat = np.zeros((ncoarse,nfine))
        empt = np.zeros(nfine)
        empt[:n] = filter_weights
        empt = np.roll(empt,-self.left_spacing)
        
        for i in range(ncoarse):
            filter_mat[i,:] = empt
            empt = np.roll(empt,self.sigma)
        self.filter_mat = filter_mat
        q,r = np.linalg.qr(self.filter_mat.T)
        self.filter_mat_inverse = (np.linalg.inv(r)@q.T).T

    def __call__(self,x,inverse =False):
        if inverse:
            mat = self.filter_mat_inverse
        else:
            mat = self.filter_mat
        if self.axis == 0:
            return mat @ x
        else:
            return x @ mat.T

    
def gcm_inversion_test():
    args = '--sigma 4 --filtering gcm --mode data'.split()
    fw = load_filter_weights(args).load()
    datargs,_ = options(args,key = 'data')
    ds,_ = load_xr_dataset(args)
    ugrid,_ = get_grid_vars(ds.isel(time = 0))
    gcminv = GcmInversion(datargs.sigma,ugrid,fw,2)
    cds,_ = load_xr_dataset(args,high_res=False)
    ubar = cds.u.isel(time = 0).load()
    utrue = ds.u.isel(time = 0).load().rename(
        {f'u{dim}':dim for dim in 'lat lon'.split()}
    )
    uopt = gcminv.solve(ubar,maxiter = 8)
    plot_ds(dict(utrue = utrue,usolv = uopt, uerr = np.log10(np.abs(utrue - uopt))),'gcm_inversion.png',ncols = 3)

def main():
    gcm_inversion_test()
    
if __name__ == '__main__':
    main()
