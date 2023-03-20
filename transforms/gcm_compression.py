from data.load import load_filter_weights
from transforms.gcm_filter_weights import FilterWeightsBase
import numpy as np
import xarray as xr 
from scipy.ndimage import convolve1d

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
    def filter_basis(self,weights_as_column:np.ndarray):
        shifted_gauss = np.zeros(self.span)
        inds = np.arange(-2*self.sigma,2*self.sigma+1)
        shifted_gauss[:4*self.sigma + 1] = np.exp( - inds**2/(self.sigma/2)**2/2)
        shifted_gauss = np.roll(shifted_gauss,-2*self.sigma)
        filter0 = np.zeros(self.span)
        for i in range(self.sigma):
            filter0 += np.roll(shifted_gauss,i)
            
        filter0 = filter0/np.linalg.norm(filter0)
        weights_as_column0 = (weights_as_column @ filter0).reshape([-1,1]) @ (filter0.reshape([1,-1]))
        weights_as_column = weights_as_column - weights_as_column0
        _,_,vh = np.linalg.svd(weights_as_column,full_matrices=False)
        # weight_map = u@np.diag(s)
        separated_filters = vh.T
        separated_filters = np.concatenate([filter0.reshape([-1,1]),separated_filters[:,:-1]],axis = 1)
        return separated_filters
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
                    filters = (('lat','lon','lat_degree','lon_degree'),self.filter_weights.values)
            ),
            coords = dict(coords,**{
                'rel_ind' : rel_ind,
                'lat_degree' : rank,
                'lon_degree' : rank,                
            })
        )
        return ds

    
class Convolutional1DFilter(FilterWeightsBase):
    def __init__(self,axis:int,filter_weights:np.ndarray,sigma:int,ncoarse:int,):
        super().__init__(sigma,None)
        self.axis = axis
        n = len(filter_weights)
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
        # q,r = np.linalg.qr(self.filter_mat.T)
        # self.filter_mat_inverse = (np.linalg.inv(r)@q.T).T
        u,s,vh = np.linalg.svd(self.filter_mat,full_matrices = False)
        sinv = np.diag(1/np.where(s/s[0] < 1e-3,np.inf,s))
        self.filter_mat_inverse = vh.T@sinv@u.T

    def __call__(self,x,inverse =False):
        if inverse:
            mat = self.filter_mat_inverse
        else:
            mat = self.filter_mat
        if self.axis == 0:
            return mat @ x
        else:
            return x @ mat.T
class FilterMapWeighting(FilterWeightsBase):
    def __init__(self,map_weights,sigma:int,):
        FilterWeightsBase.__init__(self,sigma,None)
        self.map_weights = map_weights
        rel = np.abs(self.map_weights)/np.amax(np.abs(self.map_weights))
        self.inv_map_weights = 1/np.where( rel > 1e-3, self.map_weights,np.inf)        
    def __call__(self,x,inverse = False):
        if inverse:
            x = x*self.inv_map_weights
        if not inverse:
            x = self.map_weights*x
        return x 
    

class MultiMatmult2DFilter(FilterWeightsBase):
    filter_1d_class = Matmult1DFilter
    def __init__(self,sigma,grid,filter_weights,rank = np.inf) -> None:
        FilterWeightsBase.__init__(self,sigma,grid,)
        self.filter_weights = filter_weights
        
        clat,clon = len(filter_weights.lat),len(filter_weights.lon)
        self.fine_shape = (len(grid.lat),len(grid.lon))
        self.coarse_shape =( clat,clon)
        lat_filters =[ self.filter_1d_class( 0,\
            filter_weights.latitude_filters.isel(lat_degree = i,).data,sigma,clat)
                     for i in range(self.span)]
        lon_filters =[ self.filter_1d_class( 1,\
            filter_weights.longitude_filters.isel(lon_degree = i,).data,sigma,clon)
                     for i in range(self.span)]
        energy = np.square(filter_weights.weight_map).sum(dim = 'lat lon'.split()).values
        sorted_degrees = np.argsort(-energy.flatten())
        self.rank = int(np.minimum(len(sorted_degrees),rank))
        sorted_degrees = sorted_degrees[:self.rank]
        lati,loni = np.unravel_index(sorted_degrees,energy.shape)
        self.ranked_filter_weighting_map = { (i,j): 
            FilterMapWeighting(filter_weights.weight_map.isel(lat_degree = i,lon_degree = j).values,self.sigma,)\
                for i,j in zip(lati,loni)}
        self.lat_filters = {i: lat_filters[i] for i in np.unique(lati)}
        self.lon_filters = {j: lon_filters[j] for j in np.unique(loni)}
    @staticmethod
    def xr2np(x,):
        if isinstance(x,xr.DataArray):
            return x.fillna(0).values,True
        else:
            return x,False
    def np2xr(self,x,xr_flag,fine_grid = False):
        if not xr_flag:
            return x
        if isinstance(x,list):
            return [self.np2xr(x_,xr_flag,fine_grid=fine_grid) for x_ in x]
        dims = 'lat lon'.split()
        if fine_grid:
            x =  xr.DataArray(
                data = x,
                dims = dims,
                coords = {dim:self.grid[dim] for dim in dims}
            )
            return xr.where(self.grid.wet_mask,x,np.nan)
        
        return xr.DataArray(
                data = x,
                dims = dims,
                coords = {dim: self.filter_weights[dim] for dim in dims}
            )
                
    def __call__(self,x,inverse = False,separated = False,special :int = -1):
        x,xr_flag = MultiMatmult2DFilter.xr2np(x)
        if not xr_flag:
            if inverse:
                x = x.reshape(self.coarse_shape)
            else:
                x = x.reshape(self.fine_shape)
        if special >= 0:
            i,j = list(self.ranked_filter_weighting_map.keys())[special]
            fwmap = self.ranked_filter_weighting_map[(i,j)]
            x1 = x.copy()
            if inverse:
                x1 = fwmap(x1,inverse = inverse)
            xlat = self.lat_filters[i](x1,inverse=inverse)
            cx = self.lon_filters[j](xlat,inverse = inverse)
            if not inverse:
                cx = fwmap(cx,inverse = inverse)
            cx = self.np2xr(cx,xr_flag,fine_grid=inverse)
            if separated:
                return [cx]
            else:
                return cx
                
        if not inverse:
            xclats = {i:latf(x.copy()) for i,latf in self.lat_filters.items()}            
            if not separated:
                cx = np.zeros(self.coarse_shape)
                for (i,j),fwmap in self.ranked_filter_weighting_map.items():
                    cx_ =  self.lon_filters[j](xclats[i].copy())
                    cx += fwmap(cx_,inverse = False)
                return self.np2xr(cx,xr_flag,fine_grid=inverse)
            else:
                cxs = []
                for (i,j),fwmap in self.ranked_filter_weighting_map.items():
                    cx_ =  self.lon_filters[j](xclats[i].copy())
                    cxs.append(fwmap(cx_,inverse = False))
                return self.np2xr(cxs,xr_flag,fine_grid=inverse)
        if separated:
            cx = []
        else:
            cx = np.zeros(self.fine_shape)
        for (i,j),fwmap in self.ranked_filter_weighting_map.items():
            wx = fwmap(x.copy(),inverse = True)
            latfx = self.lat_filters[i](wx,inverse = True)
            fx =  self.lon_filters[j](latfx,inverse =True)
            if separated:
                cx.append(fx)
            else:
                cx += fx
        return self.np2xr(cx,xr_flag,fine_grid=inverse)
    
def main():
    args = '--sigma 4 --filtering gcm --lsrp 1 --mode data'.split()
    fw = load_filter_weights(args,utgrid='u').load()
    import matplotlib.pyplot as plt
    import itertools
    p = 3
    fig,axs = plt.subplots(p,p,figsize = (p*10,p*5))
    for i,j in itertools.product(range(p),range(p)):
        fts= np.log10(np.abs(fw.isel(lat_degree = i,lon_degree = j).weight_map))        
        fts.plot(ax = axs[i,j],cmap = 'bwr')
    fig.savefig('filter_weight_maps.png')

if __name__ == '__main__':
    main()