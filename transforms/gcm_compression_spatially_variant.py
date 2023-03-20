from transforms.gcm_filter_weights import FilterWeightsBase
import numpy as np
import xarray as xr 

class FilterWeightSpaceVariantCompression(FilterWeightsBase):
    def __init__(self,sigma,filter_weights) -> None:
        super().__init__(sigma,None,)
        self.filter_weights = filter_weights
        self.ranked_matmult_filter = []
   
    def get_separable_components(self,):
        dims = 'lat lon'.split()
        dims = [len(self.filter_weights[dim]) for dim in dims]
        latdims= dims[0]*self.span
        londims= dims[1]*self.span
        
        weightmat = self.filter_weights.data.transpose([0,2,1,3]).reshape([latdims,londims])
        
        u,s,vh = np.linalg.svd(weightmat,full_matrices = False)
        vconv = u @ np.sqrt(np.diag(s)), np.sqrt(np.diag(s)) @ vh
        vconv = [vconv_.reshape([nl,self.span,-1]) for vconv_,nl in zip(vconv,dims)]
        coords = {c:self.filter_weights[c] for c in 'lat lon'.split()}
        sing = np.arange(len(s))
        rank = np.arange(self.span)
        rel_ind = rank - self.left_spacing
        ds = xr.Dataset(
            data_vars = dict(
                    latitude_filters = (('lat','rel_lat','sing'),vconv[0]),
                    longitude_filters = (('lon','rel_lon','sing'),vconv[1]),
                    filters = (('lat','lon','rel_lat','rel_lon'),self.filter_weights.values),
                    energy = (('sing'),s)
            ),
            coords = dict(coords,**{
                'sing' : sing,
                'rel_lat' : rel_ind,
                'rel_lon' : rel_ind,                
            })
        )
        return ds
class VariantMatmult(FilterWeightsBase):
    def __init__(self, sigma, filter_weights,axis,):
        super().__init__(sigma,None)
        nc = filter_weights.shape[0]
        nf = sigma*nc
        self.axis = axis
        fmat = np.zeros((nc,nf))
        for i in range(nc):
            f0  = np.zeros((nf,))
            f0[:self.span] = filter_weights[i]
            fmat[i] = np.roll(f0,-self.left_spacing + i*sigma)
        self.filter_mat = fmat
        # u,s,vh = np.linalg.svd(fmat,full_matrices = False)
        # sinv = np.diag(1/np.where(s/s[0]<1e-3,np.inf,s))
        # self.filter_mat_inverse = vh.T @ sinv @ u.T
        self.filter_mat_inverse = None
    def __call__(self,x,inverse =False):
        if inverse:
            mat = self.filter_mat_inverse
        else:
            mat = self.filter_mat
        if self.axis == 0:
            return mat @ x
        else:
            return x @ mat.T
class Variant2DMatmult(FilterWeightsBase):
    def __init__(self, sigma, grid, filter_weights,*args, dims=..., rank:int = 2,**kwargs):
        super().__init__(sigma, grid, *args, dims=dims, **kwargs)
        self.filter_weights = filter_weights
        clat,clon = len(filter_weights.lat),len(filter_weights.lon)
        self.fine_shape = (len(grid.lat),len(grid.lon))
        self.coarse_shape =( clat,clon)
        self.rank = rank
        
        self.mats = {}
        for ax,dim in enumerate('latitude_filters longitude_filters'.split()):
            fts = filter_weights[dim]
            for i in range(rank):
                print(f'{dim},\t rank = {i}')
                ft = fts.isel(sing = i)
                self.mats[(ax,i)] = VariantMatmult(sigma,ft.values,ax )
    
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
        x,xr_flag = Variant2DMatmult.xr2np(x)

        if not xr_flag:
            if inverse:
                x = x.reshape(self.coarse_shape)
            else:
                x = x.reshape(self.fine_shape)
        if special >= 0:
            x1 = x.copy()
            xlat = self.mats[(0,special)](x1,inverse=inverse)
            cx = self.mats[(1,special)](xlat,inverse=inverse)
            cx = self.np2xr(cx,xr_flag,fine_grid=inverse)
            if separated:
                return [cx]
            else:
                return cx
            
        if not separated:
            shp = self.fine_shape if inverse else self.coarse_shape
            cx = np.zeros(shp)
            for i in range(self.rank):
                print(f'\t self.mats[(1,{i})]')
                cx += self.mats[(1,i)](self.mats[(0,i)](x.copy(),inverse = inverse),inverse = inverse)
            return self.np2xr(cx,xr_flag,fine_grid=inverse)
        else:
            cxs = []
            for i in range(self.rank):
                cxs.append(self.mats[(1,i)](self.mats[(0,i)](x.copy(),inverse = inverse),inverse = inverse))
            return self.np2xr(cxs,xr_flag,fine_grid=inverse)  
                
        