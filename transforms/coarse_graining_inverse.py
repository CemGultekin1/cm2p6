import itertools
from transforms.coarse_graining import BaseTransform
import numpy as np
import xarray as xr



def right_inverse_matrix(mat):
    q,r = np.linalg.qr(mat.T)
    return q @ np.linalg.inv(r).T

def side_multip(mat,x,ax):
    if ax == 0:
        return mat @ x
    else:
        return x @ mat.T
    

def filter_weights_1d(sigma):
    fw = filter_weights(sigma)
    fw = fw[:,(5*sigma + 1)//2]
    # fw = fw/sum(fw)
    return fw

def filter_weights(sigma):
    inds = np.arange(-2*sigma,2*sigma+1)
    w = np.exp( - inds**2/(2*(sigma/2)**2))/np.sqrt(2*np.pi*sigma)
    w_ = w.reshape([-1,1])@w.reshape([1,-1])
    w = np.zeros((5*sigma+1,5*sigma+1))
    w[:w_.shape[0],:w_.shape[1]] = w_
    ww = w*0
    for i in itertools.product(range(sigma),range(sigma)):
        wm = np.roll(w,i,axis=(0,1))
        ww += wm
    sww = np.sum(ww)
    ww = ww/sww
    return ww

    

class matmult_1d(BaseTransform):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        area = self.grid.area.values
        n = len(area)
        self.weights =filter_weights_1d(self.sigma)
        filter_mat = np.zeros((n, n))
        m = len(self.weights)//2

        padded_weights = np.zeros(n)
        padded_weights[:2*m + 1] = self.weights
        padded_weights = np.roll(padded_weights,-m)

        for i in range(n):
            filter_mat[i,:] = np.roll(padded_weights, i)*area
        cfm = filter_mat*1
        for j in range(1,self.sigma):
            cfm[:-j] = cfm[:-j] + filter_mat[j:]
        cfm = cfm[::self.sigma]
        cfm = cfm[:n//self.sigma]
        cfm = cfm/np.sum(cfm,axis = 1,keepdims = True)
        
        self._matrix = cfm
        self._right_inv_matrix = right_inverse_matrix(self._matrix)
        self._projection = None
    def project(self,x,ax = 0):
        if self._projection is None:
            self._projection = self._matrix@self._right_inv_matrix
        mat = self._projection
        return side_multip(mat,x,ax)
    def __call__(self,x,ax = 0,inverse = False):
        if inverse:
            mat = self._right_inv_matrix
        else:
            mat = self._matrix
        return side_multip(mat,x,ax)
    
    
from scipy.ndimage import gaussian_filter
def wet_density(wet_mask,area,sigma:int,dims):
    weighting = xr.apply_ufunc(\
            lambda data: gaussian_filter(data, sigma/2, mode='wrap'),\
            wet_mask*area,dask='parallelized', output_dtypes=[float, ])
    weighting = 1/weighting
    weighting = xr.where(wet_mask,weighting,0)
    coarsening_specs = dict({axis : sigma for axis in dims},boundary = 'trim')
    cwet_mask = wet_mask.coarsen(**coarsening_specs).mean()*sigma**2
    cgrain_norm = 1/cwet_mask
    def coarse_isel(i,j):
        return dict({axis : c for axis,c in zip(dims,[i,j])})
    def fine_isel(i,j):
        return dict({axis : slice(c*sigma,(c+1)*sigma) for axis,c in zip(dims,[i,j])})

    ndims = [len(wet_mask[dim]) for dim in dims]
    cndims = [len(cwet_mask[dim]) for dim in dims]
    indices = [np.arange(ndim) for ndim in cndims]

    latitudinal_weights = np.zeros((sigma,cndims[1],ndims[0]))
    longitudinal_weights = np.zeros((sigma,cndims[0],ndims[1]))
    for i,j in itertools.product(indices):
        cisel = coarse_isel(i,j)
        fisel = fine_isel(i,j)
        wmat = weighting.isel(**fisel)
        wmat = wmat * cgrain_norm.isel(**cisel).values.item()
        u,s,vh = np.linalg.svd(wmat)
        u = u @ np.sqrt(s)
        vh = np.sqrt(s) @ vh
        latitudinal_weights[:,j,i*sigma:(i+1)*sigma] = u.T.reshape([sigma,1,sigma])
        longitudinal_weights[:,i,j*sigma:(j+1)*sigma] = vh.reshape([sigma,1,sigma])
    return latitudinal_weights,longitudinal_weights
def coordinatewise_matmultip(lw,sigma):
    inds = np.arange(-2*sigma,2*sigma+1)
    ws = np.zeros((sigma,lw.shape[0]))
    ws[0,:4*sigma + 1] =  np.exp( - inds**2/(2*(sigma/2)**2))/np.sqrt(2*np.pi*sigma)
    ws[0] = np.roll(ws[0],-2*sigma)
    for i in range(1,sigma):
        ws[i] = np.roll(ws[0],i)
    q = np.zeros((lw.shape[0]//sigma,lw.shape[0]))
    for i in range(q.shape[0]):
        sws = lw[i*sigma:(i+1)*sigma]@ws
        sws = np.roll(sws,sigma*i,axis = 1)
        q[i] = sws
    return q


class slicewise_MatmultFiltering_saver(BaseTransform):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.lat_area,self.lon_area = \
            wet_density(self.grid.wet_mask,self.grid.area,self.sigma,self.dims)    
        
class MatmultFiltering(BaseTransform):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        lon_area = self.grid.mean(dim = self.dims[0])
        lat_area = self.grid.mean(dim = self.dims[1])                
        self._lonfilt = matmult_1d(self.sigma,lon_area,**kwargs)
        self._latfilt = matmult_1d(self.sigma,lat_area,**kwargs)
        coarsen_dict = {key : self.sigma for key in self.dims}
        coarsen_dict['boundary'] = 'trim'
        self.coarse_wet_mask = self.grid.wet_mask.coarsen(**coarsen_dict).mean().values
        self.coarse_wet_mask = xr.where(self.coarse_wet_mask >0,1,0)
        self.fine_wet_mask = self.grid.wet_mask
    def np2xr(self,xvv,finegrid :bool = False):
        dims = self.dims
        if finegrid:
            return xr.DataArray(
                data = xvv,
                dims = dims,
                coords = {
                    key : self.grid[key].values for key in dims
                }
            )
        else:
            return xr.DataArray(
                data = xvv,
                dims = dims,
                coords = {
                    key : self.grid[key].coarsen(**{key : self.sigma,'boundary' : 'trim'}).mean().values for key in dims
                }
            )
    def __call__(self,x,inverse = False):
        xv = x.fillna(0).values
        xvv = self._latfilt(self._lonfilt(xv,ax=1,inverse = inverse),ax = 0,inverse = inverse)
        xvv =  self.np2xr(xvv,finegrid=inverse)
        if inverse:
            xvv = xr.where(self.fine_wet_mask,xvv,np.nan)
        # else:
        #     xvv = xr.where(self.coarse_wet_mask,xvv,np.nan)
        return xvv
class matmult_masked_filtering(MatmultFiltering):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.coarse_wet_density = super().__call__(self.grid.wet_mask)
    def __call__(self,x,inverse = False,wet_density = None):
        if wet_density is None:
            wet_density = self.coarse_wet_density
        if inverse:
            x = x*wet_density
        cx = super().__call__(x,inverse = inverse)
        if not inverse:
            cx = cx/wet_density
        return cx


