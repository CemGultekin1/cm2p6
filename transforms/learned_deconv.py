

from transforms.coarse_graining import base_transform
import xarray as xr
import numpy as np
import itertools
import matplotlib.pyplot as plt

    
class L2Fit(base_transform):
    def __init__(self, sigma, cds:xr.Dataset,fds:xr.Dataset,degree:int = 4):
        super().__init__(sigma,None,)
        self.dims = 'lat lon ulat ulon tlat tlon'.split()
        self.fine_spread = sigma
        self.coarse_spread = 5
        shpfun = lambda x: (x*2+1,x*2+1)
        self.fine_shape = shpfun(self.fine_spread)
        self.coarse_shape = shpfun(self.coarse_spread)
        self.cds,self.fds = cds,fds
        self.degree = degree
        if self.cds is not None:
            self.varnames= [key for key in self.cds.data_vars.keys() if 'S'+key in self.cds.data_vars]
            keys = list(self.cds.data_vars.keys())
            for var in keys:
                if var not in self.varnames:
                    self.cds = self.cds.drop(var)
            keys = list(self.fds.data_vars.keys())
            for var in keys:
                if var not in self.varnames:
                    self.fds = self.fds.drop(var)

        self.hann_window = self.fine_grid_hann_window()
        self.xx = None
        self.xy = None
        self.xy_moment = None
        self.solution = None
    def get_geo_features(self,cds):
        self.latlon_feats = []
        latfeats = cds.lat.values.reshape([-1,1])/90*2*np.pi
        lonfeats = cds.lon.values.reshape([1,-1])/180*2*np.pi
        fourier_components = []
        
        for i,j in itertools.product(range(self.degree),range(self.degree)):
            feats = latfeats*i + lonfeats*j
            fourier_components.append(np.cos(feats).flatten())
            fourier_components.append(np.sin(feats).flatten())
        return fourier_components
        
    def center(self,cds,fine_index,spread):  
        dims = [dim for dim in self.dims if dim in cds.coords]
        fine_index = [fine_index[dim] for dim in dims]
        coords = [cds[dim].data[ic] for dim,ic in zip(dims,fine_index)]
        mcs = {dim :  len(cds[dim])//2  - ic  for  dim,ic in zip(dims,fine_index)}
        cds = cds.roll(shifts = mcs,roll_coords=True)
        fine_index = [np.argmin(np.abs(cds[dim].values - c)) for dim,c in zip(dims,coords)]
        isel_dict = {
            dim : slice(ic - spread,ic + spread+1 ) for dim,ic in zip(dims,fine_index)
        }
        cds = cds.isel(**isel_dict)
        return cds
    def fine_grid_center(self,itime,ilat,ilon,idepth):
        fine_index =  self.multiply_index([i*self.sigma for i in (ilat,ilon)])
        sel_dict = {key:val for key,val in dict(time = itime).items() if key in self.fds.coords}
        fds = self.center(self.fds.isel(**sel_dict),fine_index,self.fine_spread)
        return fds
    def multiply_index(self,index):
        lat,lon = index
        return {
            key: lat if 'lat' in key else lon for key in self.dims
        }
    def coarse_grid_center(self,itime,ilat,ilon,idepth):
        fine_index = self.multiply_index([i for i in (ilat,ilon)])
        sel_dict = {key:val for key,val in dict(time = itime).items() if key in self.cds.coords}
        cds = self.center(self.cds.isel(**sel_dict),fine_index,self.coarse_spread)
        return cds
    def fine_grid_hann_window(self,):
        N = self.fine_spread*2
        n = np.arange(N + 1)
        hann = np.sin(n/N*np.pi)**2
        return np.outer(hann,hann)
    def prepare_outputs(self,fds:xr.DataArray):
        return fds.values * self.hann_window
    def feature_vector(self,cds:xr.Dataset,feats):
        x = cds.values.flatten()
        x = np.concatenate([x*feat for feat in feats])
        return x
    def add(self,xx,varname):
        xx_ =self.__getattribute__(varname)
        if  xx_ is None:
            xx_ = xx
        else:
            xx_ += xx
        self.__setattr__(varname,xx_)
    def collect(self,it,ilat,ilon,idepth):
        cds = self.coarse_grid_center(it,ilat,ilon,idepth)
        fds = self.fine_grid_center(it,ilat,ilon,idepth)
        cds = cds.fillna(0)
        fds = fds.fillna(0)
        feats = self.get_geo_features(cds)
        prods = {}
        for varname in self.varnames:
            x = self.feature_vector(cds[varname],feats)
            y = self.prepare_outputs(fds[varname])
            xx = np.outer(x,x)
            xy = np.outer(x,y)
            prods[varname] = (xx,xy)
        return prods
    def eval(self,varname,it,ilat,ilon,idepth):
        cds = self.coarse_grid_center(it,ilat,ilon,idepth)
        fds = self.fine_grid_center(it,ilat,ilon,idepth)
        cds = cds.fillna(0)
        fds = fds.fillna(0)
        feats = self.get_geo_features(cds)
        x = self.feature_vector(cds[varname],feats)
        y = self.prepare_outputs(fds[varname])
        pred = x.reshape([1,-1]) @ self.solution
        return pred.reshape(self.fine_shape),y.reshape(self.fine_shape)
    def solve(self,):
        halfymat =np.linalg.cholesky(self.xx + 1e-3*np.eye(self.xx.shape[0]))
        self.solution = np.linalg.solve(halfymat.T,np.linalg.solve(halfymat,self.xy))
        return self.solution
def compute_section_limits(len_axes,section):
    m = np.product(len_axes)
    dm  = m//section[1]
    m1 = np.minimum(dm*section[1],m)
    m0 = dm*section[0]
    return (m0,m1)
class Eval(L2Fit):
    def __init__(self, sigma, solution, degree: int = 4):
        super().__init__(sigma, None, None, degree)
        self.solution = solution
    def eval(self,cds,limits = None):
        nlats,nlons = len(cds.lat),len(cds.lon)
        
        coeffs = self.solution.values
        if limits is not None:
            latvals = np.arange(limits[0],limits[1])
            lonvals = np.arange(limits[2],limits[3])
            nlats = limits[1] - limits[0]
            nlons = limits[3] - limits[2]
        else:
            latvals = np.arange(nlats)
            lonvals = np.arange(nlons)
        fds = np.zeros(((nlats + 2)*self.sigma,(nlons+2)*self.sigma))
        
        k =0 
        
        for ilat,ilon in itertools.product(latvals,lonvals):
            fine_index = self.multiply_index([i for i in (ilat,ilon)])
            subcds = self.center(cds,fine_index,self.coarse_spread).fillna(0)
            feats = self.get_geo_features(subcds)
            x = self.feature_vector(subcds,feats)
            y = x.reshape([1,-1])@coeffs
            y = y.reshape(self.fine_shape)
            ilat0 = ilat - latvals[0]
            ilon0 = ilon - lonvals[0]
            
            if k < 5:
                vmax = np.amax(np.abs(y))
                plt.imshow(y,cmap = 'bwr',vmax = vmax,vmin = -vmax)
                plt.savefig(f'subfig_{k}.png')
                plt.close()
                k+=1
            latslice = slice((ilat0 +1)*self.sigma - self.fine_spread,(ilat0+1)*self.sigma + self.fine_spread+1)
            lonslice = slice((ilon0+1)*self.sigma - self.fine_spread,(ilon0+1)*self.sigma + self.fine_spread+1)
            fds[latslice,lonslice] += y
        return fds
class SectionedL2Fit(L2Fit):
    def __init__(self, sigma, cds: xr.Dataset, fds: xr.Dataset,  section = (0,1),degree: int = 8):
        super().__init__(sigma, cds, fds, degree)
        nt = 100
        
        dt = len(self.cds.time)//nt
        
        
        self.cds = self.cds.isel(time = np.arange(nt)*dt)
        self.fds = self.fds.isel(time = np.arange(nt)*dt)
        
        self.len_axes = {}
        for dim in 'lat lon depth time'.split():
            try:
                n = len(self.cds[dim]) 
            except: 
                n = 1
            self.len_axes[dim] = n
        self.limits = compute_section_limits(list(self.len_axes.values()),section)
        self.length = self.limits[1] - self.limits[0]
        
        num_dims = degree**2*np.prod(self.coarse_shape)
        print(f'self.length,num_dims = {self.length,num_dims}')
        # max_length =int(np.ceil(num_dims*100/section[1]))        
        # print(f'self.length = {self.length}, needed data = {max_length}, num_dims = {num_dims}, degree = {degree}')
        # self.length = int(np.minimum(self.length,max_length))
    def get_indices(self,i):
        inds = {}
        for key,x in self.len_axes.items():
            inds[key] = i%x
            i = i//x
        return inds
    def __len__(self,):
        return self.length
    def collect(self,lat =0,lon = 0,depth = 0,time = 0):
        prods = super().collect(time,lat,lon,depth)
        ndepth = self.len_axes['depth']
        xx,xy = prods['u']
        coords = dict(
                grid = np.arange(2),\
                depth = np.arange(ndepth),\
                ninputs1 = np.arange(xx.shape[0]),\
                ninputs2 = np.arange(xx.shape[0]),\
                noutputs = np.arange(xy.shape[1]))
        indims = [0,1,2,3]
        outdims = [0,1,3,4]
        shp = [2,ndepth] + list(xx.shape)
        xx_ = np.zeros(shp)
        shp = [2,ndepth] + list(xy.shape)
        xy_ = np.zeros(shp)
        for vn,(xx,xy) in prods.items():
            
            ut_ind = 0 if vn in 'u v'.split() else 1
            xx_[ut_ind,depth] += xx
            xy_[ut_ind,depth] += xy
            
        return coords,(indims,xx_),(outdims,xy_)
    def __getitem__(self,i):
        i = i+self.limits[0]
        return self.collect(**self.get_indices(i))
def main():
    from data.load import load_xr_dataset
    import itertools
    args = '--sigma 4 --filtering gcm'.split()
    fds,_ = load_xr_dataset(args,high_res = True)
    cds,_ = load_xr_dataset(args,high_res = False)
    

    
    
    l2fit = L2Fit(4,cds,fds,degree = 3)
    nlat,nlon = [len(cds[dim]) for dim in 'lat lon'.split()]
    nt = 100
    k = 0
    while k < 100:
    # for it,ilat,ilon in itertools.product(np.arange(nt)*5,range(nlat),range(nlon)):
        it = np.random.randint(nt)*5
        ilat = np.random.randint(nlat)
        ilon = np.random.randint(nlon)
        prods = l2fit.collect(it,ilat,ilon,0)
        xx,xy = prods['u']
        l2fit.add(xx,'xx')
        l2fit.add(xy,'xy')
        print(k,xx.shape,xy.shape)
        k+=1
        if k % 500 == 0:
            l2fit.solve()
    l2fit.solve()
    import matplotlib.pyplot as plt
    k= 0
    while k < 16:
        it = np.random.randint(10,nt+10,)*10
        ilat = np.random.randint(nlat)
        ilon = np.random.randint(nlon)
        pred,ytrue = l2fit.eval('u',it,ilat,ilon,0)        
        if ytrue.sum()==0:
            continue
        fig,axs= plt.subplots(1,2,figsize = (20,10))
        pos = axs[0].imshow(pred,cmap = 'bwr',)
        fig.colorbar(pos,ax= axs[0])
        pos = axs[1].imshow(ytrue,cmap = 'bwr',)
        axs[0].set_title(f'({it},{ilat},{ilon})')
        fig.colorbar(pos,ax= axs[1])
        fig.savefig(f'true_pred_{k}.png')
        plt.close()
        print(f'true_pred_{k}.png')
        k+=1
if __name__ == '__main__':
    main()