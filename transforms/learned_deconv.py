

from transforms.coarse_graining import BaseTransform
import xarray as xr
import numpy as np
import itertools

    
class DeconvolutionFeatures(BaseTransform):
    def __init__(self, sigma, cds:xr.Dataset,fds:xr.Dataset,spatial_encoding_degree:int = 4,correlation_spread:int = 2,coarse_spread : int = 10):
        super().__init__(sigma,None,)
        self.dims = 'lat lon ulat ulon tlat tlon'.split()
        self.fine_spread = sigma
        self.coarse_spread = coarse_spread
        self.cspan = self.coarse_spread*2+1
        self.fspan = self.fine_spread*2+1
        shpfun = lambda x: (x*2+1,x*2+1)
        self.fine_shape = shpfun(self.fine_spread)
        self.coarse_shape = shpfun(self.coarse_spread)
        self.cds,self.fds = cds,fds
        self.spatial_encoding_degree = spatial_encoding_degree
        self.correlation_spread = correlation_spread
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
        scs = self.correlation_spread
        cspan = self.coarse_spread*2+1
        slc = slice(scs,cspan + scs)
        
        latfeats = cds.lat.values[slc].reshape([-1,1])/90*2*np.pi
        lonfeats = cds.lon.values[slc].reshape([1,-1])/180*2*np.pi
        fourier_components = []
        for i,j in itertools.product(range(self.spatial_encoding_degree),range(self.spatial_encoding_degree)):
            feats = latfeats*i + lonfeats*j
            fourier_components.append(np.cos(feats).flatten())
            if i==0 and j== 0:
                continue
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
        cds = self.center(self.cds.isel(**sel_dict),fine_index,self.coarse_spread + self.correlation_spread)
        return cds
    def fine_grid_hann_window(self,):
        N = self.fine_spread*2
        n = np.arange(N + 1)
        hann = np.sin(n/N*np.pi)**2
        return np.outer(hann,hann)
    def prepare_outputs(self,fds:xr.DataArray):
        return fds.values * self.hann_window
    def feature_vector(self,cds:xr.Dataset,feats):
        x = cds.values.squeeze()
        scs = self.correlation_spread
        cspan = self.coarse_spread*2+1
        slc = slice(scs,cspan + scs)
        x1 = x[slc,slc]
       
        xs = [x1.flatten()]
        for di,dj in itertools.product(*[range(scs)]*2):
            x2i = x[di:di + cspan,dj:dj + cspan]*x1
            xs.append(x2i.flatten())
        
        
        x1 = np.concatenate([x1*feat for x1 in xs for feat in feats])
        return x1
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
        halfymat =np.linalg.cholesky(self.xx + 1e-7*np.eye(self.xx.shape[0]))
        self.solution = np.linalg.solve(halfymat.T,np.linalg.solve(halfymat,self.xy))
        return self.solution
def compute_section_limits(len_axes,section):
    m = np.product(len_axes)
    dm  = m//section[1]
    m1 = np.minimum(dm*section[1],m)
    m0 = dm*section[0]
    return (m0,m1)


import torch
class DeconvolutionTransform(DeconvolutionFeatures):
    def __init__(self, sigma, solution, spatial_encoding_degree: int = 4):
        super().__init__(sigma, None, None, spatial_encoding_degree)
        
        self.feature_map = None
        self.num_feats = 2*self.spatial_encoding_degree**2 - 1
        solution = solution.values.reshape([self.num_feats,-1,self.fspan**2]).transpose((2,0,1)).reshape([-1,self.num_feats,self.cspan,self.cspan])
        self.conv1 = torch.nn.Conv2d(solution.shape[0],solution.shape[1],(self.cspan),bias= False,)
        self.conv1.weight.data = torch.from_numpy(solution,).type( torch.float32)
        self.create_aligned_sum_convolution()
    def create_aligned_sum_convolution(self,):
        x = np.zeros((self.sigma,self.sigma,self.fspan,self.fspan,2,2))
        for fi,fj in itertools.product(*[range(self.sigma)]*2):
            ffi,ffj = self.fine_spread + fi,self.fine_spread + fj
            x[fi,fj,ffi,ffj,0,0] = 1
            x[fi,fj,ffi - self.sigma,ffj,1,0] = 1
            x[fi,fj,ffi,ffj- self.sigma,0,1] = 1
            x[fi,fj,ffi- self.sigma,ffj- self.sigma,1,1] = 1
        x = x.reshape([self.sigma**2,self.fspan**2,2,2])
        self.conv2 = torch.nn.Conv2d(self.sigma**2,self.fspan**2,(2),bias= False,)
        self.conv2.weight.data = torch.from_numpy(x).type(torch.float32)
    def effective_filter(self,cds,ilat,ilon):        
        fine_index = self.multiply_index([i for i in (ilat,ilon)])
        subcds = self.center(cds,fine_index,self.coarse_spread).fillna(0)
        feats = self.get_geo_features(subcds)
        feats = np.concatenate(feats,).reshape([self.num_feats,-1,1])
        solv = self.solution.values.reshape([self.num_feats,-1,self.fspan**2])
        eff_filts = np.sum(feats*solv,axis = 0)
        eff_filts = eff_filts.reshape([self.cspan,self.cspan,self.fspan,self.fspan])
        return eff_filts
    def create_feature_maps(self,cds):
        feats = self.get_geo_features(cds.fillna(0))
        feats = [f.reshape(cds.shape) for f in feats]
        feats = np.stack(feats,axis = 0)
        self.feature_map = feats
    def eval(self,cds,):
        x = cds.fillna(0).values.squeeze()
        x = x.reshape([1,x.shape[0],x.shape[1]])
        x = x*self.feature_map
        x = np.concatenate([x[:,-self.coarse_spread:],x,x[:,:self.coarse_spread+1]],axis = 1)
        x = np.concatenate([x[:,:,-self.coarse_spread:],x,x[:,:,:self.coarse_spread+1]],axis = 2)
        x = np.stack([x],axis = 0)
        x = torch.from_numpy(x).type(torch.float32)
        with torch.no_grad():
            y = self.conv1(x)
            y = self.conv2(y)
        y = y.numpy()
        y = y[0]
        nlat,nlon = y.shape[1],y.shape[2]
        y = y.reshape([self.sigma,self.sigma,nlat,nlon]).transpose((2,0,3,1))
        y = y.reshape([self.sigma*nlat,self.sigma*nlon])
        return y
class SectionedDeconvolutionFeatures(DeconvolutionFeatures):
    def __init__(self, sigma, cds: xr.Dataset, fds: xr.Dataset,  section = (0,1),\
                spatial_encoding_degree: int = 7,correlation_spread:int = 2,coarse_spread : int = 10):
        super().__init__(sigma, cds, fds, spatial_encoding_degree,correlation_spread,coarse_spread)
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
        
        num_dims = self.spatial_encoding_degree**2*np.prod(self.coarse_shape)*(self.correlation_spread*2+1)**2
        print(f'self.length,num_dims = {self.length,num_dims}')
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
    

    
    sigma = 4
    deconv = DeconvolutionFeatures(sigma,cds,fds,spatial_encoding_degree = 2,correlation_spread=3)
    nlat,nlon = [len(cds[dim]) for dim in 'lat lon'.split()]
    nt = 100
    k = 0
    while k < 100:
    # for it,ilat,ilon in itertools.product(np.arange(nt)*5,range(nlat),range(nlon)):
        it = np.random.randint(nt)*5
        ilat = np.random.randint(nlat)
        ilon = np.random.randint(nlon)
        prods = deconv.collect(it,ilat,ilon,0)
        xx,xy = prods['u']
        deconv.add(xx,'xx')
        deconv.add(xy,'xy')
        print(k,xx.shape,xy.shape)
        k+=1
        if k % 500 == 0:
            deconv.solve()
    deconv.solve()
    import matplotlib.pyplot as plt
    k= 0
    while k < 16:
        it = np.random.randint(10,nt+10,)*10
        ilat = np.random.randint(nlat)
        ilon = np.random.randint(nlon)
        pred,ytrue = deconv.eval('u',it,ilat,ilon,0)        
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