from copy import deepcopy
import logging
from transforms.coarse_graining import GreedyCoarseGrain, GcmFiltering
from data.load import get_data
import numpy as np
from linear.lincol import LinFun, SparseVecCollection
import sys
from constants.paths import OUTPUTS_PATH
import os
from utils.slurm import flushed_print
import matplotlib.pyplot as plt
import xarray as xr

filtering_class = GcmFiltering
coarse_grain_class =  GreedyCoarseGrain
class BaseFiltering(LinFun):
    def __init__(self,sigma,depth) -> None:
        self.sigma = sigma
        self.depth = depth
        self.indim = 2700*3600
        self.outdim = (2700//sigma)*(3600//sigma)
        
    def post__init__(self,) -> None:
        grid = get_grid(self.sigma,self.depth)
        self.grid = grid
        
        # self.grid['wet_mask'] = self.grid['wet_mask']*0 + 1
        self.indim = grid.lat.size * grid.lon.size
        
        self.outdim = (grid.lat.size//self.sigma )* (grid.lon.size//self.sigma)
        self.inshape = (grid.lat.size,grid.lon.size)
        self.outshape = (grid.lat.size//self.sigma,grid.lon.size//self.sigma)
        self.coarse_graining = coarse_grain_class(self.sigma,self.grid)
        self.wet_inds = np.where(self.grid.wet_mask.values.flatten()>0)[0]
    @property
    def nlat(self,):
        return self.grid.lat.size 
    @property
    def nlon(self,):
        return self.grid.lon.size
    def get_local_operators(self,ilat:int,ilon:int):
        rolldict,slcdict,backrolldict = self.centralization(ilat,ilon)
        grid = self.grid.roll(**rolldict)
        grid = grid.isel(**slcdict)
        filtering = filtering_class(self.sigma,grid)
        return filtering,slcdict,grid,backrolldict
    def picklable_arguments(self,):
        return self.sigma,self.depth
    @classmethod
    def from_picklable_arguments(self,sigma,depth):
        return BaseFiltering(sigma,depth)
    def capitalize(self,*args):
        return tuple((arg//self.sigma)*self.sigma for arg in args)
    def is_trimmed_out(self,ilat,ilon,):
        nlat = self.grid.lat.size
        nlon = self.grid.lon.size  
        return ilat > (nlat//self.sigma)*self.sigma or ilon > (nlon//self.sigma)*self.sigma
    def centralization(self,ilat,ilon):
        nlat = self.grid.lat.size
        nlon = self.grid.lon.size        
            
        clat,clon = nlat//2,nlon//2
        rolldict = dict(lat = -ilat + clat,lon = -ilon + clon)#roll_coords = True)
        backrolldict = deepcopy(rolldict)
        for key in 'lat lon'.split():
            backrolldict[key] = -backrolldict[key]
        
        hspan = 3
        
        bnds = {t:(c - self.sigma*hspan,c + self.sigma*hspan ) for t,c in zip('lat lon'.split(),[clat,clon])}
        bnds = {key: (np.maximum(val[0],0).astype(int),np.minimum(val[1],size).astype(int)) \
                    for (key,val),size in zip(bnds.items(),[nlat,nlon])}
        slc = {key:slice(val[0],val[1]) for key,val in bnds.items()}
        return rolldict,slc,backrolldict
    def give_ocean_point_indices(self,):
        return np.where(self.grid.wet_mask.values[0] > 0,)
    def __call__(self,ilatilon):
        ilon = ilatilon %  self.nlon
        ilat = ilatilon // self.nlon
        
        if self.grid.wet_mask.values[0,ilat,ilon] == 0:
            return None                
        dims = 'lat lon'.split()
        coords= {c:self.grid[c].values for c in dims}                
        x = np.zeros(self.inshape)
        x[ilat,ilon] = 1
        
        x = xr.DataArray(
            data = x,dims = dims, coords = coords
        ).expand_dims({'depth':[self.grid.depth.values.item()]},axis = 0)
         
        rolldict,slc,backrolldict = self.centralization(ilat,ilon)
        subgrid = self.grid.roll(**rolldict).isel(**slc)
        subx = x.roll(**rolldict).isel(**slc)                

        slc_ = tuple(slc.values())
        filtering = filtering_class(self.sigma,subgrid)

        fsubx = filtering(subx)
        fsubx_ = np.zeros(self.inshape)
        fsubx_[slc_] = fsubx.values
       
        fsubx = xr.DataArray(
            data = fsubx_,dims = dims, coords = coords
        )                    
        fsubx = fsubx.roll(**backrolldict)
        cx = self.coarse_graining(fsubx).fillna(0).compute().values    
        cx = cx.flatten()
        return cx
    
    def test(self,):
        self.post__init__()
        lat,lon = self.give_ocean_point_indices()
        np.random.seed(0)
        randints = np.random.randint(len(lat),size = 64)
        selection_points = tuple(zip(lat[randints],lon[randints]))
        class_args = (self.sigma,self.grid)
        filtering = filtering_class(*class_args)
        coarse_graining = coarse_grain_class(*class_args)
        for i,sp in enumerate(selection_points):
            x = np.zeros(self.inshape)
            x[sp] = 1
            cx = self(x).flatten()
            tcx = coarse_graining(filtering(x)).fillna(0)
            err = np.abs(cx - tcx.values.flatten())
            relerr = np.sum(err)/( np.sum(np.abs(tcx.values.flatten())) )
            logging.info(f'#{i} - {sp}, relerr = {relerr}')
            if relerr < 1e-19:
                continue
            fig,axs = plt.subplots(ncols = 3,figsize = (30,10))
            np.log10(tcx + 1e-9).plot(ax = axs[0])
            tcx.data = cx.reshape(tcx.shape)
            np.log10(tcx + 1e-9).plot(ax = axs[1])
            tcx.data = err.reshape(tcx.shape)
            np.log10(tcx + 1e-9).plot(ax = axs[2])
            for ax,title in zip(axs,'true est err'.split()):
                ax.set_title(title)
            fig.savefig(f'image-{i}.png')
            
def get_grid(sigma:int,depth:int):
    args = f'--sigma {sigma} --depth {depth} --mode data --filtering gcm'.split()
    x, = get_data(args,torch_flag=False,data_loaders=False,groups = ('train',))
    if depth == 0:
        x0 = x.per_depth[0]
    else:
        i = np.argmin(np.abs(x.ds.depth.values - depth))
        x0 = x.per_depth[i]
    ugrid = x0.ugrid
    ugrid = ugrid.drop('time co2'.split())
    return ugrid.load()
    m = 80
    return ugrid.isel(lat = slice(1000,1000+m),lon = slice(1000,1000+m))

def main():
    logging.basicConfig(level=logging.INFO,format = '%(asctime)s %(message)s',)
    args = sys.argv[1:] 
    parti = int(args[0])
    partm = int(args[1])
    sigma = int(args[2])
    depth = int(args[3])
    ncpu = int(args[4])
    foldername = os.path.join(OUTPUTS_PATH,'filter_weights')
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    bf = BaseFiltering(sigma,depth)
    
    root = f'gcm-dpth-{depth}-sgm-{sigma}-parts'
    root = os.path.join(foldername,root)
    flushed_print(
        f'parti = {parti}\tpartm = {partm}\tsigma = {sigma}\tncpu = {ncpu}\t'
    )
    if not os.path.exists(root):
        os.makedirs(root)
    fileroot = os.path.join(root,'parallel')
    flushed_print(
        f'root = {root}'
    )
    spvc = SparseVecCollection(bf,fileroot,partition = (parti,partm),ncpu=ncpu,tol = 1e-19)
    spvc.collect_basis_elements()
    
# def main():
#     logging.basicConfig(level=logging.INFO,format = '%(asctime)s %(message)s',)
#     # dummy()
#     # return
#     from linear.coarse_graining_inversion import CollectParts
#     sigma = 4
#     head = f'gcm-dpth-0-sgm-{sigma}' 
#     # CollectParts.collect(head)
#     root = os.path.join(OUTPUTS_PATH,'filter_weights')
#     path = CollectParts.latest_united_file(root,head)
    
#     # path = '/scratch/cg3306/climate/outputs/filter_weights/gcm-dpth-0-sgm-4.npz'
#     bf = BaseFiltering(sigma,0)
#     bf.post__init__()
#     grid = get_grid(sigma,0)
#     import scipy.sparse as sp
#     matform = sp.load_npz(path).toarray()
    
#     class_args = (sigma,grid)
#     filtering = filtering_class(*class_args)
#     coarse_graining = coarse_grain_class(*class_args)
#     x = np.random.randn(*bf.inshape)
#     # x[30,30] = 1
#     cx_ = (matform @ x.flatten()).reshape(bf.outshape)
#     cx = coarse_graining(filtering(x)).values.reshape(bf.outshape)
#     relerr = np.log10(np.abs(cx_ - cx) + 1e-19)
#     logging.info(f'shapes = {cx_.shape,cx.shape}')
    
#     fig,axs = plt.subplots(ncols = 3,figsize = (20,10))
    
#     for ax,val in zip(axs,[cx_,cx,relerr]):
#         vmax = np.amax(np.abs(val))
#         neg = ax.imshow(val,cmap = 'bwr',vmin = -vmax,vmax = vmax)
#         fig.colorbar(neg,ax = ax)
#     fig.savefig('dummy.png')
#     plt.close()
    
        
    
# def main():
#     args = sys.argv[1:]
#     parti = int(args[0])
#     partm = int(args[1])
#     sigma = int(args[2])
#     ncpu = int(args[3])
    
#     foldername = os.path.join(OUTPUTS_PATH,'filter_weights')
#     if not os.path.exists(foldername):
#         os.makedirs(foldername)
#     bf = BaseFiltering(sigma,0)
#     fileroot = f'gcm-dpth-{0}-sgm-{sigma}'
#     pathroot = os.path.join(foldername,fileroot)
#     flushed_print(
#         f'parti = {parti}\tpartm = {partm}\tsigma = {sigma}\tncpu = {ncpu}\t'
#     )
#     flushed_print(
#         f'pathroot = {pathroot}'
#     )
#     spvc = SparseVecCollection(bf,pathroot,partition = (parti,partm),ncpu=ncpu)
#     fu = spvc.basis_element(500)
#     cu = bf(fu)
    

if __name__ == '__main__':
    main()
    

