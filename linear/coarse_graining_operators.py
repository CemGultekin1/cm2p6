from copy import deepcopy
import itertools
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
filtering_class = GcmFiltering
coarse_grain_class =  GreedyCoarseGrain
class BaseFiltering(LinFun):
    def __init__(self,sigma,depth) -> None:
        self.sigma = sigma
        self.depth = depth
    def post__init__(self,) -> None:
        grid = get_grid(self.sigma,self.depth)
        self.grid = grid
        self.indim = grid.lat.size * grid.lon.size
        
        self.outdim = (grid.lat.size//self.sigma )* (grid.lon.size//self.sigma)
        self.inshape = (grid.lat.size,grid.lon.size)
        self.outshape = (grid.lat.size//self.sigma,grid.lon.size//self.sigma)
        self.coarse_graining = coarse_grain_class(self.sigma,self.grid)
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
        clat,clon,ilat,ilon = self.capitalize(clat,clon,ilat,ilon)
        rolldict = dict(lat = -ilat + clat,lon = -ilon + clon)#roll_coords = True)
        backrolldict = deepcopy(rolldict)
        for key in 'lat lon'.split():
            backrolldict[key] = -backrolldict[key]
        
        hspan = 5
        
        slc = {t:slice(c - self.sigma*hspan,c + self.sigma*hspan ) for t,c in zip('lat lon'.split(),[clat,clon])}
        return rolldict,slc,backrolldict
    def output_patch(self,ilat:int,ilon:int):
        _,_,cslc = self.centralization(ilat,ilon)
        return cslc
    def give_ocean_point_indices(self,):
        return np.where(self.grid.wet_mask.values[0] > 0,)
    def __call__(self,x:np.ndarray):
        x = x.reshape(*self.inshape)
        ilats,ilons = np.where(x > 0 )
        
        assert ilats.size == 1  and ilons.size == 1
        ilat,ilon = ilats.item(),ilons.item()
        if self.grid.wet_mask.values[0,ilat,ilon] == 0:
            return np.zeros((self.outdim))
        
        x = np.stack([x],axis = 0) # depth
        self.grid['x'] = ('depth lat lon'.split(),x)     
           
        filt,slcdict,lclgrid,backroll = self.get_local_operators(ilat,ilon)
        x = lclgrid.x
        fx = filt(x)
        x = self.grid.x.copy()*0
        x.data[:,slcdict['lat'],slcdict['lon']] = fx.data
        x = x.roll(**backroll)
        cx = self.coarse_graining(x).fillna(0).compute().values
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
            
def get_grid(sigma:int,depth:int,**isel_kwargs):
    args = f'--sigma {sigma} --depth {depth} --mode data --filtering gcm'.split()
    x, = get_data(args,torch_flag=False,data_loaders=False,groups = ('train',))
    x0 = x.per_depth[0]
    ugrid = x0.ugrid
    ugrid = ugrid.isel(**isel_kwargs).drop('time co2'.split())
    return ugrid#.isel(lat = slice(1000,1200 + off),lon = slice(1000,1200 + off))

def main():
    args = sys.argv[1:]
    parti = int(args[0])
    partm = int(args[1])
    sigma = int(args[2])
    ncpu = int(args[3])
    logging.basicConfig(level=logging.INFO,format = '%(message)s',)
    foldername = os.path.join(OUTPUTS_PATH,'filter_weights')
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    bf = BaseFiltering(sigma,0)
    # bf.test()
    # return
    
    fileroot = f'gcm-dpth-{0}-sgm-{sigma}'
    pathroot = os.path.join(foldername,fileroot)
    flushed_print(
        f'parti = {parti}\tpartm = {partm}\tsigma = {sigma}\tncpu = {ncpu}\t'
    )
    flushed_print(
        f'pathroot = {pathroot}'
    )
    spvc = SparseVecCollection(bf,pathroot,partition = (parti,partm),ncpu=ncpu)
    spvc.collect_basis_elements()

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
    


