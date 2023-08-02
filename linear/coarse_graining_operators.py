from transforms.coarse_graining import GreedyCoarseGrain, GcmFiltering
from data.load import get_data
import numpy as np
from linear.lincol import LinFun, SparseVecCollection
import sys
from constants.paths import OUTPUTS_PATH
import os
from utils.slurm import flushed_print

filtering_class = GcmFiltering
coarse_grain_class =  GreedyCoarseGrain
class BaseFiltering(LinFun):
    def __init__(self,sigma,depth) -> None:
        self.sigma = sigma
        self.depth = depth
    def post__init__(self,) -> None:
        grid = get_grid(self.sigma,self.depth)
        grid['wet_mask'] = grid.wet_mask*0 + 1
        self.grid = grid
        self.indim = grid.lat.size * grid.lon.size
        self.outdim = grid.lat.size//self.sigma * grid.lon.size//self.sigma
        self.inshape = (grid.lat.size,grid.lon.size)
        self.outshape = (grid.lat.size//self.sigma,grid.lon.size//self.sigma)
    def get_local_operators(self,ilat:int,ilon:int):
        rolldict,slcdict,cslc,crolldict = self.centralization(ilat,ilon)
        grid = self.grid.roll(**rolldict)
        grid = grid.isel(**slcdict)

        class_args = (self.sigma,grid)
        filtering = filtering_class(*class_args)
        coarse_graining = coarse_grain_class(*class_args)
        return filtering,coarse_graining,grid,(cslc,crolldict)
    def picklable_arguments(self,):
        return self.sigma,self.depth
    @classmethod
    def from_picklable_arguments(self,sigma,depth):
        return BaseFiltering(sigma,depth)
    def capitalize(self,*args):
        return tuple((arg//self.sigma)*self.sigma for arg in args)
    def centralization(self,ilat,ilon):
        nlat = self.grid.lat.size
        nlon = self.grid.lon.size
        
        clat,clon = nlat//2,nlon//2
        clat,clon,ilat,ilon = self.capitalize(clat,clon,ilat,ilon)
        cclat = clat//self.sigma
        cclon = clon//self.sigma
        cilat,cilon = tuple( (il//self.sigma) for il in (ilat,ilon))
        rolldict = dict(lat = -ilat + clat,lon = -ilon + clon)
        crolldict = dict(lat = cilat - cclat,lon = cilon - cclon)
        
        hspan = 5
        
        slc = {t:slice(c - self.sigma*hspan,c + self.sigma*hspan ) for t,c in zip('lat lon'.split(),[clat,clon])}
        cslc = tuple(slice(c - hspan,c + hspan ) for c in [cclat,cclon])
        return rolldict,slc,cslc,crolldict
    def output_patch(self,ilat:int,ilon:int):
        _,_,cslc = self.centralization(ilat,ilon)
        return cslc
    def __call__(self,x:np.ndarray):
        x = x.reshape(*self.inshape)
        ilats,ilons = np.where(x > 0 )
        x = np.stack([x],axis = 0) # depth
        self.grid['x'] = ('depth lat lon'.split(),x)        
        assert ilats.size == 1  and ilons.size == 1
        filt,cg,lclgrid,outpatch = self.get_local_operators(ilats.item(),ilons.item())
        x = lclgrid.x
        fx = filt(x)
        cfg = cg(fx).fillna(0)
        cropped_data = cfg.data.compute()[0]
        x = np.zeros(self.outshape)
        outslc,outroll = outpatch
        x[outslc] = cropped_data
        x = np.roll(x,tuple(outroll.values()),axis = (0,1))   
        x = x.flatten()             
        return x
        
        

def get_grid(sigma:int,depth:int,**isel_kwargs):
    args = f'--sigma {sigma} --depth {depth} --mode data --filtering gcm'.split()
    x, = get_data(args,torch_flag=False,data_loaders=False,groups = ('train',))
    x0 = x.per_depth[0]
    ugrid = x0.ugrid
    ugrid = ugrid.isel(**isel_kwargs).drop('time co2'.split())
    return ugrid


def main():
    args = sys.argv[1:]
    parti = int(args[0])
    partm = int(args[1])
    sigma = int(args[2])
    ncpu = int(args[3])
    
    foldername = os.path.join(OUTPUTS_PATH,'filter_weights')
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    bf = BaseFiltering(sigma,0)
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


if __name__ == '__main__':
    main()
    


