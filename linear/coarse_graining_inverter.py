import logging
from linear.lincol import  CollectParts
from linear.nonrecursive_hierchinv import SparseHierarchicalInversion
import numpy as np
from constants.paths import FILTER_WEIGHTS
import os
import scipy.sparse as sp
import xarray as xr
import itertools


class NormalEquations:
    def __init__(self,path,filtering:str = 'gcm',depth:int = 0, sigma:int = 16) -> None:
        self.sigma = sigma
        self.depth = depth
        self.filtering = filtering
        if not os.path.exists(path):
            logging.error(f'path doesn\'t exist, path = {path}')
            raise Exception
        self.path = path
        self.mat = None
        self.qmat = None
        self.qinvmat = None
        self.rzr = None
        self.leftinvmat = None
        
        
    def load(self,):
        logging.info(f'CollectParts.load_spmat({self.path}).tocsr()')
        self.mat = CollectParts.load_spmat(self.path).tocsr()     
        logging.info(f'mat.shape = {self.mat.shape}')
    def compute_quadratic_mat(self,):
        self.qmat =  self.mat @ self.mat.T
        # self.qmat = 
    def load_quad_inverse(self,):
        logging.info(f'loading quadratic inverse: ')
        self.qinvmat = CollectParts.load_spmat(self.inverse_path,)
        logging.info(f'\t\t qinvmat.shape = {self.qinvmat.shape}')
        # if self.qinvmat.shape[0] > self.mat.shape[0] and self.rzr is not None:
        #     self.qinvmat = self.rzr.pick_rows(self.qinvmat)
    def compute_quad_inverse(self,save_dir:str,tol = 1e-7,verbose:bool = False):
        milestones = np.arange(20)/20
        milestones = np.append(milestones,[.99,.999,1])
        
        
        hinv = SparseHierarchicalInversion(self.qmat.tocoo(),2**10,\
                        tol = tol,\
                        verbose = verbose,\
                        continue_flag= True,\
                        milestones=milestones,
                        save_dir=save_dir)
        self.qinvmat = hinv.invert(inplace = True)#self.mat.T @ qinv
    def compute_left_inverse(self,):
        self.leftinvmat = self.mat.T @ self.qinvmat
    @property
    def inverse_path(self,):
        return self.path.replace('.npz','-inv.npz')
    @property
    def left_inverse_path(self,):
        return self.path.replace('.npz','-left-inv.npz')
    def save_inverse(self,):
        CollectParts.save_spmat(self.inverse_path, self.qinvmat)
    def save_left_inverse(self,):
        CollectParts.save_spmat(self.inverse_path, self.leftinvmat)        
    
class CoarseGrainingInverter(NormalEquations):
    def __init__(self,filtering:str = 'gcm',depth:int = 0, sigma:int = 16) -> None:
        print(f'CoarseGrainingInverter.__init__ {filtering,depth,sigma}')
        self.sigma = sigma
        self.depth = depth
        self.filtering = filtering
        head = f'{filtering}-dpth-{depth}-sgm-{sigma}' 
        # self.args = f'--filtering {filtering} --sigma {sigma} --depth {depth} --co2 True'.split()
        path = CollectParts.latest_united_file(FILTER_WEIGHTS,head)
        super().__init__(path,filtering = filtering,depth = depth,sigma = sigma)
        
    def load_parts(self,):
        self.load()
        self.load_quad_inverse()
    
    def forward_model(self,u :xr.DataArray   ):
        dims = u.dims
        lat = [d for d in dims if 'lat' in d][0]
        lon = [d for d in dims if 'lon' in d][0]
        nonlatlondims = [d for d in dims if 'lat' not in d and 'lon' not in d]
        nnll = {d:u[d].values for d in nonlatlondims}
        clatlon = {lat:u[lat].coarsen(**{lat : self.sigma,'boundary' : 'trim'}).mean(),\
                    lon:u[lon].coarsen(**{lon : self.sigma,'boundary' : 'trim'}).mean()}
        keys = list(nnll.keys())
        cus = []
        for vals in itertools.product(*nnll.values()):
            curdict = dict(tuple(zip(keys,vals)))
            subu = u.sel(**curdict).fillna(0).values.squeeze().flatten()
            logging.info(f'self.mat.shape, subu.shape = {self.mat.shape, subu.shape}')
            cu = self.mat @ subu
            # cu = self.rzr.expand_with_zero_rows(cu)
            # cu = subu.coarsen({
            #     lat: self.sigma,lon:self.sigma
            # },boundary = 'trim').mean()
            cus.append(cu)
        cus = np.stack(cus,axis = 0)
        nvl = [len(val) for val in nnll.values()]
        shp = nvl + [len(clatlon[lat]),len(clatlon[lon])]
        cus = cus.reshape(shp)        
        dims = list(nnll.keys()) + [lat, lon]
        nnll.update(clatlon)
        return xr.DataArray(
            data = cus, dims = dims,coords = nnll
        )
    def project(self,u:xr.DataArray):
        dims = u.dims
        lat = [d for d in dims if 'lat' in d][0]
        lon = [d for d in dims if 'lon' in d][0]
        nonlatlondims = [d for d in dims if 'lat' not in d and 'lon' not in d]
        nnll = {d:u[d].values for d in nonlatlondims}
        keys = list(nnll.keys())
        cus = []
        for vals in itertools.product(*nnll.values()):
            curdict = dict(tuple(zip(keys,vals)))
            subu = u.sel(**curdict).fillna(0).values.squeeze().flatten()
            # logging.info(f'self.mat.shape, subu.shape = {self.mat.shape, subu.shape}')
            cu = self.mat @ subu
            cu = self.qinvmat @ cu
            fu = self.mat.T @ cu      
            cus.append(fu)
        cus = np.stack(cus,axis = 0)
        nvl = [len(val) for val in nnll.values()]
        shp = nvl + [len(u[lat]),len(u[lon])]
        cus = cus.reshape(shp)        
        return xr.DataArray(
            data = cus, dims = u.dims,coords = u.coords
        )

    def inverse_model(self,u :xr.DataArray ,fucoords:dict  ):
        dims = u.dims
        lat = [d for d in dims if 'lat' in d][0]
        lon = [d for d in dims if 'lon' in d][0]
        
        lat_ = [d for d in fucoords if 'lat' in d][0]
        lon_ = [d for d in fucoords if 'lon' in d][0]
        
        nonlatlondims = [d for d in dims if 'lat' not in d and 'lon' not in d]
        nnll = {d:u[d].values for d in nonlatlondims}
        
        clatlon = {l:v for l,v in fucoords.items() if l in (lat_,lon_)}
        
        keys = list(nnll.keys())
        cus = []
        for vals in itertools.product(*nnll.values()):
            curdict = dict(tuple(zip(keys,vals)))
            subu = u.sel(**curdict).fillna(0).values.squeeze().flatten()
            subu_ = self.rzr.pick_rows(sp.coo_matrix(subu.reshape([-1,1])))
            logging.info(f'subu_.size = {subu_.size}')
            cu = self.qinvmat @ subu_
            logging.info(f'cu.size = {cu.size}')
            fu = self.mat.T @ cu      
            logging.info(f'fu.size = {fu.size}')
            fu = fu.toarray().flatten()      
            cus.append(fu)
        cus = np.stack(cus,axis = 0)
        nvl = [len(val) for val in nnll.values()]
        shp = nvl + [len(clatlon[lat]),len(clatlon[lon])]
        cus = cus.reshape(shp)        
        dims = list(nnll.keys()) + [lat, lon]
        nnll.update(clatlon)
        return xr.DataArray(
            data = cus, dims = dims,coords = nnll
        )