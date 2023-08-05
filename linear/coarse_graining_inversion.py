import logging
from linear.lincol import  CollectParts
from linear.nonrecursive_hierchinv import SparseHierarchicalInversion,coo_submatrix_pull
import numpy as np
from constants.paths import OUTPUTS_PATH
import os
import scipy.sparse as sp
import sys
import xarray as xr
import itertools
class RemoveZeroRows:
    def __init__(self,mat,) -> None:
        x = np.ones(mat.shape[1],)
        y = mat@x
        self.nnzrows = np.where(np.abs(y)>= 0)[0]
        self.nrows = mat.shape[0]
    def remove_zero_rows(self,mat ):
        return coo_submatrix_pull(mat.tocoo(),self.nnzrows,np.arange(mat.shape[1]))
    def sprase_expansion_mat(self,):
        x = sp.lil_matrix((self.nrows,len(self.nnzrows)))
        for i,nnz in enumerate(self.nnzrows):
            x[nnz,i] = 1
        x = x.tocsr()
        return x
    def expand_with_zero_rows(self,mat,):
        assert mat.shape[0] == len(self.nnzrows)
        x = self.sprase_expansion_mat()
        return x @ mat
        
        
        
class NormalEquations:
    def __init__(self,path) -> None:
        if not os.path.exists(path):
            raise Exception
        self.path = path
        self.mat = None
        self.qmat = None
        self.invmat = None
        self.rzr = None
        self.leftinvmat = None
    def load(self,):
        self.mat = CollectParts.load_spmat(self.path).tocsr()
        rzr = RemoveZeroRows(self.mat)
        self.mat = rzr.remove_zero_rows(self.mat)
        self.rzr = rzr
    def compute_quadratic_mat(self,):
        self.qmat =  self.mat @ self.mat.T
    def load_quad_inverse(self,):
        self.invmat = CollectParts.load_spmat(self.inverse_path,)
    def compute_quad_inverse(self,save_dir:str,tol = 1e-3,verbose:bool = False):
        milestones = np.arange(10)/10
        milestones = np.append(milestones,[.99,.999,1])
        
        
        hinv = SparseHierarchicalInversion(self.qmat.tocoo(),2**10,\
                        tol = tol,\
                        verbose = verbose,\
                        continue_flag= True,\
                        milestones=milestones,
                        save_dir=save_dir)
        self.invmat = hinv.invert(inplace = True)#self.mat.T @ qinv
    def compute_left_inverse(self,):
        self.leftinvmat = self.mat.T @ self.invmat
    @property
    def inverse_path(self,):
        return self.path.replace('.npz','-inv.npz')
    @property
    def left_inverse_path(self,):
        return self.path.replace('.npz','-left-inv.npz')
    def save_inverse(self,):
        CollectParts.save_spmat(self.inverse_path, self.invmat)
    def save_left_inverse(self,):
        CollectParts.save_spmat(self.inverse_path, self.leftinvmat)
        
        
class CoarseGrainingInverter(NormalEquations):
    def __init__(self,filtering:str = 'gcm',depth:int = 0, sigma:int = 16) -> None:
        self.sigma = sigma
        self.depth = depth
        self.filtering = filtering
        head = f'{filtering}-dpth-{depth}-sgm-{sigma}' 
        root = os.path.join(OUTPUTS_PATH,'filter_weights')
        path = CollectParts.latest_united_file(root,head)
        super().__init__(path)
        
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
            cu = self.mat @ subu
            cu = self.rzr.expand_with_zero_rows(cu)
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
            subu_ = self.rzr.remove_zero_rows(sp.coo_matrix(subu.reshape([-1,1])))
            logging.info(f'subu_.size = {subu_.size}')
            cu = self.invmat @ subu_
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

def main():
    args = sys.argv[1:]
    filtering = args[0]
    depth = args[1]
    sigma = args[2]
    head = f'{filtering}-dpth-{depth}-sgm-{sigma}' 
    
    
    
    logging.basicConfig(level=logging.INFO,format = '%(message)s',)
    root = os.path.join(OUTPUTS_PATH,'filter_weights')
    
    
    
    path = CollectParts.latest_united_file(root,head)
    if not bool(path):
        CollectParts.collect(head)
        path = CollectParts.latest_united_file(root,head)
        assert bool(path)
        
    logging.info(f'loading path : {path}')
    neq = NormalEquations(path)   
     
    logging.info('neq.load()...')
    neq.load()
    logging.info('\t\t\t\t done.')
    
    
    
    # logging.info('neq.load_quad_inverse()...')
    # neq.load_quad_inverse()
    # logging.info('\t\t\t\t done.')
    
    
    
    logging.info(f'shape = {neq.mat.shape}')
    logging.info('neq.compute_quadratic_mat()...')
    neq.compute_quadratic_mat()
    logging.info('\t\t\t\t done.')    
    
    
    
    
    save_dir = os.path.join(root,head)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logging.info('neq.compute_quad_inverse()...')
    neq.compute_quad_inverse(save_dir,tol = 1e-9,verbose=True)
    logging.info('\t\t\t\t done.')
    logging.info('neq.save_inverse()...')
    neq.save_inverse()
    logging.info('\t\t\t\t done.')
    
    
    
    # logging.info('neq.compute_left_inverse()...')
    # neq.compute_left_inverse()
    # logging.info('\t\t\t\t done.')
    
    # logging.info('neq.save_left_inverse()...')
    # neq.save_left_inverse()
    # logging.info('\t\t\t\t done.')
    
    

if __name__ == '__main__':
    main()
    


