import logging
from linear.coarse_graining_operators import  get_grid,coarse_grain_class
from linear.lincol import  CollectParts
from linear.nonrecursive_hierchinv import SparseHierarchicalInversion,coo_submatrix_pull
import numpy as np
from constants.paths import FILTER_WEIGHTS, OUTPUTS_PATH
import os
import scipy.sparse as sp
import sys
import xarray as xr
import itertools

class RowPick:
    def __init__(self,rows:np.ndarray,nrows:int) -> None:
        self.nnzrows = rows
        self.nrows = nrows
    def pick_rows(self,mat ):
        expmat = self.sprase_expansion_mat()        
        return expmat.T @ mat
        # return coo_submatrix_pull(mat.tocoo(),self.nnzrows,np.arange(mat.shape[1]))
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
class ColPick(RowPick):
    def pick_cols(self, mat):
        return super().pick_rows(mat.T).T
    def expand_with_zero_cols(self, mat):
        return super().expand_with_zero_rows(mat.T).T
class RemoveZeroRows(RowPick):
    def __init__(self,mat,) -> None:
        x = np.ones(mat.shape[1],)
        y = mat@x
        nnzrows = np.where(np.abs(y)> 0)[0]
        nrows = mat.shape[0]
        super().__init__(nnzrows,nrows)
    def pick_rows(self,mat ):
        expmat = self.sprase_expansion_mat()        
        return expmat.T @ mat
        # return coo_submatrix_pull(mat.tocoo(),self.nnzrows,np.arange(mat.shape[1]))
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
        
    def cut_zero_rows(self,):
        logging.info(f'self.mat.nrows = {self.mat.shape[0]}')
        rzr = RemoveZeroRows(self.mat)        
        self.mat = rzr.pick_rows(self.mat)
        logging.info(f'...self.mat.nrows = {self.mat.shape[0]}')
        self.rzr = rzr
        
    def load(self,):
        logging.info(f'CollectParts.load_spmat({self.path}).tocsr()')
        self.mat = CollectParts.load_spmat(self.path).tocsr()        
    def compute_quadratic_mat(self,):
        self.qmat =  self.mat @ self.mat.T
        # self.qmat = 
    def load_quad_inverse(self,):
        self.qinvmat = CollectParts.load_spmat(self.inverse_path,)
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
    def apply_mask(self,):
        grid = get_grid(self.sigma,self.depth)
        coarse_graining = coarse_grain_class(self.sigma,grid)
        wet_density = coarse_graining.coarse_wet_density
        cwet_mask = (wet_density.fillna(0).values >= 0.5).astype(float)
        logging.info(f'cwet_mask density = {np.sum(cwet_mask)/np.sum(cwet_mask*0 + 1)}')
        spcwet = sp.diags(cwet_mask.flatten())
        logging.info(f'mat density = {self.mat.nnz/ (self.mat.shape[0]*self.mat.shape[1])}')
        self.mat = spcwet@self.mat
        logging.info(f'mat density = {self.mat.nnz/ (self.mat.shape[0]*self.mat.shape[1])}')
class CoarseGrainingInverter(NormalEquations):
    def __init__(self,filtering:str = 'gcm',depth:int = 0, sigma:int = 16) -> None:
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

def main():
    args = sys.argv[1:]
    filtering = args[0]
    depth = int(args[1])
    sigma = int(args[2])
    head = f'{filtering}-dpth-{depth}-sgm-{sigma}' 
    
    
    
    logging.basicConfig(level=logging.INFO,format = '%(asctime)s %(message)s',)
    root = os.path.join(OUTPUTS_PATH,'filter_weights')
    
    
    
    path = CollectParts.latest_united_file(root,head)
    if not bool(path):
        CollectParts.collect(head)
        path = CollectParts.latest_united_file(root,head)
        assert bool(path)
        
    logging.info(f'loading path : {path}')
    neq = NormalEquations(path,sigma = sigma,depth = depth,filtering = filtering)   
    logging.info('neq.load()...')
    neq.load()
    logging.info('\t\t\t\t done.')
    
    
    logging.info('neq.apply_mask()...')
    neq.apply_mask()
    # neq.cut_zero_rows()    
    logging.info('\t\t\t\t done.')
    
    
    logging.info(f'shape = {neq.mat.shape}')
    logging.info('neq.compute_quadratic_mat()...')
    neq.compute_quadratic_mat()
    logging.info('\t\t\t\t done.')    
    
    save_dir = os.path.join(root,head)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    logging.info('neq.compute_quad_inverse()...')
    neq.compute_quad_inverse(save_dir,tol = 1e-11,verbose=True)
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
    


