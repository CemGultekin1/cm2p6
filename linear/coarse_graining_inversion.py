import logging
from linear.lincol import  CollectParts
from linear.nonrecursive_hierchinv import SparseHierarchicalInversion,coo_submatrix_pull
import numpy as np
from constants.paths import OUTPUTS_PATH
import os
import scipy.sparse as sp
logging.basicConfig(level=logging.INFO,\
                format = '%(asctime)s %(message)s',)
class RemoveZeroRows:
    def __init__(self,mat,) -> None:
        x = np.ones(mat.shape[1],)
        y = mat@x
        self.nnzrows = np.where(np.abs(y)>0)[0]
        self.nrows = mat.shape[0]
    def remove_nnz_rows(self,mat ):
        return coo_submatrix_pull(mat.tocoo(),self.nnzrows,np.arange(mat.shape[1]))
    def sprase_expansion_mat(self,nrows):
        x = sp.lil_matrix((nrows,len(self.nnzrows)))
        for i,nnz in enumerate(self.nnzrows):
            x[nnz,i] = 1
        x = x.tocsr()
        return x
    def expand_with_zero_rows(self,mat,nrows):
        assert mat.shape[0] == len(self.nnzrows)
        x = self.sprase_expansion_mat(nrows)
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
    def load(self,):
        self.mat = CollectParts.load_spmat(self.path).tocsr()
        rzr = RemoveZeroRows(self.mat)
        self.mat = rzr.remove_nnz_rows(self.mat)
        self.rzr = rzr
    def compute_quadratic_mat(self,):
        self.qmat =  self.mat @ self.mat.T
        
    def compute_left_inv(self,tol = 1e-3):
        hinv = SparseHierarchicalInversion(self.qmat.tocoo(),2**9,tol = tol,)
        qinv = hinv.invert()
        self.invmat = self.mat.T @ qinv
        self.invmat = self.rzr.expand_with_zero_rows(self.invmat.T,self.rzr.nrows).T
    @property
    def inverse_path(self,):
        return self.path.replace('.npz','-inv.npz')
    def save_inverse(self,):
        CollectParts.save_spmat(self.inverse_path, self.invmat)
def main():
    root = os.path.join(OUTPUTS_PATH,'filter_weights')
    head = 'gcm-dpth-0-sgm-4'
    path = CollectParts.latest_united_file(root,head)
    logging.info(f'loading path : {path}')
    return
    neq = NormalEquations(path)    
    logging.info('neq.load()...')
    neq.load()
    logging.info('\t\t\t\t done.')
    logging.info('neq.compute_quadratic_mat()...')
    neq.compute_quadratic_mat()
    logging.info('\t\t\t\t done.')    

    logging.info('neq.compute_left_inv()...')
    neq.compute_left_inv(tol = 1e-3)
    logging.info('\t\t\t\t done.')


if __name__ == '__main__':
    main()
    


