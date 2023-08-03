import logging
from typing import Tuple
import numpy as np
import scipy.sparse as sp
logging.basicConfig(level=logging.INFO,\
                format = '%(asctime)s %(message)s',)


def coo_submatrix_pull(matr, rows, cols):
    """
    Pulls out an arbitrary i.e. non-contiguous submatrix out of
    a sparse.coo_matrix. 
    """
    if type(matr) != sp.coo_matrix:        
        raise TypeError(f'Matrix must be sparse COOrdinate format, type is {type(matr)}')
    
    gr = -1 * np.ones(matr.shape[0])
    gc = -1 * np.ones(matr.shape[1])
    
    lr = len(rows)
    lc = len(cols)
    
    ar = np.arange(0, lr)
    ac = np.arange(0, lc)
    gr[rows[ar]] = ar
    gc[cols[ac]] = ac
    mrow = matr.row
    mcol = matr.col
    newelem = (gr[mrow] > -1) & (gc[mcol] > -1)
    newrows = mrow[newelem]
    newcols = mcol[newelem]
    return sp.coo_matrix((matr.data[newelem], np.array([gr[newrows],
        gc[newcols]])),(lr, lc))
    
class RecursiveHierarchicalInversion:
    def __init__(self,mat,tol = 1e-5,max_inversion_size:int = 128,num_inversions :int = 0,tot_inversions:int = -1,level:int = 0) -> None:
        self.tol = tol
        self.mat = mat
        self.dim = self.mat.shape[0]
        self.max_inversion_size = max_inversion_size
        self.num_inversions= num_inversions
        self.tot_inversions = tot_inversions        
        self.level = level
        
    def extend2nearest2power(self,):
        if self.level > 0:
            return

        k = np.ceil(np.log2(self.dim) - np.log2(self.max_inversion_size) - 1).astype(int)
        sep = 2**(k+1)*self.max_inversion_size - self.dim
        speye = sp.eye(sep).tocoo()
        self.mat = sp.bmat([[self.mat.tocoo(),None],[None,speye]]).tocoo()
        if self.tot_inversions < 0:
            self.tot_inversions = self.mat.shape[0]//self.max_inversion_size
        
    def pull_quad_submatrices(self,):
        dim = self.mat.shape[0]
        dx = dim //2
        x0 = 0
        x1 = x0 + dx
        x2 = x1 + dx
        i0 = np.arange(x0,x1)
        i1 = np.arange(x1,x2)
        self.mat = self.mat.tocoo()
        m00 = coo_submatrix_pull(self.mat,i0,i0)
        m01 = coo_submatrix_pull(self.mat,i0,i1)
        m10 = coo_submatrix_pull(self.mat,i1,i0)
        m11 = coo_submatrix_pull(self.mat,i1,i1)
        return m00,m01,m10,m11
    
    def sparsify(self,mat:np.ndarray):        
        amat = np.abs(mat)
        mat[amat < self.tol] = 0
        mat = sp.csr_matrix(mat)
        return mat
    
    def create_child(self,mat)->'RecursiveHierarchicalInversion':
        return RecursiveHierarchicalInversion(mat,\
                    tol = self.tol, \
                    max_inversion_size=self.max_inversion_size,\
                    num_inversions=self.num_inversions,\
                    tot_inversions=self.tot_inversions,\
                    level = self.level + 1)  
    def eat_child(self,mat:'RecursiveHierarchicalInversion'):
        self.num_inversions = mat.num_inversions
        
    def invert(self,):
        self.extend2nearest2power()
        dim = self.mat.shape[0]
        if dim == self.max_inversion_size:            
            invmat = np.linalg.inv(self.mat.toarray())
            self.num_inversions += 1
            mat =  self.sparsify(invmat)            
            return mat
        
        amat,bmat,cmat,dmat= self.pull_quad_submatrices()
        
        dinv_creator = self.create_child(dmat,)
        dinv = dinv_creator.invert()
        self.eat_child(dinv_creator)
        
        amat,bmat,cmat,dmat  = (x.tocsr() for x in (amat,bmat,cmat,dmat))
        m_sl_d = amat - bmat @ dinv @ cmat
        
        m_sl_dinv_creator = self.create_child(m_sl_d,)        
        m_sl_d_inv = m_sl_dinv_creator.invert()
        self.eat_child(m_sl_dinv_creator)
        
        new_a = m_sl_d_inv
        new_b = -m_sl_d_inv @ bmat @ dinv
        new_c = - dinv @ cmat @ m_sl_d_inv
        new_d = dinv  + dinv @ cmat @ m_sl_d_inv @ bmat @ dinv  
            

        mat =  sp.bmat([[new_a,new_b],[new_c,new_d]])
        logging.info(f'inversion #{self.num_inversions}/{self.tot_inversions}, level = {self.level}, density = {mat_density(mat)},side = {mat.shape[0]}')
        if self.level== 0:
            return coo_submatrix_pull(mat.tocoo(),np.arange(self.dim),np.arange(self.dim))
        return mat
def mat_density(mat):
    return mat.nnz/(mat.shape[0]**2)
def main():
    sigma = 16
    n = (2700*3600)//(sigma**2)
    x = sp.coo_matrix((n,n))
    x.setdiag(np.ones(n,))
    for k in range(1,10):
        x.setdiag(np.ones(n,)/(2**k),k = k)
        x.setdiag(np.ones(n,)/(2**k),k = -k)
    hinv = RecursiveHierarchicalInversion(x,tol = 1e-9,max_inversion_size=2**6,)
    matinv = hinv.invert()
    err = np.mean(np.abs(x @ matinv - sp.eye(n)))
    logging.info(f'n = {n},\t err = {err}')

    
    
    
if __name__ == '__main__':
    main()