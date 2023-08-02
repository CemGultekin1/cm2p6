import logging
from typing import Tuple
import numpy as np
import scipy.sparse as sp
import multiprocessing
logging.basicConfig(level=logging.INFO,\
                format = '%(asctime)s %(message)s',)

class HierarchicalInversion:
    def __init__(self,mat,tol = 1e-5,max_inversion_size:int = 128) -> None:
        self.tol = tol
        self.mat = mat
        self.dim = self.mat.shape[0]
        self.max_inversion_size = max_inversion_size
    def extend2nearest2power(self,):
        pow = np.ceil(np.log2(self.dim)).astype(int)
        sep = pow - np.log2(self.max_inversion_size*2)
        if sep  < 1:
            return self.mat
        sep = np.ceil(sep).astype(int)
        sep2 = 2**sep
        new_span = np.ceil(self.dim/sep2).astype(int) * sep2
        extended_mat = sp.dok_array((new_span,new_span))
        extended_mat[:self.dim,:self.dim] = self.mat
        for i in range(self.dim,new_span):
            extended_mat[i,i] = 1
        return extended_mat
    def get_slices(self,mat,nsplits,i):
        dim = mat.shape[0]
        dx = dim//nsplits
        x0 = dx*i
        x1 = x0 + dx//2
        x2 = x1 + dx//2
        slc01 = slice(x0,x1)
        slc12 = slice(x1,x2)

        tp00 = (slc01,slc01)
        tp01 = (slc01,slc12)
        tp10 = (slc12,slc01)
        tp11 = (slc12,slc12)
        
        m00 = mat[tp00]
        m01 = mat[tp01]
        m10 = mat[tp10]
        m11 = mat[tp11]
        
        return (m00,m01,m10,m11),(tp00,tp01,tp10,tp11)
    
    def sparsify(self,mat:np.ndarray):
        amat = np.abs(mat)
        mat[amat < self.tol] = 0
        return sp.dok_array(mat)
    def invert(self,):
        # logging.info(f'HierarchicalInversion.invert,dim = {self.dim}')
        extended_mat = self.extend2nearest2power()
        inverses_mat = sp.dok_array(extended_mat.shape)
        dim = extended_mat.shape[0]
        # logging.info(f'extended_mat.shape[0] = {dim}')
        nsplits =  dim//self.max_inversion_size//2
        for i in range(nsplits):
            # logging.info(f'\t\t for {i} in range({nsplits})')
            (amat,bmat,cmat,dmat),tps = self.get_slices(extended_mat,nsplits,i)
            dinv = np.linalg.inv(dmat.toarray())
            dinv = self.sparsify(dinv)
            m_sl_d = amat - bmat @ dinv @ cmat
            # logging.info(f'{m_sl_d.shape} = {amat.shape} - {bmat.shape} @ {dinv.shape} @ {cmat.shape}')
            m_sl_d_inv = np.linalg.inv(m_sl_d.toarray())
            m_sl_d_inv = self.sparsify(m_sl_d_inv)

            # logging.info(f'tps = {tps[0][0].start,tps[0][0].stop,tps[1][1].start,tps[1][1].stop}')
            inverses_mat[tps[0]] = m_sl_d_inv
                        
            inverses_mat[tps[1]] = -m_sl_d_inv @ bmat @ dinv
            
            inverses_mat[tps[2]] = - dinv @ cmat @ m_sl_d_inv
            
            inverses_mat[tps[3]] = dinv  + dinv @ cmat @ m_sl_d_inv @ bmat @ dinv
            
        
        nsplits= nsplits//2        
        while nsplits >= 1:       
            # logging.info(f'\t\t while {nsplits} > 1:       ')     
            for i in range(nsplits):
                (amat,bmat,cmat,dmat),tps = self.get_slices(extended_mat,nsplits,i)
                (_,_,_,dinv),tps = self.get_slices(inverses_mat,nsplits,i)
                
                m_sl_d = amat - bmat @ dinv @ cmat
                hinv = HierarchicalInversion(m_sl_d, tol = self.tol, max_inversion_size=self.max_inversion_size)
                m_sl_d_inv = hinv.invert()
                                
                inverses_mat[tps[0]] = m_sl_d_inv
                        
                inverses_mat[tps[1]] = -m_sl_d_inv @ bmat @ dinv
                
                inverses_mat[tps[2]] = - dinv @ cmat @ m_sl_d_inv
                
                inverses_mat[tps[3]] = dinv  + dinv @ cmat @ m_sl_d_inv @ bmat @ dinv
            nsplits= nsplits//2
        return inverses_mat[:self.dim,:self.dim]

        
            
def main():
    n = 1843
    x = np.random.randn(n,n)
    mat = sp.dok_array(x)
    hinv = HierarchicalInversion(mat,tol = 1e-9,max_inversion_size=16,)
    matinv = hinv.invert()
    matinv = matinv.toarray()
    err = np.mean(np.abs(x @ matinv - np.eye(n)))
    logging.info(f'n = {n},\t err = {err}')


if __name__ == '__main__':
    main()