import logging
from typing import Tuple
import numpy as np
import scipy.sparse as sp
class BinaryCounter:
    def __init__(self, numdigs:int) :
        self.index= 0 
        self.numdigs =numdigs
    @property
    def binary(self,):
        x = format(self.index,'b')
        zs = self.numdigs - len(x)
        return "0"*zs + x
    @property
    def array(self,):
        bin = self.binary
        return np.array([int(x) for x in bin])
    def __str__(self,):
        return self.binary
    def increment(self,):
        self.index+= 1
        self.index = self.index % 2**self.numdigs
    def decrement(self,):
        self.index -= 1



class HierarchicalMatrix:
    org_size:int
    invertible_size:int
    size:int
    nlevels:int
    mat:np.ndarray
    def __init__(self,mat:np.ndarray,invertible_size:int) -> None:
        size = mat.shape[0]
        self.invertible_size = invertible_size
        self.nlevels = int(np.log2(size//invertible_size))
        self.mat = mat
        self.pows = np.power(2,np.arange(self.nlevels)[::-1])
    def arr2slc(self,arr:Tuple[int,...]):
        ind =  self.pows[:len(arr)] @ arr * self.invertible_size
        ind1 = ind + self.pows[len(arr)-1] *  self.invertible_size
        return slice(ind,ind1)
    def __getitem__(self,arr:Tuple[int,...]):
        if not bool(arr):
            return self.mat
        slc = self.arr2slc(arr)
        return self.mat[slc,slc]
    def __setitem__(self,arr:Tuple[int,...],mat):
        if not bool(arr):
            self.mat = mat
        slc = self.arr2slc(arr)
        self.mat[slc,slc] = mat
    def divide(self,mat):
        hside = mat.shape[0]//2
        s0,s1 = slice(0,hside),slice(hside,2*hside)
        m00 = mat[s0,s0]
        m01 = mat[s0,s1]
        m10 = mat[s1,s0]
        m11 = mat[s1,s1]
        return m00,m01,m10,m11

    def act(self,arr:Tuple[int,...]):
        if arr[-1] == 0:
            self.m_a_placement(arr)
        else:
            while bool(arr):
                if arr[-1] == 1:
                    self.complete_inversion(arr)
                    arr = arr[:-1]
                else:
                    self.m_a_placement(arr)
                    break
                
                
    def m_a_placement(self,arr:Tuple[int,...]):
        '''
        at 'level'
        mat[ind,ind] <- A^{-1}
        mat[ind+1,ind+] <- (D - C@A^{-1}@B)^{-1}
        '''
        mat = self[arr[:-1]]
        ainv,b,c,d = self.divide(mat)
        level = len(arr)
        if level == self.nlevels - 1: 
            ainv = np.linalg.inv(ainv)
            self[arr] = ainv
        m_a = d - c@ ainv @ b
        if level == self.nlevels - 1: 
            m_a = np.linag.inv(m_a)
        arr0 = list(arr)
        arr0[-1] = 1
        self[tuple(arr0)] = m_a
        
        

    def complete_inversion(self,arr:Tuple[int,...]):
        mat = self[arr[:-1]]
        ainv,b,c,m_a_inv = self.divide(mat)

            
        new_a = ainv + ainv @ b @ m_a_inv @ c @ ainv
        new_b = - ainv @ b @ m_a_inv
        new_c = - m_a_inv @ c @ ainv
        new_d = m_a_inv
        mother_inverse = np.block([[new_a,new_b],[new_c,new_d]])
        self[arr[:-1]] = mother_inverse
        
    
    
class Inversion(BinaryCounter):
    def __init__(self, hm:HierarchicalMatrix):
        super().__init__(hm.nlevels)
        self.mat = hm
    def finish_all(self,):
        self.mat.act(self.array)
        self.increment()
        while self.index !=  0:
            self.mat.act(self.array)
            self.increment()
        
        
def main():
    n = 2**4
    m = 2**2
    mat = np.zeros((n,n))
    hm = HierarchicalMatrix(mat,m)
    inverter = Inversion(hm)
    logging.basicConfig(level=logging.INFO,\
                format = '%(asctime)s %(message)s',)
    inverter.finish_all()
    
if __name__ == '__main__':
    main()