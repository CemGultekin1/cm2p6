import logging
from typing import Any, List, Tuple
import numpy as np
import time
from datetime import datetime, timedelta
import os
class BinaryCounter:
    def __init__(self, numdigs:int,) :
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
        return tuple([int(x) for x in bin])
    def __str__(self,):
        return self.binary
    def increment(self,):
        self.index+= 1
        self.index = self.index % 2**self.numdigs
    def decrement(self,):
        self.index -= 1

class Reporter:
    def __init__(self,message:str = '',significance = 1.,progression_milestone = 0.05) -> None:
        self.message = message
        self.header_length = 0
        self.t0 = 0
        self.times = []
        self.significance = significance
        self.begin_time = datetime.now()
        self.last_progress_report = -1
        self.progression_milestone = progression_milestone
    def take_message(self,msg:str):
        self.message= msg        
    def msg_print(self,):
        self.header_length = np.maximum(len(self.message),self.header_length)
        # logging.info(self.message)
        self.t0 = time.time()
    def time_print(self,prog:float):
        t1 = time.time()
        dt = t1 - self.t0
        self.times.append(dt)
        avgtime = sum(self.times)/len(self.times)
        formatter = "{:.2e}"
        tottime = sum(self.times)
        milestone_flag = np.floor(prog/self.progression_milestone).astype(int) > self.last_progress_report
        if milestone_flag:
            self.last_progress_report =  np.floor(prog/self.progression_milestone).astype(int)
        expected_time_duration = tottime/prog
        tdelta = timedelta(seconds = expected_time_duration)
        expected_finish = self.begin_time + tdelta
        remaining = expected_finish  - datetime.now()
        if dt/avgtime > self.significance or milestone_flag:
            dtmess = self.message + ' '*(self.header_length - len(self.message))  + \
                        f'| dt = {formatter.format(dt)} secs' + \
                        f', remaining time = {remaining}'
            logging.info(dtmess)
            
reporter = Reporter()
def report_decorator(fun):
    def wrapped_fun(self, *args: Any, **kwds: Any) -> Any:
        vb = self.__dict__.get('verbose',False)
        
        if vb:
            
            reporter.take_message(self.message_writer(fun.__name__,args[0]))            
            reporter.msg_print()
        outputs =  fun.__call__(self,*args, **kwds)
        if vb:
            port = self.task_portion(args[0])
            reporter.time_print(port)
        return outputs
    return wrapped_fun


class HierarchicalMatrixInverter:
    invertible_size:int
    size:int
    nlevels:int
    mat:np.ndarray
    def __init__(self,mat:np.ndarray,invertible_size:int,verbose:bool = False) -> None:
        size = mat.shape[0]
        self.invertible_size = invertible_size
        self.nlevels = int(np.log2(size//invertible_size))
        self.mat = mat
        self.verbose = verbose
        self.pows = np.power(2,np.arange(self.nlevels)[::-1])
    def arr2slc(self,arr:Tuple[int,...]):
        ind =  self.pows[:len(arr)] @ arr * self.invertible_size
        ind1 = ind + self.pows[len(arr)-1] *  self.invertible_size
        return slice(ind,ind1)
    def message_writer(self,fun_name:str,arr:Tuple[int,...]):
        formatter = "{:.2e}"
        funcall =  f'{fun_name}(' + ''.join([str(x) for x in arr])+ '0'*(self.nlevels - len(arr)) + ')'
        nnz = f'nnz = {formatter.format(self.mat.nnz)}'
        density =  f'density = {formatter.format(self.mat.nnz/np.prod(self.mat.shape))}'
        return ' '.join([funcall ,nnz,density])
    def task_portion(self,arr:Tuple[int,...]):
        t = self.pows @ arr
        tot = self.pows @ np.ones((len(arr)))
        return (t+1)/tot
        
    def __getitem__(self,arr:Tuple[int,...]):
        if not bool(arr):
            return self.mat
        slc = self.arr2slc(arr)
        return self.mat[slc,slc]
    def __setitem__(self,arr:Tuple[int,...],mat):
        if not bool(arr):
            self.mat = mat
            return
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
    
    @report_decorator
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
                
    @staticmethod
    def invert_mat(mat):
        return np.linalg.inv(mat)
    
    def m_a_placement(self,arr:Tuple[int,...]):
        '''
        at 'level'
        mat[ind,ind] <- A^{-1}
        mat[ind+1,ind+] <- (D - C@A^{-1}@B)^{-1}
        '''
        mat = self[arr[:-1]]
        ainv,b,c,d = self.divide(mat)
        level = len(arr)
        if level == self.nlevels: 
            ainv = self.invert_mat(ainv)
            self[arr] = ainv
        m_a = d - c@ ainv @ b
        if level == self.nlevels : 
            m_a = self.invert_mat(m_a)
        arr0 = list(arr)
        arr0[-1] = 1
        self[tuple(arr0)] = m_a
        
    @staticmethod
    def merge_quad(a,b,c,d):
        mother_inverse = np.block([[a,b],[c,d]])
        return mother_inverse

    def complete_inversion(self,arr:Tuple[int,...]):
        mat = self[arr[:-1]]
        ainv,b,c,m_a_inv = self.divide(mat)

            
        new_a = ainv + ainv @ b @ m_a_inv @ c @ ainv
        new_b = - ainv @ b @ m_a_inv
        new_c = - m_a_inv @ c @ ainv
        new_d = m_a_inv
        
        self[arr[:-1]] = self.merge_quad(new_a,new_b,new_c,new_d)
        
    def invert(self,inplace:bool = True):
        if not inplace:
            self.mat = self.mat.copy()
        bc = BinaryCounter(self.nlevels)
        self.act(bc.array)
        bc.increment()
        while bc.index !=  0:
            self.act(bc.array)
            bc.increment()
        return self.mat
    
class SizeChagingHierarchicalInversion(HierarchicalMatrixInverter):
    def __init__(self, mat: np.ndarray, invertible_size: int,**kwargs) -> None:
        self.org_size = mat.shape[0]
        mat = self.extend(mat,invertible_size)
        super().__init__(mat, invertible_size,**kwargs)
    @staticmethod
    def get_expansion_size(mat,invsize,):
        k = np.log2(mat.shape[0]) - np.log2(invsize) - 1
        k = int(np.ceil(k))
        expansion = (2**(k+1))*invsize - mat.shape[0]
        return expansion
    @staticmethod
    def extend(mat,invsize):
        expansion = SizeChagingHierarchicalInversion.get_expansion_size(mat,invsize)
        if expansion == 0:
            return mat
        z = np.zeros((mat.shape[0],expansion))
        e = np.eye(expansion)
        return np.block(
            [[mat,z],[z.T,e]]
        )
    @staticmethod
    def submatrix(mat,arr0,arr1):
        arr0 = slice(arr0[0],arr0[-1] + 1)
        arr1= slice(arr1[0],arr1[-1] + 1)
        return mat[arr0,arr1]
    def invert(self,**kwargs):
        mat =  super().invert(**kwargs)
        arr = np.arange(self.org_size)
        return self.submatrix(mat,arr,arr)






import scipy.sparse as sp


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


def coo_submatrix_push(mat,matr, r0, c0):
    """
    Pulls out an arbitrary i.e. non-contiguous submatrix out of
    a sparse.coo_matrix. 
    """
    if type(matr) != sp.coo_matrix or type(mat) != sp.coo_matrix:        
        raise TypeError(f'Matrix must be sparse COOrdinate format, type is {type(mat),type(matr)}')
        
    r1 = matr.shape[0] + r0
    c1 = matr.shape[1] + c0
    
   
    mrow = mat.row
    mcol = mat.col
    data = mat.data
    
    newelem = (mrow >= r0) & (mrow < r1) & (mcol >= c0) & (mcol < c1)

    mrow = np.delete(mrow,newelem)
    mcol = np.delete(mcol,newelem)
    data = np.delete(data,newelem)

    mrow = np.concatenate([mrow, matr.row  + r0])
    mcol = np.concatenate([mcol, matr.col  + c0])
    data = np.concatenate([data, matr.data ])
    
    lr,lc = mat.shape
    return sp.coo_matrix((data, np.array([mrow,
        mcol])),(lr, lc))

# class SparseSubmatrixSetter:
#     def __init__(self,mat:sp.coo_matrix,) -> None:
#         self.mat = mat
#     def __call__(self,mat:sp.coo_matrix,rowi:int,coli:int):
#         self.mat = coo_submatrix_push(self.mat,mat,rowi,coli)
#         return self.mat
    


class SparseHierarchicalInversion(SizeChagingHierarchicalInversion):
    mat : sp.coo_matrix
    def __init__(self, mat: sp.coo_matrix, invertible_size: int,\
                        tol:float = 1e-3,save_dir:str = '',\
                            milestones :List[float] = (0,0.5),\
                                continue_flag:bool  = False,\
                                **kwargs) -> None:
        if not isinstance(mat,sp.coo_matrix):
            raise Exception
        
        self.save_dir = save_dir
        self.continue_flag = continue_flag
        self.latest_arr = ()
        self.continue_file = ''
        
        if continue_flag:
            arr,ltst = self.find_latest_progression()
            self.latest_arr = arr
            self.continue_file = ltst
            mat_ = self.load_from_file()   
            if mat_ is not None:
                mat = mat_
            else:
                self.continue_flag = False
            
        super().__init__(mat, invertible_size,**kwargs)
        self.tol = tol

        self.milestones = milestones
        self.cur_milestone_index = 0
        
        
        # self.sub_setter = SparseSubmatrixSetter(mat)
    @staticmethod
    def extend(mat:sp.coo_matrix,invsize:int):
        expansion = SizeChagingHierarchicalInversion.get_expansion_size(mat,invsize)
        if expansion == 0:
            return mat
        return sp.bmat(
            [[mat,None],[None,sp.eye(expansion)]]
        ).tocoo()
    @staticmethod
    def submatrix(mat:sp.coo_matrix,arr0,arr1):
        return coo_submatrix_pull(mat.tocoo(),arr0,arr1)
    @staticmethod
    def merge_quad(a:sp.csr_matrix,b:sp.csr_matrix,c:sp.csr_matrix,d:sp.csr_matrix):
        mother_inverse = sp.bmat(
            [[a,b],[c,d]]
        )
        return mother_inverse

    def invert_mat(self,mat:sp.csr_matrix):
        mat = mat.toarray()
        mat = np.linalg.inv(mat)
        mat = np.where(np.abs(mat)<=self.tol,0,mat)        
        return sp.csr_matrix(mat)
    
    
    def __getitem__(self,arr:Tuple[int,...])->sp.coo_matrix:
        if not bool(arr):
            return self.mat
        slc = self.arr2slc(arr)
        x = np.arange(slc.start,slc.stop)
        return coo_submatrix_pull(self.mat,x,x)
        # matarr = self.mat.toarray()
        # return sp.coo_matrix(matarr[slc,slc])
    def __setitem__(self,arr:Tuple[int,...],mat:sp.csr_matrix):
        if not bool(arr):
            self.mat = mat
            return
        slc = self.arr2slc(arr)
        self.mat =  coo_submatrix_push(self.mat,mat.tocoo(),slc.start,slc.start)
        
        
    @staticmethod
    def divide(mat:sp.csr_matrix):
        hside = mat.shape[0]//2
        s0,s1 = np.arange(0,hside),np.arange(hside,2*hside)
        mat = mat.tocoo()
        m00 = coo_submatrix_pull(mat,s0,s0)
        m01 =  coo_submatrix_pull(mat,s0,s1)
        m10 =  coo_submatrix_pull(mat,s1,s0)
        m11 =  coo_submatrix_pull(mat,s1,s1)
        return m00.tocsr(),m01.tocsr(),m10.tocsr(),m11.tocsr()
    def save_to_file(self,arr):
        
        arrst = ''.join([str(x) for x in arr])
        fl = os.path.join(self.save_dir,'spmat_' + arrst + '.npz')
        sp.save_npz(fl,self.mat)
    def load_from_file(self,):
        _,ltst = self.find_latest_progression()
        if not bool(ltst):
            return None
        return sp.load_npz(os.path.join(self.save_dir,ltst))
    def find_latest_progression(self,):
        fls = os.listdir(self.save_dir)
        starrs = [fl.replace('spmat_','').replace('.npz','') for fl  in fls if 'spmat_' in fl and '.npz' in fl]
        if not bool(starrs):
            return -1,''
        arrs = [np.array([int(x) for x in arrst]) for arrst in starrs]
        progs = [arr @ (2**np.arange(len(arr))[::-1])/np.sum(2**np.arange(len(arr))[::-1]) for arr in arrs]
        
        i = np.argmax(progs)
        
        arr = tuple(arrs[i].tolist())
        return arr,fls[i]
    def act(self, arr: Tuple[int, ...]):
        prog = self.task_portion(arr)        
        if self.continue_flag:
            if arr != self.latest_arr:
                return None
            self.continue_flag = False
            self.cur_milestone_index = 0
            while self.cur_milestone_index < len(self.milestones):
                curm = self.milestones[self.cur_milestone_index]
                if prog > curm:
                    self.cur_milestone_index += 1
                else:
                    break
                    
        else:
            curm = self.milestones[self.cur_milestone_index]            
            if prog > curm:
                self.cur_milestone_index += 1
                self.save_to_file(arr)
            
        outputs =  super().act(arr)
        
        return outputs
    