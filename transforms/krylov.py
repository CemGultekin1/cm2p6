import math
import numpy as np
class iterative_inversion:
    def __init__(self,degree) -> None:
        self.degree = degree
        pass
    def invert(self,forward_pass:callable,pseudo_inverse:callable,x0):
        '''
        Returns approximate inverse of x0
        '''
        x = x0.copy()
        m = self.degree
        y = math.comb(m, 1 + 0)*math.pow(-1,0)*x
        for i in range(1,m):
            xinv = pseudo_inverse(x)
            x = forward_pass(xinv)
            y += math.comb(m, 1 + i)*math.pow(-1,i)*x
        xinv = pseudo_inverse(y)
        return xinv
    
class growing_orthogonals_decomposition:
    def __init__(self) -> None:
        self.qmat = None
        self.rmat = None
    def q_orthogonal(self,v:np.ndarray):
        if self.qmat is None:
            return None,v
        r = self.qmat.T@v
        return r,v - self.qmat@r
    def r_grow(self,r,n):
        if r is None:
            return np.eye(1)*n
        m = self.rmat.shape[0] + 1
        rr = np.empty((m,m))
        rr[:-1,:-1] = self.rmat
        rr[-1,:] = 0
        rr[:-1,-1] = r
        rr[-1,-1] = n
        return rr
    def q_grow(self,v):
        v = v.reshape([-1,1])
        if self.qmat is None:
            return v
        return np.concatenate([self.qmat,v],axis = 1)
    def __len__(self,):
        if self.qmat is None:
            return 0
        return self.qmat.shape[1]
    def add(self,v,tol = 1e-5):    
        n0 = np.linalg.norm(v)
        r,v_orth =self.q_orthogonal(v)
        n = np.linalg.norm(v_orth)
        relnorm = n/n0
        if relnorm < tol:
            return False
        v_orth = v_orth/n
        self.qmat = self.q_grow(v_orth)
        self.rmat = self.r_grow(r,n)
        return True
    def res(self,v):
        return self.q_orthogonal(v)[1]
    def nres(self,v):
        return np.linalg.norm(self.res(v))
    def solve(self,v):
        print(f'type(self.rmat),type(self.qmat),type(v) = {type(self.rmat),type(self.qmat),type(v)}')
        return np.linalg.solve(self.rmat,self.qmat.T@v)
    def orthogonality(self,):
        return np.linalg.norm(self.qmat.T @ self.qmat - np.eye(self.qmat.shape[1]))

class krylov_inversion(growing_orthogonals_decomposition):
    def __init__(self,maxiter,reltol,implicit_matmultip) -> None:
        super().__init__()
        self.maxiter = maxiter
        self.reltol = reltol
        self.implicit_matmultip = implicit_matmultip
        self.iterates = []
    def add(self, x):
        y = self.implicit_matmultip(x)
        if super().add(y):
            self.iterates.append(x)
    def solve(self,y:np.ndarray):
        assert isinstance(y,np.ndarray)
        self.add(y)
        nres = [self.nres(y)]
        i = 0
        while i < self.maxiter and self.reltol < nres[-1]/nres[0]:
            x = self.implicit_matmultip(self.qmat[:,-1])
            self.add(x)
            nres.append(self.nres(y))
            i+=1
            print(f'\t\t{i}')
            if nres[-1]/nres[0] > 1:
                break
        assert isinstance(y,np.ndarray)
        coeffs = super().solve(y)
        return np.stack(self.iterates,axis=1)@coeffs


        
def test_krylov_inversion():
    np.random.seed(0)
    d = 256
    mat = np.random.randn(d,d)
    q,r = np.linalg.qr(mat)
    signdiag = np.diag( (np.diag(r) > 0)*2 -1 ) 
    q = q@signdiag
    r = signdiag@r
    qrerr = np.sum(np.square(q@r - mat))
    print(f'qrerr = {qrerr}')

    god = growing_orthogonals_decomposition()

    
    for i in range(mat.shape[1]):
        god.add(mat[:,i])
    qerr = np.sum(np.square(god.qmat - q))
    rerr = np.sum(np.square(god.rmat - r))
    print(f'qerr,rerr:{qerr,rerr}')
    print(f'orthogonality:\t{god.orthogonality()}')

    y = np.random.randn(d)
    god = growing_orthogonals_decomposition()

    
    def matmultip(x_):
        return mat@x_
    gres = krylov_inversion(d,1e-2,matmultip)
    sltn = gres.solve(y)


if __name__ == "__main__":
    test_krylov_inversion()