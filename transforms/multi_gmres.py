from typing import List, Optional
from transforms.krylov import growing_orthogonals_decomposition
import numpy as np

class MultiLinearOps:
    rank:int
    def __call__(self,x:np.ndarray,inverse = False,separated = False,special:int = -1):...
    def projection(self,x:np.ndarray,special:int = -1):...
class MultiLinearMats(MultiLinearOps):
    def __init__(self,mats) -> None:
        self.mats = mats
        self.rank = len(mats)
        self.pseudo_inv_mat = []
        self.input_shape,self.output_shape = mats[0].shape
        for mat in mats:
            u,s,vh = np.linalg.svd(mat,full_matrices = False)
            sinv= np.diag(1/np.where(s/s[0]>1e-2,s,np.inf))
            self.pseudo_inv_mat.append(vh.T @ sinv @ u.T)

    def __call__(self,x:np.ndarray,inverse = False,separated= False,special = -1):
        matlist = self.mats if not inverse else self.pseudo_inv_mat
        shape = self.input_shape if not inverse else self.output_shape
        if special >= 0:
            mat = matlist[special]
            matx =  mat@x
            if separated:
                return [matx]
            else:
                return matx
        if separated:
            y = []
        else:
            y = np.zeros(shape)
        for  mat in matlist:
            matx = mat @ x
            if separated:
                y.append(matx)
            else:
                y += matx
        return y
            
class MultiGmres:
    def __init__(self,linear_ops:MultiLinearOps,rhs:np.ndarray,maxiter:int = 100,reltol:float = 1e-2, sufficient_decay_limit:float = 0.9) -> None:
        self.linear_ops = linear_ops
        self.rhs = rhs
        self.rank = linear_ops.rank
        self.growing_orthogonals_decomposition = growing_orthogonals_decomposition()
        self.reg_lambda = 1e-9
        self.reltol = reltol
        self.maxiter = maxiter
        self.sufficient_decay_limit = sufficient_decay_limit
        self.iternum = 0
    def find_best_fit(self,y:np.ndarray):
        xmat = self.linear_ops(y,inverse = True,separated = True)
        ymat = [self.linear_ops(x,inverse = False,separated = False) for x in xmat]
        ymat = np.stack([ymat_.flatten() for ymat_ in ymat],axis = 1)
        ymatymat = ymat.T@ymat
        ymaty = ymat.T@y
        coeffs = self.solve_linear_system(ymatymat,ymaty)
        yopt =ymat@coeffs 
        xopt = np.add.reduce([x*c for x,c in zip(xmat,coeffs)])
        return xopt.flatten(),yopt
    def get_solution(self,xs,ys,):
        n = len(xs)
        if n > 1:
            xmat = np.stack(xs,axis = 1) 
            ymat = np.stack(ys,axis = 1)
            w0 = np.linalg.solve(ymat.T@ymat + self.reg_lambda*np.eye(ymat.shape[1]),\
                            ymat.T@self.rhs)
            xstar = xmat @  w0 #+ xs[0]
            ystar = ymat @  w0 #+ ys[0]
        else:
            ystar = ys[0]
            xstar = xs[0]
        err = self.rhs - ystar
        relerr = np.linalg.norm(err)/np.linalg.norm(self.rhs)
        return xstar, ystar,relerr
    def solve(self):
        e = self.rhs.copy()
        xs = []
        ys = []
        relerrs = []
        self.iternum = 0
        while self.iternum < self.maxiter:
            self.iternum += 1
            xopt,yopt = self.find_best_fit(e)
            e = e - yopt
            if self.iternum < self.maxiter:
                continue_flag =  self.growing_orthogonals_decomposition.add(e,tol = 1e-12)
            else:
                continue_flag = False
            xs.append(xopt)
            ys.append(yopt)
            
            if continue_flag:
                e = self.growing_orthogonals_decomposition.qmat[:,-1]
            xstar,ystar,relerr = self.get_solution(xs,ys)
            print(self.iternum,relerr)
            # yield 
            relerrs.append(relerr)
            if not continue_flag:
                break
            if relerrs[-1]< self.reltol:
                break
            if len(relerrs) == 1:
                continue
            if relerrs[-1]/relerrs[-2] > self.sufficient_decay_limit:
                break
        return xstar,ystar,relerr
    def solve_linear_system(self,ymatymat,ymaty):
        halfymat =np.linalg.cholesky(ymatymat + self.reg_lambda*np.eye(ymatymat.shape[0]))
        coeffs = np.linalg.solve(halfymat.T,np.linalg.solve(halfymat,ymaty))
        return coeffs


def main():
    d1,r,d2 = 600,100,2400
    m = 64
    np.random.seed(0)
    mats = [np.random.randn(d1,r)@np.random.randn(r,d2) for _ in range(m)]
    mlm = MultiLinearMats(mats)
    mgmres = MultiGmres(mlm,np.random.randn(d1,),maxiter = m*2,sufficient_decay_limit=0.99)
    # mgmres = AlternatingMultiGmres(mlm,np.random.randn(d1,),maxiter = m*m,sufficient_decay_limit=np.inf)
    mgmres.solve(initial_operator=0)
    
if __name__ == '__main__':
    main()