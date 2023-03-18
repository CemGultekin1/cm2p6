from typing import List
from transforms.krylov import growing_orthogonals_decomposition
import numpy as np

class PseudoInvertibleMatmultBase:
    def __call__(self,x:np.ndarray,inverse = False)->np.ndarray:...
    def reshaper(self,x:np.ndarray)->np.ndarray:...
class PseudoInvertibleMatmult(PseudoInvertibleMatmultBase):
    def __init__(self,mat) -> None:
        self.mat = mat
        q,r = np.linalg.qr(mat.T)
        self.pseudo_inv_mat = q@np.linalg.inv(r.T)

    def __call__(self,x:np.ndarray,inverse = False)->np.ndarray:
        if inverse:
            mat = self.pseudo_inv_mat
        else:
            mat = self.mat
        return mat @ x
class MultiGmres:
    def __init__(self,invertiblematmult_list:List[PseudoInvertibleMatmultBase],rhs:np.ndarray,maxiter:int = 100) -> None:
        self.pim_list = invertiblematmult_list
        self.rhs = rhs
        self.rank = len(self.pim_list)
        self.growing_orthogonals_decomposition = growing_orthogonals_decomposition()
        self.reg_lambda = 1e-7
        self.maxiter = maxiter
    def find_best_fit(self,y:np.ndarray):
        xmat = [pim(y,inverse=True) for pim in self.pim_list]
        ymat = np.stack([np.add.reduce([pim(x,inverse = False) for pim in self.pim_list]).flatten() for x in xmat],axis = 1)
        ymatymat = ymat.T@ymat
        ymaty = ymat.T@y.flatten()
        coeffs = self.solve_linear_system(ymatymat,ymaty)
        yopt =ymat@coeffs
        xopt = np.add.reduce([x*c for x,c in zip(xmat,coeffs)])
        return xopt.flatten(),yopt
    def get_solution(self,xs,ys,e,iternum):
        n = len(xs)
        if n > 1:
            rinv = np.linalg.inv(self.growing_orthogonals_decomposition.rmat[:n-1,:n-1])
            rinv[1:,:-1] -= np.eye(rinv.shape[0] - 1)
            xmat = np.stack(xs[1:],axis = 1)
            ymat = np.stack(ys[1:],axis = 1)
            z0 = np.zeros(rinv.shape[0])
            z0[0] = 1
            print(f'np.linalg.cond(rinv)) = {np.linalg.cond(rinv)}')
            w0 = np.linalg.solve(rinv,z0)
            err = w0[-1]*e 
            xstar = xmat @  w0 + xs[0]
            ystar = ymat @  w0 + ys[0]
        else:
            err = e
            ystar = ys[0]
            xstar = xs[0]
        print(f'iternum = {iternum}, r2 = {1 - np.linalg.norm(err)/np.linalg.norm(self.rhs)}')
      
        
        return xstar, ystar
    def solve(self,):
        y = self.rhs
        xs = []
        ys = []
        iternum = 0
        while iternum < self.maxiter:
            iternum += 1
            xopt,yopt = self.find_best_fit(y)
            e = y - yopt
            if iternum < self.maxiter:
                continue_flag =  self.growing_orthogonals_decomposition.add(e,tol = 1e-8)
            else:
                continue_flag = False
            xs.append(xopt)
            ys.append(yopt)
            print(iternum)
            if continue_flag:
                y = self.growing_orthogonals_decomposition.qmat[:,-1]
            xstar,ystar = self.get_solution(xs,ys,e,iternum)
            if not continue_flag:
                break
        return xstar,ystar
    def solve_linear_system(self,ymatymat,ymaty):
        halfymat =np.linalg.cholesky(ymatymat + self.reg_lambda*np.eye(ymatymat.shape[0]))
        coeffs = np.linalg.solve(halfymat.T,np.linalg.solve(halfymat,ymaty))
        
        # u,s,vh = np.linalg.svd(ymatymat,full_matrices = False)
        # invs = np.diag(1/np.where(s/s[0]<self.reg_lambda,np.inf,s))
        # coeffs = vh.T@(invs@(u.T@ymaty))
        return coeffs

class MultiGmresForFiltering(MultiGmres):
    def find_best_fit(self, y: np.ndarray):
        y = self.pim_list[0].reshaper(y)
        xmat = [pim(y,inverse=True) for pim in self.pim_list]
        ymat = np.stack([np.add.reduce([pim(x,inverse = False) for pim in self.pim_list]).flatten() for x in xmat],axis = 1)
        ymatymat = ymat.T@ymat
        ymaty = ymat.T@y.flatten()
        coeffs = self.solve_linear_system(ymatymat,ymaty)
        yopt =ymat@coeffs
        xopt = np.add.reduce([x*c for x,c in zip(xmat,coeffs)])
        return xopt.flatten(),yopt

def main():
    pims = [PseudoInvertibleMatmult(np.random.randn(50,500)) for _ in range(1)]
    mgmres = MultiGmres(pims,np.random.randn(50,),)
    mgmres.solve()
    
if __name__ == '__main__':
    main()