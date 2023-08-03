import logging
import os
from typing import List, Tuple
from constants.paths import OUTPUTS_PATH
import numpy as np
import scipy.sparse as sp
from scipy.ndimage import gaussian_filter as gfilt
import multiprocessing

class LinFun:
    indim:int
    outdim:int
    def __call__(self,x:np.ndarray)->np.ndarray:...
    def picklable_arguments(self,):
        return self.indim,self.outdim
    @classmethod
    def from_picklable_arguments(cls,*args)->'LinFun':...
    def post__init__(self,*args):...
class FilteringCoarseGraining:
    def __init__(self,sigma:int,inshape:Tuple[int,...]) -> None:
        self.sigma = sigma
        self.inshape = inshape
        self.indim = np.prod(inshape)
        outshape = np.array(inshape)//sigma
        self.outdim = np.prod(outshape)
    def picklable_arguments(self,):
        return self.sigma,self.inshape,self.indim,self.outdim
    @classmethod
    def from_picklable_arguments(cls,sigma,inshape,indim,outdim):
        fcg = FilteringCoarseGraining.__new__(FilteringCoarseGraining,)
        fcg.sigma = sigma
        fcg.inshape = inshape
        fcg.indim = indim
        fcg.outdim = outdim
        return fcg
    def axis_cg(self,y:np.ndarray,axis :int):
        return np.stack([np.mean(y_,axis = axis) for y_ in np.split(y,y.shape[axis]//self.sigma,axis = axis)],axis = axis)
    def __call__(self,x:np.ndarray):
        x = x.reshape(self.inshape)
        assert self.indim == x.size
        y  = gfilt(x,self.sigma/2,mode = 'wrap',radius = 2*self.sigma)
        y = y/gfilt(x*0 + 1,self.sigma/2,mode = 'wrap',radius = 2*self.sigma)
        for axis in range(y.ndim):
            y = self.axis_cg(y,axis)
        assert y.size == self.outdim
        return y.flatten()
    
    
    
def partition_a_range(arr,parti,numparts):
    return np.array_split(arr,numparts)[parti]



def single_cpu_process(kwargsdict):
    defdict = dict(
        part_ind = 0,tot_parts = 1,linfuncls = LinFun,args= (),tol = 1e-9
    )
    defdict.update(kwargsdict)
    args = defdict['args']
    linfuncls = defdict['linfuncls']
    tol = defdict['tol']
    part_ind = defdict['part_ind']
    cpu_ind = defdict['cpu_ind']
    tot_parts = defdict['tot_parts']
    path = defdict['path']
    last_num_el = 1
    '''
    cpu_ind
    part_ind
    args
    linfuncls
    tot_parts
    tol
    path
    '''
    lf = linfuncls.from_picklable_arguments(*args)
    lf.post__init__()
    def basis_element(i:int):
        x = np.zeros((lf.indim,))
        x[i] = 1
        return x    
    dok_array = sp.dok_array((lf.outdim,lf.indim))
    parts = partition_a_range(np.arange(lf.indim),part_ind,tot_parts)
    for enumi,i in enumerate(parts):
        be = basis_element(i)
        yi = lf(be)
        yi = np.where(np.abs(yi)>tol,yi,0)
        inds = np.where(yi != 0,)[0]
        yi = yi[inds]
        for yii,indsi in zip(yi,inds):
            dok_array[indsi,i] = yii
        if (dok_array.nnz - last_num_el)/ last_num_el > 1e-1 or enumi%50 == 0:
            if cpu_ind == 0:                
                formatter = "{:.2e}"
                logging.info(f'\t\t{i - parts[0]}/{len(parts)} = {formatter.format((i - parts[0])/len(parts))},\t total # of entries = {dok_array.nnz}')
            if enumi%200 == 0:
                sp.save_npz(path,dok_array.tocsc())
            last_num_el = dok_array.nnz
    return dok_array

class SparseVecCollection:
    def __init__(self,linfun:LinFun,fileroot:str,tol = 1e-5,ncpu:int = multiprocessing.cpu_count(),partition:Tuple[int,int] = (0,1)) -> None:
        self.linfun= linfun
        self.tol = tol
        self.fileroot = fileroot
        self.ncpu = ncpu
        
        part_ind,tot_parts = partition
        logging.basicConfig(level=logging.INFO,\
                format = '%(asctime)s %(message)s',)
        
        logging.info(f'ncpu ={ncpu}')
        self.partition = partition
    def single_cpu_kwargs(self,i:int):
        part_ind = self.partition[0] * self.ncpu + i
        tot_parts = self.ncpu*self.partition[1]
        kwargs = dict(
                cpu_ind = i,
                part_ind = part_ind,
                args = self.linfun.picklable_arguments(),
                linfuncls = self.linfun.__class__,
                tot_parts = tot_parts,
                tol = self.tol,
                path = CollectParts.to_filename(self.fileroot,part_ind,tot_parts)
                #self.fileroot.split('.')[0] + f'-part-{part_ind}-{tot_parts}.npz',
        )
        return kwargs
    def collect_basis_elements(self,):
        kwargss = [self.single_cpu_kwargs(i) for i in range(self.ncpu)]
        
        pool = multiprocessing.Pool(self.ncpu)
        _ = list(pool.map(single_cpu_process, kwargss,))
        
    def basis_element(self,i:int):
        x = np.zeros((self.linfun.indim,))
        x[i] = 1
        return x
    def save(self,):
        sp.save_npz(self.path,self.dok_array.tocsc())
    def load(self,):
        self.dok_array = sp.load_npz(self.path,)

class CollectParts:
    def __init__(self,root:str) -> None:
        self.root = root
    def all_parts_collect(self,head:str = ''):
        root = self.root
        flist = os.listdir(root)
        splist = [CollectParts.separate(fl) for fl in flist if CollectParts.is_conformal(fl)]
        heads, _, num_parts,flist = tuple(zip(
            *splist
        ))
        uheads = np.unique(heads)
        united_files = []
        for uh in uheads:
            if head not in uh:
                continue
            uhlist = []
            for hd,nump,fl in zip(heads,num_parts,flist):
                if hd != uh:
                    continue
                uhlist.append(os.path.join(root,fl))
            if nump > len(uhlist):
                continue
            path = CollectParts.unite_all(root,uh,uhlist)
            united_files.append(path)
        return united_files
    @classmethod
    def unite_all(cls,root,head,files):
        dok_array = CollectParts.unite_all_sparse(files)
        united_file = CollectParts.united_filename(head)
        path = os.path.join(root,united_file)
        if os.path.exists(path):
            i = 1
            path = path.replace('.npz',f'-{i}.npz')
            while os.path.exists(path):
                i+= 1
                path = path.replace(f'-{i-1}.npz',f'-{i}.npz')
        logging.info(f'\t\t saving to {path.split("/")[-1]} ')
        sp.save_npz(path,dok_array.tocsc())
        return path
    @classmethod
    def load_spmat(cls,path:str):
        return sp.load_npz(path)
    @classmethod
    def save_spmat(cls,path:str,mat):
        return sp.save_npz(path,mat.tocsc())
    @classmethod
    def united_filename(cls,head:str):
        return head + '.npz'
    @classmethod
    def latest_united_file(cls,path:str,head:str):
        files = os.listdir(path)
        nfiles = [fl for fl in files if not CollectParts.is_conformal(fl) and head in fl and '.npz' in fl]
        partnum = [fl.replace(head,'').split('-')[-1].replace('.npz','') for fl in nfiles]
        partnum = [0 if len(part) == 0  else int(part) for part in partnum]
        i = np.argmax(partnum)
        return  os.path.join(path,nfiles[i])
    @classmethod    
    def unite_all_sparse(cls,files:List[str]):
        i = 0
        
        dok_arr = sp.load_npz(files[i])
        logging.info(f'\t\t load_npz {i}/{len(files)}\t {files[i].split("/")[-1]} nnz = {dok_arr.nnz}')
        for i in range(1, len(files)):            
            dok_arr += sp.load_npz(files[i])
            logging.info(f'\t\t load_npz {i}/{len(files)}\t {files[i].split("/")[-1]} nnz = {dok_arr.nnz}')
        return dok_arr
                
                
    
    @classmethod
    def is_conformal(cls,path:str):
        if '-part-' not in path or '.npz' not in path:
            return False
        return True
    @classmethod
    def separate(cls,path:str):
        head = path.split('-part-')[0]
        tail = path.split('-part-')[1]
        part = tail.split('-')[0]
        totpart = tail.split('-')[1].split('.npz')[0]
        part,totpart = [int(x) for x in (part,totpart)]
        return head,part,totpart,path
    @classmethod
    def to_filename(cls,fileroot:str,part_ind,tot_parts):
        return fileroot.split('.')[0] + f'-part-{part_ind}-{tot_parts}.npz'
class SparseInversion(SparseVecCollection):
    def __init__(self,  path: str,tol :float = 1e-2) -> None:
        self.path = path
        self.load()
        self.tol = tol
       
    def compute_left_inverse(self,):
        sqdok = self.dok_array @ self.dok_array.T
        linv = self.dok_array.T @ sp.linalg.inv(sqdok)
        alinv = np.abs(linv)
        linv[alinv < self.tol] = 0
        return linv.T
    def save_left_inverse(self,):
        new_path = self.path.replace('.npz','-inverse.npz')
        inv = self.compute_left_inverse()
        numel = np.prod(inv.shape)
        logging.info(f'shape = {inv.shape},nnz = {inv.nnz}, sparsity = {inv.nnz/numel}')
        sp.save_npz(new_path,inv)
    def singular_vals(self,):
        dok = self.dok_array.toarray()
        dok = dok @ dok.T
        _,s,_ = np.linalg.svd(dok,full_matrices = False)
        return s

class OperatorPlotter:
    def __init__(self,path0,) -> None:
        self.path = path0
    def plot(self,fig,ax):
        z = sp.load_npz(self.path).toarray()
        z = np.log10(np.abs(z))
        pos = ax.imshow(z,cmap='Blues', interpolation='none')
        fig.colorbar(pos, ax=ax)
        
def main():
    logging.basicConfig(level=logging.INFO,\
                format = '%(asctime)s %(message)s',)
    root = os.path.join(OUTPUTS_PATH,'filter_weights')
    cp = CollectParts(root)
    flist = cp.all_parts_collect(head = 'gcm-dpth-0-sgm-4')
    logging.info('\n'.join(flist))
if __name__ == '__main__':
    main()