from copy import deepcopy
import itertools
import logging
import os
from typing import List, Union
from metrics.geomean import WetMaskCollector
from models.load import get_statedict

from utils.arguments import options
from plots.metrics_ import metrics_dataset
from constants.paths import EVALS
from plots.murray import MurrayPlotter, MurrayWithSubplots
import xarray as xr
from utils.slurm import ArgsFinder, read_args
from utils.xarray_oper import drop_unused_coords, sel_available, select_coords_by_extremum
import os
from plots.metrics_ import metrics_dataset
from constants.paths import EVALS
import xarray as xr
from models.load import get_statedict
import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level = logging.INFO)
def load_r2map(linenum:int):
    args = read_args(linenum,filename = 'offline_sweep2.txt')
    _,_,_,modelid = get_statedict(args)
    path =  os.path.join(EVALS,modelid + '.nc')
    assert os.path.exists(path)
    ds = xr.open_dataset(path).isel(depth = 0,co2 = 0)
    return metrics_dataset(ds,dim = [])

class ModelLookUp:
    def __init__(self,from_file:bool,args:str) -> None:
        self.from_file = from_file
        self.args = args        
    def get_modelid(self,):
        # logging.info(f'args = {self.args}')
        _,modelid = options(self.args.split(),key = "model")
        return modelid
    def read_dataset(self,):
        modelid = self.get_modelid()
        path =  os.path.join(EVALS,modelid + '.nc')
        assert os.path.exists(path)
        # logging.info(f'reading dataset for line = {self.args}')
        ds = xr.open_dataset(path)
        return ds
class MostSimilarR2Map(ArgsFinder):
    def __init__(self, filename: str):
        super().__init__(filename)
        self.model_look_up = None
    def lookup(self,model_look_up:ModelLookUp):
        self.model_look_up = model_look_up
        self.find_true_line()
        return model_look_up.read_dataset()
    def find_true_line(self,):
        if self.model_look_up.from_file:
            lines = self.find_fits(self.model_look_up.args,key = 'model')
            logging.info(f' number of matching lines = {len(lines)}')
            if len(lines) == 0:
                # logging.info(f'args = \n\t{self.model_look_up.args}')
                raise Exception
            # elif len(lines) > 1:
            #     logging.info('\n\n'.join(lines))
            self.model_look_up.args = lines[0]
    
class MixedModelR2Map:
    def __init__(self,filename:str,**test_kwargs) -> None:
        self.reader = MostSimilarR2Map(filename)        
        self.test_kwargs = test_kwargs
    def load_dataset(self,model_:str = 'fcnn',**training_kwargs:str):
        args = ' '.join([f'--{key} {val}' for key,val in training_kwargs.items()])
        # logging.info(f' model = {model_}, {model_ == "fcnn"}')
        lookup  = model_ == 'fcnn'
        mlu = ModelLookUp(lookup,args)
        self.reader.lookup(mlu)
        ds =  mlu.read_dataset()
        ds =  sel_available(ds,self.test_kwargs)
        d = dict()
        d.update(**training_kwargs,)
        d.update(**self.test_kwargs)
        return metrics_dataset(ds,dim = [])

class WetMasks(WetMaskCollector):
    def __init__(self) -> None:
        super().__init__()
        self.collection = {}
    def get_wet_mask(self,sigma:int,stencil:int,depth:Union[float,int]):
        d = int(depth)
        x = (sigma,stencil,d)   
        if x not in self.collection:                 
            wetmask = super().get_wet_mask(sigma,stencil,sel_depth = depth)
            self.collection[x] = wetmask
        return self.collection[x]
    def mask_dataset(self,x,sigma,stencil,depth):
        wm = self.get_wet_mask(sigma,stencil,depth)
        wm1 = select_coords_by_extremum(wm,x.coords,'lat lon'.split())
        x = xr.where(wm1, x,np.nan)
        return x
        
        
class TriModels:
    def __init__(self,filename:str,train_kwargs,test_kwargs,training_pick) -> None:
        self.test_kwargs = test_kwargs
        self.train_kwargs = train_kwargs
        for tp in training_pick:
            self.test_kwargs[tp] = train_kwargs[tp]
        self.filename = filename
        self.datasets = {}
        self.wetmasks = WetMasks()
    def load_datasets(self,):
        newkwargs = deepcopy(self.train_kwargs)
        glb = 'global'
        r4 = 'four_regions'
        lin = 'linear'
        dom = 'domain'
        text = 'offline_sweep2.txt'
        fcnn = 'fcnn'

        mmrm = MixedModelR2Map(text,**self.test_kwargs)
        newkwargs[dom] = r4
        self.datasets[r4] =  mmrm.load_dataset(model_ = fcnn,**newkwargs)
        
        newkwargs[dom] = glb
        self.datasets[glb] =  mmrm.load_dataset(model_ = fcnn,**newkwargs)
        
        linear_args = {k:v for k,v in newkwargs.items() if k in 'sigma filtering'.split()}
        linear_args.update(dict(model = 'lsrp:0'))
        self.datasets[lin] = mmrm.load_dataset(model_ = lin,**linear_args)

    def give_values(self,forcing:str = 'Su', ext:str = '_r2'):
        metric = forcing + ext 
        datarrs = {}
        for key,val in self.datasets.items():
            datarr = val[metric] 
            datarr =self.wetmasks.mask_dataset(datarr, self.train_kwargs['sigma'],1,self.test_kwargs['depth'])
            datarrs[key] = datarr
        return datarrs

class DictionaryMultiplier:
    def __init__(self,argsdict:dict) -> None:
        self.argsdict = argsdict
        for key,val in self.argsdict.items():
            if np.isscalar(val):
                self.argsdict[key] = [val]
                
    def iterate_dict(self,):
        keys = list(self.argsdict.keys())
        for vals in itertools.product(*tuple(self.argsdict.values())):
            seldict = dict(zip(keys,vals))
            yield seldict


class NamingScheme:
    def __init__(self,training_kwargs,testing_kwargs):
        mltpnames0 = [key for key,val in training_kwargs.items() if isinstance(val,(list,tuple))]
        mltpnames1 = [key for key,val in testing_kwargs.items() if isinstance(val,(list,tuple))]
        self.multipnames = mltpnames0 + mltpnames1


    def shorten(self,name:str):
        vowels = ['a','e','i','o','u']
        return ''.join([l for l in name if l not in vowels])
    def dict_to_name(self,trainsel):        
        name = []
        numlim = 4
        for key,val in trainsel.items():
            if key not in self.multipnames:
                continue
            shrt = self.shorten(key)
            name.append(shrt[:3])
            if isinstance(val,float):
                valstr = "{:.2f}".format(val).replace('.','p')
            elif isinstance(val,(int,bool)):
                valstr = str(val)
            elif isinstance(val,str):
                valstr = val
            else:
                continue
            valstr = valstr.replace('_','')
            valstr1 = self.shorten(valstr)
            name.append(valstr1[:numlim])        
        return '_'.join(name)
    def get_name(self,trkwargs,tstkwargs):
        name0 = self.dict_to_name(trkwargs)
        name1 = self.dict_to_name(tstkwargs)
        return '_'.join([name0,name1])
    
        
                
from data.coords import DEPTHS        

def main():
    training_kwargs = dict(
        depth = 0,
        filtering = ['gcm'],
        temperature = True,
        lossfun = ['heteroscedastic','MSE'],
        sigma = [4,8,12,16],
        domain = 'four_regions'
    )
    
    testing_kwargs = dict(            
        filtering = ['gcm',],
        co2 = [0.,0.01],
    )
    pick_from_training = ['depth']
    
    naming_scheme = NamingScheme(training_kwargs,testing_kwargs)
    training_multips = DictionaryMultiplier(training_kwargs)
    testing_multips = DictionaryMultiplier(testing_kwargs)
    
    
    for trdict in training_multips.iterate_dict():
        for tstdict in testing_multips.iterate_dict():
            trim = TriModels('offline_sweep2.txt',trdict,tstdict,pick_from_training)
            try:
                trim.load_datasets()
            except:
                continue
            for forcing in 'Su Sv Stemp'.split():
                darr = trim.give_values(forcing = forcing)
                name = naming_scheme.get_name(trdict,tstdict)
                plot(darr,trdict['sigma'],name,forcing)
            # return
    
    
class ColorbarTracker:
    def __init__(self,folder:str) -> None:
        self.root = folder
        self.saved = False
    def is_there_any(self,):
        files = os.listdir(self.root)
        files = [file for file in files if 'colorbar' in file]
        return bool(files)
    
def plot(data,sigma,filename,ftype):
    target_folder = 'paper_images/r2maps'
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    ctrac = ColorbarTracker(target_folder)
    import matplotlib
    matplotlib.rcParams.update({'font.size': 14})
    kwargs = dict(
        vmin = 0,
        vmax = 1,
        set_bad_alpha = 0.,
        projection_flag = True,
        sigma = sigma,
        cmap = matplotlib.cm.magma,
        shading = 'gouraud'
    )
    for source in data.keys():
        u = data[source]
        mp = MurrayWithSubplots(1,1,xmargs = (0.1,0.,0.02),ymargs = (0.06,0.,0.),figsize = (8,3.5),)
        _,ax,cs = mp.plot(0,0,u,title = None,**kwargs) 
        ax.set_facecolor('black')
        fpng = f'{filename}_{source}_{ftype}.png'
        path1 = os.path.join(target_folder,fpng)
        print(path1)
        mp.save(path1,transparent=False)        
        # raise Exception
       
        if not ctrac.is_there_any():
            mp = MurrayWithSubplots(1,1,xmargs = (0.,0.,0.55),ymargs = (0.06,0.,0.),figsize = (0.75,3.5),)
            mp.plot_colorbar(0,0,cs,)
            cbarfilename = f'colorbar.png'
            path1 = os.path.join(target_folder,cbarfilename)
            print(path1)
            mp.save(path1,transparent=False)

if __name__ == '__main__':
    main()