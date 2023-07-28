from collections import defaultdict
from copy import deepcopy
import itertools
import os
from typing import Any, Callable, List, Tuple
from data.load import load_xr_dataset
from data.coords import DEPTHS
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from constants.paths import OUTPUTS_PATH


def convert2absolutes(edges,dens):
    edges = np.abs(edges)
    z = np.argmin(edges)
    if edges[z]< 0:
        z += 1
    negdens = dens[:z]
    posdens = dens[z:]
    m = np.maximum(len(negdens),len(posdens)).astype(int)
    newdens = np.zeros((m,))
    newdens[:len(negdens)] += negdens[::-1]
    newdens[:len(posdens)] += posdens
    negedges = np.abs(edges[:z+1][::-1])
    posedges = np.abs(edges[z:])
    if len(negedges) < len(posedges):
        newedges = posedges
    else:
        newedges = negedges
    return newedges,newdens
def get_medges(edges):
    return (edges[1:] + edges[:-1])/2

class Histogram:
    def __init__(self,num_bins:int = 100,extremes:Tuple[float,float] = (0,0)) -> None:
        self.dx = (extremes[1] - extremes[0])/num_bins
        x0,x1 = extremes
        self.edges = np.linspace(x0,x1,num_bins+1)
        self.extremes = extremes
        self.num_bins = num_bins
        self.histogram = np.zeros((num_bins,))
        self.medges = (self.edges[1:] + self.edges[:-1])/2
    def feed(self,vals:np.ndarray):
        vals = vals[vals == vals].flatten()
        nvals = (vals - self.extremes[0])/self.dx
        nvals = np.floor(nvals).astype(int)
        nvals = nvals[nvals >= 0]
        nvals = nvals[nvals < self.num_bins]
        vals,counts = np.unique(nvals,return_counts = True)
        self.histogram[vals] += counts
        return self
    def to_absolutes(self,):
        edges,hist = convert2absolutes(self.edges,self.histogram)
        return Histogram.from_hist_edges(hist,edges)
    @classmethod
    def from_hist_edges(self,hist:np.ndarray,edges:np.ndarray):
        hs = Histogram.__new__(Histogram,)
        hs.edges = edges
        hs.num_bins = len(edges) - 1
        hs.extremes = tuple([np.amin(edges),np.amax(edges)])
        hs.medges = (hs.edges[1:] + hs.edges[:-1])/2
        hs.dx = edges[1] - edges[0]
        hs.histogram = hist
        return hs
    def cut_density(self,cutoff:float):
        density = self.histogram/np.sum(self.histogram)
        I = np.where(density > cutoff)[0]
        f0 = I[0]
        f1 = I[-1]
        return self.edges[f0],self.edges[f1+1]
        edges = self.edges[f0:f1 + 1]
        hist = self.histogram[f0:f1]
        return Histogram.from_hist_edges(hist,edges)
class AdaptiveHistogramInitiator:
    def __init__(self,initial_buffer_size :int = 100, num_bins:int = 100) -> None:
        self.initial_buffer_size = initial_buffer_size
        self.buffer = np.zeros((0,))
        self.num_bins = num_bins
    def add2buffer(self,vec:np.ndarray):
        self.buffer = np.concatenate([self.buffer,vec])
    def feed(self,values:np.ndarray):
        values = values[values==values].flatten()
        self.add2buffer(values)
        if self.buffer.size < self.initial_buffer_size:
            return self
        x0,x1 = np.amin(self.buffer),np.amax(self.buffer)
        # print(f'x0,x1 = {x0,x1}')
        m = np.amax([np.abs(x0),np.abs(x1)])/2
        x0 -= m
        x1 += m
        histogram =  Histogram(num_bins=self.num_bins,extremes=(x0,x1))
        histogram = histogram.feed(self.buffer)
        return histogram
class KeyedHistograms:
    def __init__(self,**kwargs) -> None:
        self.histdict = defaultdict(lambda : AdaptiveHistogramInitiator(**kwargs))
    def feed(self,key,values):
        x = self.histdict[key]
        adhflag = isinstance(x,AdaptiveHistogramInitiator)
        self.histdict[key] = x.feed(values)
        if adhflag and isinstance(self.histdict[key],Histogram):
            print(f'{key} has transitioned from AdaptiveHistogramInitiator to Histogram')
    @property
    def np_histograms_dictionary(self,):
        dict1 =  {
            key + '_histogram': val.histogram for key,val in self.histdict.items()
        }
        dict2 = {
            key + '_edges' : val.edges for key,val in self.histdict.items()
        }
        dict1.update(dict2)
        return dict1
    @classmethod
    def name_fix(cls,filename):
        if '.npz' not in filename:
            filename =filename.split('.')[0]
            filename += '.npz'
        return filename
    def save2npz(self,filename):
        filename = self.name_fix(filename)
        np.savez(filename,**self.np_histograms_dictionary)
    
    @classmethod
    def load_from_npz(cls,filename)->'KeyedHistograms':
        filename = cls.name_fix(filename)
        npz = np.load(filename)
        keys = list(npz.keys())
        keys = [key.replace('_histogram','') for key in keys if '_histogram' in key]
        kh = KeyedHistograms.__new__(KeyedHistograms,)
        kh.histdict = {}
        for key in keys:
            hkey = key + '_histogram'
            ekey = key + '_edges'
            hist = npz[hkey]
            edges = npz[ekey]
            kh.histdict[key] = Histogram.from_hist_edges(hist,edges)
        return kh

        
class Dataset(xr.Dataset):
    def __init__(self,ds:xr.Dataset) -> None:
        super().__init__(
            ds.data_vars,ds.coords
        )
    def mask_with_nan(self,):
        if 'wet_density' not in self.data_vars:
            return self
        return Dataset(xr.where(self.wet_density <0.5,np.nan,self))
    def drop_timeless_vars(self,):
        dropkeys = []
        for data,val in self.data_vars.items():
            if 'time' not in val.dims:
                dropkeys.append(data)
        return Dataset(self.drop(dropkeys))
    def drop_unrelated_vars(self,):
        dropkeys = []
        for data,val in self.data_vars.items():
            if data not in 'u v temp Su Sv Stemp'.split():
                dropkeys.append(data)
        return Dataset(self.drop(dropkeys))
    def separate_by_depth(self,depths):
        dss = []
        for depth in depths:
            dss.append(Dataset(self.sel(depth = depth,method = 'nearest')))
        return dss
class Datasets(dict):
    def __init__(self,**keyval_dict) -> None:
        dict.__init__(self,)
        self.keyval_dict = keyval_dict

    def iterate_choices(self,):
        keys = self.keyval_dict.keys()
        for values in itertools.product(*self.keyval_dict.values()):
            keyvaldict = dict(tuple(zip(keys,values)))
            args = self.to_args(keyvaldict)
            yield keyvaldict,args
    def to_args(self,keyvaldict,bz_correct: bool = True):
        keyvaldict_ = deepcopy(keyvaldict)
        if bz_correct:
            if keyvaldict['sigma'] == 1:
                keyvaldict_['depth'] = 0 if keyvaldict['depth'] == 0 else 5
                keyvaldict_['sigma'] = 4
        return ' '.join([f'--{key} {val}' for key,val in keyvaldict_.items()])
    def load_lowres_datasets(self,):
        for keyvaldict,args in self.iterate_choices():
            if keyvaldict['sigma'] == 1:
                continue            
            ds,_ = load_xr_dataset(args.split(),high_res=False)
            ds = Dataset(ds)
            ds = ds.mask_with_nan()
            ds = ds.drop_timeless_vars()
            ds = ds.drop_unrelated_vars()
            self[args] = ds
            
    def load_highres_datasets(self,):
        depths = self.keyval_dict['depth']
        bzdepths = [d for d in depths if d > 0]
        hreskeys = {}
        for keyvaldict,args in self.iterate_choices():
            if keyvaldict['sigma'] > 1:
                continue
            if args in hreskeys:
                continue
            hreskeys[args] = 0
            ds,_ = load_xr_dataset(args.split(),high_res=True)
            ds = Dataset(ds)
            ds = ds.drop_timeless_vars()   
            ds = ds.drop_unrelated_vars()
            if keyvaldict['depth'] > 0:
                dss = ds.separate_by_depth(bzdepths)
                for d,ds  in zip(bzdepths,dss):
                    keyvaldict['depth'] = d
                    args = self.to_args(keyvaldict,bz_correct=False)
                    self[args] = ds
            elif keyvaldict['depth'] == 0:
                args = self.to_args(keyvaldict,bz_correct=False)
                self[args] = ds
    def __iter__(self,):
        lengths = []
        for key in self.keys():
            lengths.append(len(self[key].time))
        keyvaltpls = tuple((key,val) for key,val in self.items())
        it =  IterateUntilAllFinished(keyvaltpls,lengths)
        return it.__iter__()
class RandomInds:
    def __init__(self,maxnum:int, ) -> None:
        x = np.arange(maxnum)
        np.random.shuffle(x)
        self.keys = x.tolist()
    def pop(self,):
        if bool(self.keys):
            return self.keys.pop()
        return None
    @property
    def depleted(self,):
        return not bool(self.keys)
class IterateUntilAllFinished:
    def __init__(self,objs,lengths,):
        self.objs = objs
        self.lengths = lengths
        self.rinds = [RandomInds(length) for length in lengths]
        self.mini_counter = 0
        self.num = len(objs)
    def __iter__(self,):
        return self
    def inc(self,):
        self.mini_counter += 1
        self.mini_counter = self.mini_counter%self.num
    def get_obj_ind(self,):
        randomind = self.rinds[self.mini_counter]
        obj = self.objs[self.mini_counter]
        return randomind,obj
    def __next__(self,):
        randomind,obj = self.get_obj_ind()
        tour_counter = 0
        while randomind.depleted and tour_counter <  self.num:
            self.inc()
            randomind,obj = self.get_obj_ind()
            tour_counter+= 1
        if tour_counter == self.num:
            raise StopIteration        
        self.inc()
        return obj,randomind.pop()
        
        
        
def main():
    
    
    depths = [int(d) for d in DEPTHS]
    depths.pop(2)
    print(depths)
    
    dss = Datasets(sigma = (1,4,8,12,16),depth = depths, co2 = (False,True))
    dss.load_lowres_datasets()
    dss.load_highres_datasets()
    for key,ds in dss.items():
        print(key,'\t',list(ds.data_vars.keys()))

    
    
    adh = KeyedHistograms(initial_buffer_size=1000,num_bins=512)
    filename = os.path.join(OUTPUTS_PATH,'distributions_of_all')
    total_count = 0
    save_freq = 100
    for (key,ds0),t in dss:        
        print(total_count,'\t\t\t',key,'\t\t\t',t)
        ds = ds0.isel(time = t)
        for data in ds.data_vars:
            newkey = ' '.join([key,f'--field {data}'])
            vals = ds[data].values
            adh.feed(newkey,vals)
        total_count+=1
        if total_count % save_freq == 0 :
            adh.save2npz(filename)
    adh.save2npz(filename)
    
    
    # adh = KeyedHistograms.load_from_npz('dummy')
    # for i,(key,val) in enumerate(adh.histdict.items()):
    #     if not isinstance(val,Histogram):
    #         continue
    #     hist = val.histogram
    #     edges = val.medges
    #     plt.semilogy(edges,hist/np.sum(hist))
    #     plt.title(key)
    #     plt.savefig(f'plot_{i}.png')
    #     plt.close()
    

if __name__ == '__main__':
    main()