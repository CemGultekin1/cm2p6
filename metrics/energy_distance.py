from dataclasses import dataclass
import itertools
import os
from models.nets.cnn import kernels2spread
from plots.metrics_ import metrics_dataset
from constants.paths import JOBS, EVALS, all_eval_path
from constants.paths import DISTS
from metrics.modmet import ModelMetric, ModelResultsCollection
from utils.slurm import flushed_print
from utils.xarray import plot_ds, skipna_mean
import xarray as xr
from utils.arguments import args2dict, options
import numpy as np
import matplotlib.pyplot as plt 
def merge_and_save(stats):
    xr.merge(list(stats.values())).to_netcdf(all_eval_path(),mode = 'w')

def energy_test_for_gaussian(density0:np.ndarray,density1:np.ndarray,dx:float):
    ecdf0 = np.cumsum(density0,axis = density0.ndim - 1)
    ecdf1 = np.cumsum(density1,axis = density1.ndim - 1)
    def stack2ndim(d0:np.ndarray,d1:np.ndarray):
        if d0.ndim == d1.ndim:
            return d0,d1
        elif d0.ndim < d1.ndim:
            d1,d0 = stack2ndim(d1,d0)
            return d0,d1
        elif d0.ndim > d1.ndim:
            x = d0.ndim -  d1.ndim
            newshp = [1]*x + list(d1.shape)
            d1 = d1.reshape(newshp)
            return d0,d1
    _,density1 = stack2ndim(ecdf0,density1)
    ecdf0,ecdf1 = stack2ndim(ecdf0,ecdf1)
    # kl = ecdf0*np.log(ecdf0/ecdf1)
    # kl = np.where(ecdf0 == 0,0,kl)
    ks = np.sum((ecdf0-ecdf1)**2*density1,axis = ecdf0.ndim - 1)
    return (ecdf0,ecdf1),ks

@dataclass
class EnergyDistance:
    densities:xr.Dataset = None
    modelid:str = ''
    modelid_str:str = ''
    density_str:str = ''
    ncoord_str:str = ''
    energy_test_str:str = ''
    def plot(self,title):
        for key in self.densities.data_vars.keys():
            coord = key.replace(self.density_str,self.ncoord_str)
            coords = self.densities[coord].values
            kvar = self.densities[key]
            assert kvar.dims[-1] == coord
            shp = kvar.shape
            kvals = kvar.values
            fkvals = kvals.reshape([-1,shp[-1]])
            
            gaussdist = np.exp( - coords**2/2)
            gaussdist = gaussdist/np.sum(gaussdist)
            coords = self.densities[coord].values
            for i in range(fkvals.shape[0]):
                plt.plot(coords,fkvals[i],label = 'empirical')
                plt.plot(coords,gaussdist,label = 'theoratical')
                plt.grid( which='major', color='k', linestyle='--',alpha = 0.5)
                plt.grid( which='minor', color='k', linestyle='--',alpha = 0.5)
                plt.title(title)
                plt.savefig(f'distribution-{title}-{i}.png')
                plt.close()
    def compute(self,):
        data_vars = {}
        coords_dict = {}
        for key in self.densities.data_vars.keys():
            coord = key.replace(self.density_str,self.ncoord_str)

            kvar = self.densities[key]
            assert kvar.dims[-1] == coord
            shp = kvar.shape
            kvals = kvar.values
            fkvals = kvals.reshape([-1,shp[-1]])
            
            
            coords = self.densities[coord].values
            
            # dfkvals = np.abs(fkvals[:,1:] - fkvals[:,:-1])
            # mfkvals = (fkvals[:,1:] + fkvals[:,:-1])/2
            # maxfkv = np.amax(dfkvals,axis = 1,keepdims = True)
            # mfkvals[dfkvals > maxfkv/2] = np.nan

            
            gaussdist = np.exp( - coords**2/2)
            gaussdist = gaussdist/np.sum(gaussdist)
            
            dx = coords[1] -coords[0]
            # loglikelihood = loglikelihood_test(fkvals,coords)
            _,ks = energy_test_for_gaussian(fkvals,gaussdist,dx)
            ks = ks.reshape(shp[:-1])
            dims =  list(kvar.dims[:-1])
            data_vars[key.replace(self.density_str,self.energy_test_str)] = (dims,ks)
            for dim in dims:
                coords_dict[dim] = self.densities[dim].values
        return xr.Dataset(data_vars,coords_dict)
            
            # for i in range(fkvals.shape[0]):
            #     plt.plot(coords,fkvals[i],label = 'empirical')
            #     plt.plot(coords,gaussdist,label = 'theoratical')
            #     plt.grid( which='major', color='k', linestyle='--',alpha = 0.5)
            #     plt.grid( which='minor', color='k', linestyle='--',alpha = 0.5)
            #     plt.title(f'ks = {ks[i]}')
            #     plt.savefig(f'distribution-{i}.png')
            #     plt.close()
            
            
def kernel_size_fun(kernels):    
    ks = kernels2spread(kernels)*2 + 1
    return ks
def main():
    root = DISTS
    models = os.path.join(JOBS,'offline_sweep.txt')
    file1 = open(models, 'r')
    lines = file1.readlines()
    file1.close()

    
    transform_funs = dict(
        kernel_size = dict(
            inputs = ['kernels'],
            fun = kernel_size_fun
        )
    )
    coords = ['sigma','temperature','domain','latitude','depth',\
        'seed','kernel_size','filtering']
    rename = dict(depth = 'training_depth')
    data = {}
    coord = {}
    base_kwargs = dict(
        modelid_str = 'modelid',
        density_str = '_density',
        ncoord_str = '_normalized',
        energy_test_str = '_test'
    )
    mrc = ModelResultsCollection()
    for i,line in enumerate(lines):
        _,modelid = options(line.split(),key = 'model')
        snfile = os.path.join(root,modelid + '.nc')
        if not os.path.exists(snfile):
            continue
        try:
            sn = xr.open_dataset(snfile)
        except:
            continue
        # print(sn)
        kst = EnergyDistance(densities=sn,modelid=modelid,**base_kwargs)
        # kst.plot(str(i))
        # continue
        metrics = kst.compute()
        mm = ModelMetric(line.split(),metrics)
        mrc.add_metrics(mm)
        # if i == 2:
        #     break
    # return
    ds = mrc.merged_dataset()
    ds1 = xr.where( np.isnan(ds), ds,1)
    ds = ds.sum(dim = ['minibatch','stencil'],skipna = True)    
    ds = ds/ds1.sum(dim = ['minibatch','stencil'],skipna = True)
    # print(ds.isel(domain = 1,temperature=1,training_depth = 0,sigma = 0,co2 = 0,depth = 0))
    # print(ds1.isel(domain = 1,temperature=1,training_depth = 0,sigma = 0,co2 = 0,depth = 0))
    filename = os.path.join(DISTS,'all.nc')
    ds.to_netcdf(filename,mode = 'w')
if __name__=='__main__':
    main()