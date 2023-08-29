import itertools
import logging
from constants.paths import TEMPORARY_DATA
from data.coords import TIMES
from data.load import load_xr_dataset
from linear.coarse_graining_inverter import CoarseGrainingInverter
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.sparse as sp
from transforms.grids import get_grid_vars
from utils.xarray_oper import concat
import xarray as xr
import logging
import sys
from transforms.subgrid_forcing import GcmSubgridForcing



def collect_grid(fds,depth):
    fd = fds.isel(time = 0)
    # if depth == 0:
    fd = fd.drop('depth')
    fd = fd.expand_dims({"depth" : [depth]},axis = 0)
    keepkeys = 'depth ulon ulat tlon tlat'.split()
    dropkeys = [key for key in fd.coords.keys() if key not in keepkeys]
    fd = fd.drop(dropkeys)
    ugrid,tgrid = get_grid_vars(fd)
    return ugrid,tgrid

def plot():
    ciseldict = dict(
        # lat = slice(100,300),lon = slice(100,300)
    )
    fiseldict = dict(
        # lat = slice(1000,1300),lon = slice(1000,1300)
    )
    hres = xr.open_dataset('hres.nc').isel(**fiseldict)
    hres0 = xr.open_dataset('hres0.nc').isel(**fiseldict)
    forcings = xr.open_dataset('forcings.nc').isel(**ciseldict)
    forcings0 = xr.open_dataset('forcings0.nc').isel(**ciseldict)
    err = np.square(forcings - forcings0).sum()
    sc2 = np.square(forcings).sum()
    rsq = 1 - err/sc2
    logging.info(f'rsquare = {rsq}')
    def err_plot(x,y,filename):
        varnames = list(x.data_vars.keys())
        nrows = 3
        ncols = len(varnames)
        fig,axs = plt.subplots(nrows, ncols, figsize = (5*ncols,5*nrows))        
        for (i,ds),j in itertools.product(list(enumerate([x,y,x-y])), range(ncols)):
            vn = varnames[j] 
            np.log10(np.abs(ds[vn])).plot(ax = axs[i,j])#,vmin = -12,vmax = 0)
            axs[i,j].set_title(vn)
        fig.savefig(f'{filename}.png')
        logging.info(f'{filename}.png')
        plt.close()
    err_plot(hres,hres0,'hres_compare')
    err_plot(forcings,forcings0,'forcings_compare') 

class  LinearForcingCompute:
    def __init__(self,sigma,depth,co2) -> None:
        args = f'--sigma {sigma} --depth {depth} --filtering gcm --co2 {co2}'.split()
        fds,_ = load_xr_dataset(args,high_res = True)
        if depth >0:
            fds = fds.sel(depth =depth, method = 'nearest')
        ugrid,_ = collect_grid(fds,depth)
        gsf = GcmSubgridForcing(sigma,ugrid)
        cginv = CoarseGrainingInverter(filtering = 'gcm',depth = depth,sigma = sigma)
        cginv.load_parts() 
        
        self.gcmsub = gsf
        self.cginv = cginv
        self.ds = fds
        
    def __len__(self,):
        return len(self.ds.time)
    def __getitem__(self,i):    
        # return np.abs(u).mean()
        ds = self.ds.isel(time = i,)
        
        u,v,temp = ds.u.load(),ds.v.load(),ds.temp.load()
        u = u.rename(ulat = "lat",ulon = "lon")
        v = v.rename(ulat = "lat",ulon = "lon")
        temp = temp.rename(tlat = "lat",tlon = "lon")
        
        
        temp['lat'] = u['lat'].values
        temp['lon'] = u['lon'].values
        mask = xr.where(np.isnan(u) + np.isnan(temp),0,1)
        u,v,temp = [kg.fillna(0)*mask for kg in [u,v,temp]]
        
        hres = dict(
            u = u, v = v, temp = temp
        )

        hres0 = {}
        for key,val in hres.items():
            val1 = self.cginv.project(val)        
            hres0[key] = val1
        
        forcings , (cres,_) = self.gcmsub(hres,'u v temp'.split(), 'Su Sv Stemp'.split())
        forcings0,  _       = self.gcmsub(hres0,'u v temp'.split(), 'Su Sv Stemp'.split())
        hres0 = {key + '0':val for key,val in hres0.items()}
        
        hds = concat(**hres0,**hres)
        
        forcings0 = {key + '_linear' : val for key,val in forcings0.items()}
        ds = concat(**forcings,**forcings0,**cres)
        tmv =  self.ds.time.values
        ds = ds.expand_dims({"time":[tmv[i]]},axis = 0)
        return ds,hds
        # .to_zarr('linear_recovery.zarr')
        
def plot(path):
    ds = xr.open_zarr(path)
    logging.info(ds)
    ds = ds.isel(time = 0,depth = 0)
    vns = list(ds.data_vars.keys())
    fig,axs = plt.subplots(len(vns),1,figsize = (5,5*len(vns)))
    for i,key in enumerate(vns):
        ds[key].plot(ax = axs[i])
    fig.savefig('ds.png')
    plt.close()
    


def main():    
    
    logging.basicConfig(level=logging.INFO,format = '%(asctime)s %(message)s',)
   
    for sigma  in [4,8,12,16]:
        lfc = LinearForcingCompute(sigma,0,'False')     
        ds,hds = lfc[0]        
        logging.info(f'lres_sample_sgm_{sigma}.nc')
        ds.to_netcdf(f'lres_sample_sgm_{sigma}.nc')
        hds.to_netcdf(f'hres_sample_sgm_{sigma}.nc')
    
            
    
if __name__ == '__main__':
    main()