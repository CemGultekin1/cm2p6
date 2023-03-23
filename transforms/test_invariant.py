

from data.load import load_filter_weights,load_xr_dataset
from utils.arguments import options
from transforms.grids import get_grid_vars
from transforms.gcm_inversion import GcmInversion
from utils.xarray import plot_ds
import itertools
import numpy as np
import matplotlib.pyplot as plt
def main():
    sigma = 4
    foldername = 'tobedeleted_'
    import os
    if not os.path.exists(foldername):
        os.makedirs(foldername)
        
    args = f'--sigma {sigma} --filtering gcm --lsrp 0 --mode data'.split()
    fw = load_filter_weights(args,utgrid='u',svd0213 = False)
    m = 4
    latd,lond = [np.arange(m)]*2    
    nrows = m**2
    fig,axs= plt.subplots(nrows,3,figsize = (3*10,nrows*10))
    for i,(lat,lon) in enumerate(itertools.product(latd,lond)):
        fww = fw.weight_map.isel(depth = 0,lat_degree = lat,lon_degree = lon)
        fwlat = fw.latitude_filters.isel(depth = 0,lat_degree = lat)
        fwlon = fw.longitude_filters.isel(depth = 0,lon_degree = lon)        
        ax= axs[i,0]
        log10fww = np.log10(np.abs(fww))
        log10fww.plot(ax = ax,cmap = 'seismic')
        ax = axs[i,1]
        fwlat.plot(ax = ax)
        ax = axs[i,2]
        fwlon.plot(ax = ax)
    fig.savefig('filter_weight_maps.png')
def test():
    sigma = 4
    foldername = 'tobedeleted_'
    import os
    if not os.path.exists(foldername):
        os.makedirs(foldername)
        
    args = f'--sigma {sigma} --filtering gcm --lsrp 1 --mode data'.split()
    fw = load_filter_weights(args,utgrid='u',svd0213 = False)
    datargs,_ = options(args,key = 'data')
    ds,_ = load_xr_dataset(args)
    cds,_ = load_xr_dataset(args,high_res=False)
    varname = 'temp'
    gridtype = 'u' if varname in 'u v'.split() else 't'
    hres = ds[varname].isel(time = 0).rename(
        {f'{gridtype}{dim}':dim for dim in 'lat lon'.split()}
    )
    cres = cds[varname].isel(time = 0,depth= 0)
    
    ugrid,tgrid = get_grid_vars(ds.isel(time = 0))
    if gridtype == 'u':
        grid = ugrid
    else:
        grid = tgrid
    
    rank = 1
    gcminv = GcmInversion(datargs.sigma,grid.isel(depth = 0),fw.isel(depth = 0),rank = rank)
    hres_opt = gcminv.fit(cres,maxiter = 8,sufficient_decay_limit=0.95)
    coords = (
        (-60,0,-40,20),
        (0,60,-40,20),
        (-60,0,-150,-90),
        (0,60,-150,-90),
        (-60,0,-90,-30),
        (0,60,-90,-30),
    )
    sel_dicts = [
        dict(lat = slice(c[0],c[1]),lon = slice(c[2],c[3])) for c in coords
    ]
    udict = dict(utrue = hres,usolv = hres_opt, uerr = np.log10(np.abs(hres - hres_opt)))
    plot_ds(udict,foldername+f'/gcm_invariant_inversion_{varname}_{rank}.png',ncols = 3)
    for i,sel_dict in enumerate(sel_dicts):
        udict_ = {key:val.sel(**sel_dict) for key,val in udict.items()}
        plot_ds(udict_,foldername+f'/gcm_invariant_inversion_{varname}_{i}_{rank}.png',ncols = 3)
if __name__ == '__main__':
    test()