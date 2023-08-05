import logging
from data.load import get_data, load_xr_dataset
from linear.coarse_graining_inversion import CoarseGrainingInverter
import matplotlib.pyplot as plt
import numpy as np
from transforms.coarse_graining import GcmFiltering, GreedyCoarseGrain,PlainCoarseGrain,ScipyFiltering
filtering_class = ScipyFiltering#GcmFiltering#ScipyFiltering#
coarse_grain_class = PlainCoarseGrain#GreedyCoarseGrain# PlainCoarseGrain#
def get_grid(sigma:int,depth:int,):
    args = f'--sigma {sigma} --depth {depth} --mode data --filtering gcm'.split()
    x, = get_data(args,torch_flag=False,data_loaders=False,groups = ('train',))
    x0 = x.per_depth[0]
    ugrid = x0.ugrid
    ugrid = ugrid.drop('time co2'.split())
    return ugrid

def main():
    logging.basicConfig(level=logging.INFO,format = '%(message)s',)
    sigma = 16
    args = f'--sigma {sigma} --depth 0 --filtering gcm'.split()
    cds,_ = load_xr_dataset(args,high_res = False)
    
    import xarray as xr
    cds1 = xr.open_zarr('/scratch/cg3306/climate/outputs/data/coarse_16_surface_gcm_0_20.zarr')
    cds2 = cds1#xr.open_zarr('/scratch/cg3306/climate/outputs/data/coarse_16_surface_gcm_0_20_non_greedy.zarr')
    
    cds1 = cds1.isel(depth = 0)
    cds2 = cds2.isel(depth = 0)
    
    
    cds = cds.isel(time = 0)#,depth = 5)
    cds1 = cds1.isel(time = 0)
    cds2 = cds2.isel(time = 0)

    cu,cu_,cu__ = tuple(fd.temp.fillna(0) for fd in (cds,cds1,cds2))
    err = np.abs(cu - cu_)
    err_ = np.abs(cu - cu__)
    err__ = np.abs(cu_ - cu__)
    fig,axs = plt.subplots(ncols = 6,figsize = (60,10))
    cu.plot(ax = axs[0])
    cu_.plot(ax = axs[1])
    cu__.plot(ax = axs[2])
    kwargs = dict(
        vmin = -6,vmax = 1
    )
    np.log10(err).plot(ax = axs[3],**kwargs)
    np.log10(err_).plot(ax = axs[4],**kwargs)
    np.log10(err__).plot(ax = axs[5],**kwargs)
    plt.savefig('comparison.png')
    plt.close()
    
    return
    
    fds,_ = load_xr_dataset(args,high_res = True)

    cu = cds.u.isel(time = 0).drop('depth time co2'.split()).expand_dims(dict(depth = [0]),axis = 0)
    fu = fds.u.isel(time = 0).drop('depth time co2'.split()).expand_dims(dict(depth = [0]),axis = 0)
    fu = fu.rename({'ulat':'lat','ulon':'lon'})
    ugrid = get_grid(sigma,0)

    iseldict = dict(lat = slice(1000,1300),lon = slice(1000,1300))
    ciseldict = {key:slice(slc.start//sigma,slc.stop//sigma) for key,slc in iseldict.items()}

    fu = fu.isel(**iseldict).fillna(0)
    ugrid = ugrid.isel(**iseldict)
    cu = cu.isel(**ciseldict).fillna(0)
    
    
    filter_ = filtering_class(sigma,ugrid)
    coarsegrain = coarse_grain_class(sigma,ugrid)
    
    
    ffu = filter_(fu)
    fig,axs = plt.subplots(ncols = 1,figsize = (15,10))
    ffu.plot(ax = axs)
    plt.savefig('filtered.png')
    plt.close()
    
    cu_ = coarsegrain(ffu)
    err = np.abs(cu - cu_)
    
    fig,axs = plt.subplots(ncols = 3,figsize = (45,10))
    cu.plot(ax = axs[0])
    cu_.plot(ax = axs[1])
    np.log10(err).plot(ax = axs[2])
    plt.savefig('comparison.png')
    plt.close()
    return
    cginv = CoarseGrainingInverter(filtering = 'gcm',depth = 0,sigma = sigma)
    cginv.load()
    # logging.info(f'cginv.mat.shape= {cginv.mat.shape}')
    # logging.info(f'fu.shape= {fu.shape}')
    ccu = cginv.forward_model(fu)
    ccu = ccu.rename({
        'ulat':'lat','ulon':'lon'
    })
    
    fig,axs = plt.subplots(ncols = 3,figsize = (30,10))
    err = np.abs(ccu - cu)
    cu.fillna(0).plot(ax = axs[0])
    ccu.plot(ax = axs[1])
    err.plot(ax = axs[2])
    fig.savefig('coarse_graining_example.png')
    plt.close()
    

if __name__ == '__main__':
    main()