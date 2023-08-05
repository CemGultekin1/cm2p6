import logging
from data.load import get_data, load_xr_dataset
from linear.coarse_graining_inversion import CoarseGrainingInverter
import matplotlib.pyplot as plt
import numpy as np
from transforms.coarse_graining import GcmFiltering, GreedyCoarseGrain,PlainCoarseGrain,ScipyFiltering
filtering_class = GcmFiltering#ScipyFiltering#
coarse_grain_class = GreedyCoarseGrain# PlainCoarseGrain#
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
    fds,_ = load_xr_dataset(args,high_res = True)
    
    wet_mask = cds.wet_density >= 0.5


    # cu = cds.u.isel(time = 0).drop('depth time co2'.split()).expand_dims(dict(depth = [0]),axis = 0)
    fu = fds.u.isel(time = 0).drop('depth time co2'.split()).expand_dims(dict(depth = [0]),axis = 0)
    fu = fu.rename({
        'ulat':'lat','ulon':'lon'
    })
    
    
    ugrid = get_grid(sigma,0)
    
    filtering = filtering_class(sigma,ugrid)
    coarse_graining =coarse_grain_class(sigma,ugrid)
    
    logging.info(f'cu = coarse_graining(filtering(fu))...')
    cu = coarse_graining(filtering(fu)) * wet_mask
    
    cginv = CoarseGrainingInverter(filtering = 'gcm',depth = 0,sigma = sigma)
    logging.info(f'cginv.load_parts()')
    cginv.load_parts()
    logging.info(f'ccu = cginv.forward_model(fu) * wet_mask')
    ccu = cginv.forward_model(fu) * wet_mask
    
    # fcoords = {k:fu[k].values for k in fu.coords}
    # logging.info(f'fcu = cginv.inverse_model(ccu,fcoords) ')    
    # fcu = cginv.inverse_model(ccu,fcoords) 
    err = np.log10(np.abs(ccu - cu) + 1e-9)
    logging.info(f'fig,axs = plt.subplots(ncols = 3,figsize = (30,10))...')
    fig,axs = plt.subplots(nrows = 3,figsize = (10,30))
    cu.plot(ax = axs[0])
    ccu.plot(ax = axs[1])
    err.plot(ax = axs[2])
    fig.savefig('invmat_recovery.png')
    plt.close()
    
    return
    
    
    
    logging.info(f'fig,axs = plt.subplots(ncols = 3,figsize = (30,10))...')
    fig,axs = plt.subplots(ncols = 3,figsize = (30,10))
    err = np.log10(np.abs(ccu - cu) + 1e-9)
    cu.fillna(0).plot(ax = axs[0])
    ccu.plot(ax = axs[1])
    err.plot(ax = axs[2])
    fig.savefig('coarse_graining_example.png')
    plt.close()
    

if __name__ == '__main__':
    main()