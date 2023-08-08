import logging
from constants.paths import OUTPUTS_PATH
from data.load import get_data, load_xr_dataset
from linear.coarse_graining_inversion import CoarseGrainingInverter
from linear.lincol import CollectParts
import matplotlib.pyplot as plt
from linear.coarse_graining_operators import BaseFiltering
import numpy as np
import os
import scipy.sparse as sp
import xarray as xr
import logging
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

def forward_test():
    logging.basicConfig(level=logging.INFO,format = '%(message)s',)
    sigma = 16
    args = f'--sigma {sigma} --depth 0 --filtering gcm --co2 True'.split()
    cds,_ = load_xr_dataset(args,high_res = False)
    
    # cds.isel(time = 0).u.plot()
    # plt.savefig('coarse_u.png')
    # plt.close()
    # return
    fds,_ = load_xr_dataset(args,high_res = True)
    
    cwet_mask = cds.wet_density.fillna(0) >= 0.5
    

    # cu = cds.u.isel(time = 0).drop('depth time co2'.split()).expand_dims(dict(depth = [0]),axis = 0)
    ftemp = fds.temp.isel(time = 0).drop('depth time co2'.split()).expand_dims(dict(depth = [0]),axis = 0)
    fu = fds.u.isel(time = 0).drop('depth time co2'.split()).expand_dims(dict(depth = [0]),axis = 0)
    fu = xr.DataArray(
        data = ftemp.values,dims = fu.dims,coords = {key:fu[key].values for key in fu.dims}
    ).rename(
        dict(ulat = 'lat',ulon = 'lon')
    )
    
        
    cginv = CoarseGrainingInverter(filtering = 'gcm',depth = 0,sigma = sigma)
    
    ugrid = get_grid(sigma,0)
    

    filtering = filtering_class(sigma,ugrid)
    coarse_graining =coarse_grain_class(sigma,ugrid)
    
    logging.info(f'cu = coarse_graining(filtering(fu))...')
    # logging.info(f'wet_mask = {wet_mask}')
    # logging.info(f'fu = {fu}')
    # return
    fuv = fu.fillna(0)*ugrid.wet_mask
    # fuv = np.random.randn(*bf.inshape)*ugrid.wet_mask.values.reshape(bf.inshape)
    # fuv = fu.values.reshape(bf.inshape)*ugrid.wet_mask.values.reshape(bf.inshape)
    cu = coarse_graining(filtering(fuv  )).compute().fillna(0)
    cu = cu*cwet_mask
    
    logging.info(f'cginv.load_parts()')
    cginv.load_parts()
    # cginv.load()
    
    logging.info(f'ccu = cginv.forward_model(fu) * wet_mask')
    # ccu = cginv.mat @ fuv.values.flatten()
    cfu = cginv.mat.T @ (cginv.qinvmat @ cu.fillna(0).values.flatten())
    cfu = xr.DataArray(
        data = cfu.reshape(fu.shape), dims = fu.dims,coords= {key:fu[key].values for key in fu.dims}
    )
    # ccu = xr.DataArray(
    #     data = ccu.reshape(cu.shape), dims = cu.dims,coords= {key:cu[key].values for key in cu.dims}
    # )
    # ccu = cginv.forward_model(fu) 
    err =cfu - fuv # np.log10(np.abs(cfu - fu) + 1e-9)
    logging.info(f'fig,axs = plt.subplots(ncols = 3,figsize = (30,10))...')
    fig,axs = plt.subplots(nrows = 3,figsize = (10,30))
    
    fuv.plot(ax = axs[0])
    cfu.plot(ax = axs[1])
    err.plot(ax = axs[2])
    fig.savefig('forward_test.png')
    plt.close()
    return
    fucoords = {k:fu[k].values for k in fu.coords}
    fnu = cginv.inverse_model(ccu,fucoords)

    
def main():
    from linear.lincol import CollectParts
    txts = CollectParts.check_if_erred(jobid = '36197229') # '36197499'
    
    logging.basicConfig(level=logging.INFO,format = '%(asctime)s %(message)s',)
    logging.info(txts)
    return
    forward_test()
    return
    from linear.coarse_graining_inversion import CollectParts
    for sigma in [16,]:
        head = f'gcm-dpth-0-sgm-{sigma}' 
        root = os.path.join(OUTPUTS_PATH,'filter_weights')
        root = os.path.join(OUTPUTS_PATH,'filter_weights')
        path = CollectParts.latest_united_file(root,head)
        bf = BaseFiltering(sigma,0)
        bf.post__init__()
        grid = get_grid(sigma,0)
        
        
        
        
        x = np.random.randn(*bf.inshape)#*0
        wet_mask = grid.wet_mask.values
        
        # print(x.shape)
        x = x*wet_mask
       
        # print(wet_mask.shape)
        # return
        
        matform = sp.load_npz(path)
        cx_ = (matform @ x.flatten()).reshape(bf.outshape)
    
        class_args = (sigma,grid)
        filtering = filtering_class(*class_args)
        coarse_graining = coarse_grain_class(*class_args)
        cx = coarse_graining(filtering(x)).values.reshape(bf.outshape)
        relerr = np.abs(cx_ - cx)
        
        fig,axs = plt.subplots(nrows = 3,figsize = (10,45))
        titles = 'mat cf err'.split()
        for ax,val,title in zip(axs,[cx_,cx,relerr],titles):
            vmax = np.amax(np.abs(val[val==val]))
            neg = ax.imshow(val,cmap = 'bwr',vmin = -vmax,vmax = vmax)
            fig.colorbar(neg,ax = ax)
            ax.set_title(title)
        fig.savefig(f'dummy-{sigma}.png')
        plt.close()
        logging.info(f'dummy-{sigma}.png')
        
    

if __name__ == '__main__':
    main()