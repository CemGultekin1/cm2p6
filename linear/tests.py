import itertools
import logging
from constants.paths import OUTPUTS_PATH
from data.coords import DEPTHS
from data.load import get_data, load_xr_dataset
from linear.coarse_graining_inversion import CoarseGrainingInverter
from linear.lincol import CollectParts
import matplotlib.pyplot as plt
from linear.coarse_graining_operators import BaseFiltering,get_grid,filtering_class,coarse_grain_class
import numpy as np
import os
import scipy.sparse as sp
import xarray as xr
import logging
from transforms.coarse_graining import GcmFiltering, GreedyCoarseGrain


def forward_test(sigma,depth):
    logging.basicConfig(level=logging.INFO,format = '%(message)s',)
    
    args = f'--sigma {sigma} --depth {depth} --filtering gcm --co2 True'.split()
    cds,_ = load_xr_dataset(args,high_res = False)
    
    # cds.isel(time = 0).u.plot()
    # plt.savefig('coarse_u.png')
    # plt.close()
    # return
    fds,_ = load_xr_dataset(args,high_res = True)
    cwet_mask = cds.wet_density.fillna(0) >= 0.5
    

    # cu = cds.u.isel(time = 0).drop('depth time co2'.split()).expand_dims(dict(depth = [0]),axis = 0)
    isel = dict(time = 0)
    sel = dict(depth = depth,method = 'nearest')
    ffs = [fds.u,fds.temp]
    for i,ff in enumerate(ffs):
        ff = ff.isel(**isel).drop('time co2'.split())
        if 'depth' in ff.coords:
            if ff.values.ndim > 2:
                ff = ff.sel(**sel)             
            ff = ff.drop('depth')
        ffs[i] = ff.expand_dims(dict(depth = [depth]),axis = 0)
    fu,ftemp = ffs
    fu = xr.DataArray(
        data = ftemp.values,dims = fu.dims,coords = {key:fu[key].values for key in fu.dims}
    ).rename(
        dict(ulat = 'lat',ulon = 'lon')
    )
    
    # print(fu)
    cginv = CoarseGrainingInverter(filtering = 'gcm',depth = depth,sigma = sigma)
    
    ugrid = get_grid(sigma,depth)
    ugrid['depth'] = fu.depth.values
    # print(f'ugrid.depth = {ugrid.depth.values}')
    # print(f'fu.depth = {fu.depth.values}')
    
    filtering = filtering_class(sigma,ugrid)
    coarse_graining =coarse_grain_class(sigma,ugrid)
    
    # logging.info(f'wet_mask = {wet_mask}')
    # logging.info(f'fu = {fu}')
    # return
    fuv = fu.fillna(0)*ugrid.wet_mask
    # fuv = np.random.randn(*bf.inshape)*ugrid.wet_mask.values.reshape(bf.inshape)
    # fuv = fu.values.reshape(bf.inshape)*ugrid.wet_mask.values.reshape(bf.inshape)

    cu = coarse_graining(filtering(fuv)).compute().fillna(0)
    cu = cu*cwet_mask
    
    # cginv.load_parts()
    cginv.load()
    
    # ccu = cginv.mat @ fuv.values.flatten()
    ccu = cginv.forward_model(fuv)
    ccu = xr.DataArray(
        data = ccu.values.reshape(cu.shape), dims = cu.dims,coords= {key:cu[key].values for key in cu.dims}
    )*cwet_mask
    err = cu - ccu
    
    
    # ccu = cginv.forward_model(fu) 
    # cfu = cginv.mat.T @ (cginv.qinvmat @ cu.fillna(0).values.flatten())
    # cfu = xr.DataArray(
    #     data = cfu.reshape(fu.shape), dims = fu.dims,coords= {key:fu[key].values for key in fu.dims}
    # )
    # err =cfu - fuv 
    logging.info(f'fig,axs = plt.subplots(ncols = 3,figsize = (30,10))...')
    fig,axs = plt.subplots(nrows = 3,figsize = (10,30))
    
    cu.plot(ax = axs[0])
    ccu.plot(ax = axs[1])
    err.plot(ax = axs[2])
    fig.savefig(f'forward_test_{sigma}_{depth}.png')
    logging.info(f'forward_test_{sigma}_{depth}.png')
    plt.close()
    return
    fucoords = {k:fu[k].values for k in fu.coords}
    fnu = cginv.inverse_model(ccu,fucoords)

def backward_test():
    pass
def main():
    from linear.lincol import CollectParts
    # txts = CollectParts.check_if_erred(jobid = '36197229') # '36197499'
    
    logging.basicConfig(level=logging.INFO,format = '%(asctime)s %(message)s',)
    sigmas= [4,8,12,16]
    
    depths= list(map(int,DEPTHS))
    sds = list(itertools.product(sigmas,depths))
    
    for sigma,depth in sds:
        try:
            forward_test(sigma,depth)
        except:
            continue


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