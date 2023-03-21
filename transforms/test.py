
from data.load import load_filter_weights, load_xr_dataset
from transforms.gcm_compression_spatially_variant import Variant2DMatmult,GcmInversion
import numpy as np
from utils.xarray import plot_ds
import xarray as xr 
from transforms.grids import get_grid_vars
from utils.arguments import options
import matplotlib.pyplot as plt

def old_main():
    sigma = 4
    args = f'--sigma {sigma} --filtering gcm --lsrp 1 --mode data'.split()
    dfw = load_filter_weights(args,utgrid='t',svd0213 = True).load()
    # nlon,span,sng = dfw.longitude_filters.shape
    # lonf = dfw.longitude_filters.values
    # lonf = lonf.reshape([sng,nlon,span]).transpose([1,2,0])
    # dfw['longitude_filters'].data = lonf
    
    coords = dict(lat = (0,3),lon = (-140,-137))
    crs_ic = {
        dim: [np.argmin(np.abs(dfw[dim].values - lims[j])) for j in range(2)] for dim,lims in coords.items()
    }
    datargs,_ = options(args,key = 'data')
    ds,_ = load_xr_dataset(args,high_res = True)
    fine_isel = {
        dim: slice(lims[0]*sigma,lims[1]*sigma) for dim,lims in crs_ic.items()
    }
    cors_isel = {
        dim: slice(lims[0],lims[1]) for dim,lims in crs_ic.items()
    }
    
    cors_isel = {}
    fine_isel = {}
    
    
    ugrid,_ = get_grid_vars(ds.isel(time = 0))
    
    
    ugrid = ugrid.isel(**fine_isel)
    dfw = dfw.isel(**cors_isel)

    rank = 32
    v2dm = Variant2DMatmult(sigma,ugrid,dfw,rank = rank)

    u = ds.temp.isel(time = 0).rename(
        {'t'+dim : dim for dim in coords}
    )
    u = u.isel(**fine_isel)
    ubar = v2dm(u,inverse = False,separated= False)
    plot_ds(dict(ubar = ubar,),f'tempbar_.png',ncols = 1)
    return
    for i in range(rank):
        ubar = v2dm(u,inverse = False,separated= False,special = i)
        plot_ds(dict(ubar = ubar,),f'ubar_{i}.png',ncols = 1)
        
def isel(coarse_indexes,sigma):
    if len(coarse_indexes)== 0:
        return {},{}
    fbounds = {ci:((l0-2)*sigma,(l1+3)*sigma) for ci,(l0,l1)in coarse_indexes.items()}
    fslices = {ci:slice(l0,l1)  for ci,(l0,l1) in fbounds.items()}
    cslices = {ci:slice(l0//sigma,l1//sigma) for ci,(l0,l1) in fbounds.items()}
    return fslices,cslices
def zero_fill(cres,pad,sigma):
    cres.data[:pad,:] = 0
    cres.data[-pad:,:] = 0
    cres.data[:,:pad] = 0
    cres.data[:,-pad:] = 0
    fslice = dict(
        lat = slice(pad*3*sigma,(cres.shape[1] - 3*pad)*sigma),
        lon = slice(pad*3*sigma,(cres.shape[1] - 3*pad)*sigma),
    )
    return cres,fslice
def main():
    sigma = 4
    foldername = 'tobedeleted_'
    import os
    if not os.path.exists(foldername):
        os.makedirs(foldername)
        
    args = f'--sigma {sigma} --filtering gcm --lsrp 1 --mode data'.split()
    fw = load_filter_weights(args,utgrid='u',svd0213 = True)
    datargs,_ = options(args,key = 'data')
    ds,_ = load_xr_dataset(args)
    cds,_ = load_xr_dataset(args,high_res=False)
    varname = 'u'
    gridtype = 'u' if varname in 'u v'.split() else 't'
    hres = ds[varname].isel(time = 0).rename(
        {f'{gridtype}{dim}':dim for dim in 'lat lon'.split()}
    )
    cres = cds[varname].isel(time = 0,depth= 0)
    
    ugrid,_ = get_grid_vars(ds.isel(time = 0))
    
    # fslc,cslc = isel(dict(lat = (130,360),lon = (130,360)),sigma)
    # ugrid = ugrid.isel(**fslc).load()
    # hres = hres.isel(**fslc).load()
    # cres = cres.isel(**cslc).load()
    # fw = fw.isel(**cslc).load()
    # zero_pad = 20
    # cres,fslice = zero_fill(cres,zero_pad,sigma)
    
    ranks = [1,2,4,8,12,16,24,32,48]
    collect_rank_iter = {rank:([],[]) for rank in ranks}
    for ir,rank in enumerate(ranks):
        gcminv = GcmInversion(datargs.sigma,ugrid,fw,rank = rank)
        
        for iternum,(hres_opt,cres_opt,cres_err) in enumerate(gcminv.fit(cres,maxiter = 16,sufficient_decay_limit=0.95)):
            hres_log_err = np.log10(np.abs(hres - hres_opt))
            # cres_log_err = np.log10(np.abs(cres - cres_opt))
            hresdict = dict(true = hres,pred =hres_opt,err = hres_log_err)
            # hresdict = {key:hres.isel(**fslice) for key,hres in hresdict.items()}
            hres_err = np.square(hresdict['true'] - hresdict['pred']).sum().values.item()
            print(hres_err,cres_err)
            collect_rank_iter[rank][0].append(cres_err)
            collect_rank_iter[rank][1].append(hres_err)
            
        fig,axs = plt.subplots(1,2,figsize = (14,7))
        for rank_ in ranks[:ir+1]:
            for i in range(2):
                axs[i].plot(collect_rank_iter[rank_][i],label = f'rank = {rank_}')
        for i in range(2):
            ax = axs[i]
            ax.legend()
            ax.grid( which='major', color='k', linestyle='--',alpha = 0.5)
            ax.grid( which='minor', color='k', linestyle='--',alpha = 0.5)
        fig.savefig(foldername + f'/rank_iter.png')
            # plot_ds(hresdict,foldername + f'/hres_{iternum}_{rank}.png',ncols = 3)
            # plot_ds(dict(cres = cres,cres_opt =cres_opt,cres_err = cres_log_err),foldername + f'/cres_{iternum}_{rank}.png',ncols = 3)

    return
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
    udict = dict(utrue = utrue,usolv = uopt, uerr = np.log10(np.abs(utrue - uopt)))
    plot_ds(udict,f'gcm_inversion_{rank}.png',ncols = 3)
    for i,sel_dict in enumerate(sel_dicts):
        udict_ = {key:val.sel(**sel_dict) for key,val in udict.items()}
        plot_ds(udict_,f'gcm_inversion_{i}_{rank}.png',ncols = 3)
if __name__ == '__main__':
    main()