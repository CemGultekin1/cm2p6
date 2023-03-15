import itertools
from data.load import load_xr_dataset
from data.paths import get_low_res_data_location
import numpy as np
def get_dataset(filtering):
    args = f'--filtering {filtering} --sigma 4'.split()
    ds_zarr,_ = load_xr_dataset(args)
    data_address = get_low_res_data_location(args)
    return ds_zarr,data_address
def check_existing_datasets():
    dsgauss,address1 = get_dataset('gaussian')
    dsgcm,address2  = get_dataset('gcm')
    dsgauss,dsgcm = [x.isel(time = 0,lat = range(100,200),lon = range(100,200)) for x in [dsgauss,dsgcm ]]
    print(address1,address2)
    import matplotlib.pyplot as plt
    fig,axs = plt.subplots(3,1)
    dsgauss.u.plot(ax = axs[0])
    dsgcm.u.plot(ax = axs[1])
    dvar = np.log10(np.abs(dsgcm.u - dsgauss.u))
    dvar.plot(ax = axs[2])
    plt.savefig('different_datasets.png')
def check_forcings():
    # root = '/scratch/zanna/data/cm2.6/coarse_datasets/'
    root = '/scratch/cg3306/climate/CM2P6Param/saves/data/'
    import xarray as xr
    select = dict(time = 0,depth =0)#,lat = range(100,500),lon = range(100,500))
    files = [
        '/scratch/zanna/data/cm2.6/coarse_datasets/coarse_4_beneath_surface_gaussian.zarr',
        root + f'coarse_4_surface_gaussian_0_1.zarr'
    ]
    dss = {i:xr.open_zarr(f).isel(**select) for i,f in enumerate(files)}
    # dss = {i:xr.open_zarr(root + f'coarse_4_surface_gcm_0_{i}.zarr').isel(**select) for i in [1,10]}
    fields = [f'S{x}{y}' for x in 'u v temp'.split() for y in ['','_res']] + 'u v temp'.split()
    k0,k1 = list(dss.keys())
    import matplotlib.pyplot as plt
    for f in fields:
        fig,axs = plt.subplots(2,3,figsize = (27,9))
        for ir in range(2):
            
            
            v0 = dss[k0][f]
            v1 = dss[k1][f]
            adv = np.abs(v1)/np.abs(v0)
            fs = [v0,v1,adv]

            if ir==0:
                fs = [xr.where(f == 0,np.nan,f) for f in fs]
                fs = [np.log10(np.abs(f)) for f in fs]

            v0,v1,adv = fs
                
            vmax = np.amax([np.amax((x if ir==0 else np.abs(x)).fillna(-np.inf).values) for x in [v0,v1]])
            vmin = np.amin([np.amin((x if ir==0 else -np.abs(x)).fillna(np.inf).values) for x in [v0,v1]])
            if ir == 1:
                v0.plot(ax = axs[ir,0],cmap = 'RdBu_r')#,vmax = vmax,vmin=vmin)
                v1.plot(ax = axs[ir,1],cmap = 'RdBu_r')#,vmax = vmax,vmin=vmin)
            else:
                v0.plot(ax = axs[ir,0],cmap = 'RdBu_r',vmax = vmax,vmin=vmin)
                v1.plot(ax = axs[ir,1],cmap = 'RdBu_r',vmax = vmax,vmin=vmin)

            n0 = np.sqrt(np.nanmean(np.abs(v0)**2))
            n1 = np.sqrt(np.nanmean(np.abs(v1)**2))
            n2 = np.nanmean(np.abs(adv)**2)
            
            if ir == 1:
                adv.plot(ax = axs[ir,2],vmin = 0)
            else:
                adv.plot(ax = axs[ir,2])#,vmin = 0,vmax = 2)
            if ir == 0:
                title = 'in logscale'
            else:
                title = ''
            axs[ir,0].set_title(f'without land correction {title}')
            axs[ir,1].set_title(f'with land correction {title}')
            axs[ir,2].set_title(f'|land_corrected|/|default| {title}')

                
        print(f + '.png')
        plt.savefig(f + '.png')
        plt.close()
check_forcings()