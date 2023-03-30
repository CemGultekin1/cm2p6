from data.load import load_xr_dataset,load_grid
import matplotlib.pyplot as plt
import numpy as np
earth_radius_in_meters = 6.3710072e6
import xarray as xr

from data.paths import get_high_res_grid_location
def main():
    # path = '/scratch/zanna/data/cm2.6/geometry/ocean.static.CM2p6.nc'
    # grid_data = xr.open_dataset(path)#get_high_res_grid_location())
    # print(grid_data)


    args = '--mode data'.split()
    ds, _ = load_xr_dataset(args,high_res = True)
    ds = load_grid(ds)    
    
    fig,axs  = plt.subplots(1,2,figsize = (30,15))
    
    mean_dxu = ds.dxu.mean(dim ='xu_ocean').values
    mean_dyu = ds.dyu.mean(dim ='xu_ocean').values
    
    print(f'maximum dxu = {np.amax(ds.dxu.values)}')
    
    xu = np.sort(ds.ulon.values)
    yu = np.sort(ds.ulat.values)
    dxu = xu[1:] - xu[:-1]
    med_dxu = np.median(dxu)
    print(f'med_dxu = {med_dxu}')
    dxu = med_dxu*np.ones(len(xu))
    dxu = dxu*earth_radius_in_meters/180*np.pi
    
    dyu = yu[1:] - yu[:-1]
    dyu = np.concatenate([dyu,dyu[-1:]])
    dyu = dyu*earth_radius_in_meters/180*np.pi
    
    
    axs[0].plot(mean_dxu,label = 'true')
    axs[0].plot(dxu,label = 'estimate')
    axs[0].legend()
    mean_true_dxu = np.mean(mean_dxu)
    mean_our_dxu = np.mean(dxu)
    axs[0].set_title(f'true_mean = {mean_true_dxu},\tpred_mean = {mean_our_dxu}')
    axs[1].plot(mean_dyu,label = 'true')
    axs[1].plot(dyu,label = 'estimate')
    axs[1].legend()
    fig.savefig('grid_seapartions.png')
if __name__ == '__main__':
    main()