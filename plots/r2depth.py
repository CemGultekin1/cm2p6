import itertools
import os
import matplotlib.pyplot as plt
from constants.paths import  all_eval_path
from utils.xarray import drop_unused_coords, skipna_mean
import xarray as xr
import numpy as np

def class_functions(Foo):
    return [func for func in dir(Foo) if callable(getattr(Foo, func)) and not func.startswith("__")]

MODELCODES = {
    'model': {'fcnn':'','lsrp:0':'lsrp','lsrp:1':'lsrp1'},
    'domain': {'four_regions':'R4','global':'G'},
    'latitude':{1:'L',0:''},
    'temperature':{1:'T',0:''},
    'lsrp':{0:'',1:'-lsr',2:'-lsr1'},
}
namecut = dict(
    model =  ['lsrp:0','lsrp:1'],
)


def group_by_extension(ds,groups):
    grouping = {}
    for name in ds.data_vars.keys():
        for kc in groups:
            if kc in name:
                if kc not in grouping:
                    grouping[kc] = []
                v = ds[name]
                v.name = v.name.replace(f"_{kc}",'')
                grouping[kc].append(v)
                break

    for kc,vals in grouping.items():
        grouping[kc] = xr.merge(vals)
    return list(grouping.values())

def ax_sel_data(dsl,i,j):
    ds = dsl[j]
    keys = list(ds.data_vars.keys())
    keysi = keys[i] 
    return ds[keysi],keysi
def separate_lsrp_values(stats):
    # lossfun = 'MSE',
    lsrpmodel = stats.isel(seed = 0,latitude = 0, domain = 0,\
                temperature = 0,lsrp = 0,training_depth = 0,\
                co2 = 0,model = 1).sel(lossfun = 'heteroscedastic',kernel_size = 21,)
    
    nonlsrp = stats.isel(model = 0,).sel(lossfun = 'MSE')
    return drop_unused_coords(nonlsrp),drop_unused_coords(lsrpmodel)

def depth_plot(stats):
    stats,lsrp_ = separate_lsrp_values(stats)#.isel(sigma= range(1,4))
    kernels_dict = {
        4:21,
        8:11,
        12:7,
        16:7
    }
    
    stats_ = stats.isel(seed = 0,latitude = 0, \
            domain = 1, temperature = 1, sigma = range(4),co2= 0,training_depth =[i for i in range(8) if i!=2 ] ,)
    from data.coords import DEPTHS
    DEPTHS = list(DEPTHS)
    DEPTHS.pop(2)
    for sigma_i,r2corr in itertools.product(range(4),range(2)):
        sigma = stats_.sigma.values[sigma_i]
        kernel_size = kernels_dict[sigma]
        
        stats = stats_.isel(sigma = sigma_i).sel(kernel_size = kernel_size)
        lsrp = lsrp_.isel(sigma = sigma_i)#.sel(kernel_size = kernel_size)

        fcnn = group_by_extension(stats,'r2 corr'.split())[r2corr]
        lsrp = group_by_extension(lsrp,'r2 corr'.split())[r2corr]
        r2corr_str = 'r2' if not r2corr else 'corr'
        # fcnn = stats.isel(model = 0,)
        fcnn_lsrp = fcnn.isel(lsrp = 1)
        fcnn = fcnn.isel(lsrp = 0)

        fcnn,fcnn_lsrp,lsrp = drop_unused_coords(fcnn),drop_unused_coords(fcnn_lsrp),drop_unused_coords(lsrp)

        ylim = [0,1]
        nrows = 3
        ncols = 1#2
        fig,axs = plt.subplots(nrows,ncols,figsize = (9*ncols,6*nrows))
        varnames = 'Su Sv Stemp'.split()
        for i,j in itertools.product(range(nrows),range(ncols)):
            # ax = axs[i,j]
            ax = axs[i]
            # y,rowsel = ax_sel_data([fcnn,fcnn_lsrp],i,j)
            y,rowsel = ax_sel_data([fcnn,],i,j)
            ylsrp,_ = ax_sel_data([lsrp,lsrp],i,j)
            ixaxis = np.arange(len(y.training_depth))
            # print(y.training_depth.values)
            # print(y.depth.values)
            # print(DEPTHS)
            # raise Exception
            
            minimumr2 = np.amin(y.values)
            minimumr2 = np.minimum(minimumr2,np.amin(ylsrp.values ))
            negative_r2_management = minimumr2 < 0
            if negative_r2_management:
                ymin = np.floor(minimumr2)
                ymax = 1.
                y = xr.where(y < 0, -y/ymin/5,y)
                ylsrp = xr.where(ylsrp < 0, -ylsrp/ymin/5,ylsrp)
                yticks = [-.2,-.1,0,0.2,0.4,0.6,0.8,1.]
                yticklabels = [
                    str(int(ymin)),str(ymin/2),'0','0.2','0.4','0.6','0.8','1.'
                ]
                
                
            markers = ['^','v','<','>','s','p','D']
            for l in range(len(DEPTHS)):
                yl = y.isel(training_depth =l)
                ax.plot(ixaxis,yl,f'{markers[l]}--',label = f'{str(int(DEPTHS[l]))} m',markersize = 4)
            for l in range(len(DEPTHS)):
                yl = y.isel(training_depth =l)
                ax.plot(ixaxis[l],yl.values[l],'k.',markersize = 12)
            ax.plot(ixaxis,ylsrp,f'{markers[-1]}--',label = 'LSRP',markersize = 4)
            if negative_r2_management:
                ax.set_ylim([-.25,1.05])
                ax.set_yticks(yticks)
                ax.set_yticklabels(yticklabels)
            else:
                ax.set_ylim(ylim)
            ax.set_xticks(ixaxis)
            xaxis = DEPTHS#y.depth.values
            xaxis = [int(v) for v in xaxis]#["{:.2e}".format(v) for v in xaxis]
            
            ax.set_xticklabels(xaxis)
            ax.legend()
            ax.grid(which = 'major',color='k', linestyle='--',linewidth = 1,alpha = 0.8)
            ax.grid(which = 'minor',color='k', linestyle='--',linewidth = 1,alpha = 0.6)
            # if j==0:
            #     ax.set_ylabel(rowsel)
            title = varnames[i] + ' '+ r2corr_str.capitalize() + f' \u03C3={sigma}'
            ax.set_title(title)
            ax.set_xlabel('depths (m)')
        target_folder = 'paper_images/depth'
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        fig.savefig(os.path.join(target_folder,f'{r2corr_str}_sigma_{sigma}.png'))
        print(os.path.join(target_folder,f'{r2corr_str}_sigma_{sigma}.png'))


def main():
    all_eval_filename = '/scratch/cg3306/climate/outputs/evals/all20230615.nc' #all_eval_path()
    stats = xr.open_dataset(all_eval_filename).sel(filtering = 'gcm')
    depth_plot(stats)
if __name__=='__main__':
    main()