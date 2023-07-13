import itertools
import os
import matplotlib.pyplot as plt
from constants.paths import  all_eval_path
from plots.for_paper.saliency import SubplotAxes
from utils.xarray import drop_unused_coords, skipna_mean
import xarray as xr
import numpy as np
import matplotlib

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
    # plt.rcParams.update({'font.size': 14})
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
        nrows = 1
        ncols = 2
        figsizesc = 6
        # fig,axs = plt.subplots(ncols,nrows,figsize = (figsizesc*nrows,figsizesc/3*2*ncols))
        matplotlib.rcParams.update({'font.size': 14})
        fig = plt.figure(figsize = (ncols*figsizesc,nrows*figsizesc/1.5))
        spaxes = SubplotAxes(1,ncols,sizes = ((1,),(5,2,5)),ymargs=(0.15,0.005,0.09),xmargs = (0.04,0.005,0.03))
        r2variable_names = '$R^2_u$ $R^2_T$'.split()
        corrvariable_names = '$C_u$ $C_T$'.split()
        for i,j in itertools.product(range(nrows),range(ncols)):
            # ax = axs[i,j]
            # ax = axs[i]            
            ax = fig.add_axes(spaxes.get_ax_dims(i,j*2))
            # y,rowsel = ax_sel_data([fcnn,fcnn_lsrp],i,j)
            y,rowsel = ax_sel_data([fcnn,],j*2,i)
            ylsrp,_ = ax_sel_data([lsrp,lsrp],j*2,i)
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
            import matplotlib as mpl
            color_offset = 10
            norm = mpl.colors.Normalize(vmin=0, vmax=len(DEPTHS) + color_offset-1)
            cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)
            cmap.set_array([])

            colors = [cmap.to_rgba(i )for i in range(color_offset,len(DEPTHS)+color_offset)]
            markers = ['^','v','<','>','s','p','D']
            for l in range(len(DEPTHS)):
                yl = y.isel(training_depth =l)
                # ax.plot(ixaxis,yl,f'{markers[l]}',linestyle = 'dotted', label = f'{str(int(DEPTHS[l]))} m',markersize = 6)
                ax.plot(ixaxis,yl,c = colors[l],marker = markers[l],linestyle = 'dotted', label = f'{str(int(DEPTHS[l]))} m',markersize = 6)
            # for l in range(len(DEPTHS)):
            #     yl = y.isel(training_depth =l)
            #     ax.plot(ixaxis[l],yl.values[l],'k.',markersize = 12)
            ax.plot(ixaxis,ylsrp,f'{markers[-1]}',fillstyle = 'none',c = 'red',linestyle = 'dotted', label = 'Linear',markersize = 6)
            if negative_r2_management:
                ax.set_ylim([-.25,1.05])
                ax.set_yticks(yticks)
                ax.set_yticklabels(yticklabels)
            else:
                pass
                # ax.set_ylim(ylim)
            ax.set_xticks(ixaxis)
            xaxis = DEPTHS#y.depth.values
            xaxis = [int(v) for v in xaxis]#["{:.2e}".format(v) for v in xaxis]
            
            ax.set_xticklabels(xaxis)
            # ax.legend()
            if j == 0:
                ax.legend(bbox_to_anchor=(1.025, 0.6), loc="center left")
            ax.grid(which = 'major',color='k', linestyle='--',linewidth = 1,alpha = 0.8)
            ax.grid(which = 'minor',color='k', linestyle='--',linewidth = 1,alpha = 0.6)
            
            
            
            if not r2corr:
                vn = r2variable_names[j]
            else:
                vn = corrvariable_names[j]
            title = vn# + f': $\kappa$={sigma}'
            ax.set_title(title)
            ax.set_xlabel('Test depth (m)')
        target_folder = 'paper_images/depth'
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        # plt.subplots_adjust(bottom=0.12, right=0.98, top=0.91, left= 0.05)
        # fig.savefig(os.path.join(target_folder,f'{r2corr_str}_sigma_{sigma}.png'),transparent=True)
        fig.savefig(os.path.join(target_folder,f'{r2corr_str}_sigma_{sigma}.png'),transparent=False)
        print(os.path.join(target_folder,f'{r2corr_str}_sigma_{sigma}.png'))


def main():
    all_eval_filename = '/scratch/cg3306/climate/outputs/evals/all20230615.nc' #all_eval_path()
    stats = xr.open_dataset(all_eval_filename).sel(filtering = 'gcm')
    depth_plot(stats)
if __name__=='__main__':
    main()