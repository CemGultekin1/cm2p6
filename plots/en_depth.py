import itertools
import os
import matplotlib.pyplot as plt
from constants.paths import  DISTS, all_eval_path
from plots.for_paper.saliency import SubplotAxes
from utils.xarray_oper import drop_unused_coords, skipna_mean
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
    # plt.rcParams.update({'font.size': 14})
    
    stats_ = stats.isel(domain = 1, temperature = 1, sigma = range(4),co2= 0,training_depth =[i for i in range(8) if i!=2 ] ,)
    from data.coords import DEPTHS
    DEPTHS = list(DEPTHS)
    DEPTHS.pop(2)
    
    for sigma_i,r2corr in itertools.product(range(4),range(2)):
        sigma = stats_.sigma.values[sigma_i]
        
        nrows = 1
        ncols = 2
        figsizesc = 6
        # fig,axs = plt.subplots(ncols,nrows,figsize = (figsizesc*nrows,figsizesc/3*2*ncols))
        fig = plt.figure(figsize = (ncols*figsizesc,nrows*figsizesc/1.5))
        spaxes = SubplotAxes(1,ncols,sizes = ((1,),(5,2,5)),ymargs=(0.12,0.01,0.1),xmargs = (0.05,0.01,0.1))
        r2variable_names = '$\mathcal{E}^2_u$ $\mathcal{E}^2_T$'.split()
        corrvariable_names = '$\mathcal{E}^2_u$ $\mathcal{E}^2_T$'.split()
        varnames = 'Su_test Stemp_test'.split()
        for i,j in itertools.product(range(nrows),range(ncols)):   
            ax = fig.add_axes(spaxes.get_ax_dims(i,j*2))
            y = stats_[varnames[j]]

            ixaxis = np.arange(len(y.depth))
            
            yticks = [0,0.2,0.4,0.6,0.8,1.]
            yticklabels = [
                '0','0.2','0.4','0.6','0.8','1.'
            ]

                
            markers = ['^','v','<','>','s','p','D']
            for l in range(len(DEPTHS)):
                yl = y.isel(training_depth =l,sigma =sigma_i)
                ax.semilogy(ixaxis,yl,f'{markers[l]}',linestyle = 'dotted', label = f'{str(int(DEPTHS[l]))} m',markersize = 6)

            ax.set_xticks(ixaxis)
            xaxis = DEPTHS#y.depth.values
            xaxis = [int(v) for v in xaxis]#["{:.2e}".format(v) for v in xaxis]
            
            ax.set_xticklabels(xaxis)
            # ax.legend()
            if j == 0:
                ax.legend(bbox_to_anchor=(1.1, 0.5), loc="center left")
            ax.grid(which = 'major',color='k', linestyle='--',linewidth = 1,alpha = 0.8)
            ax.grid(which = 'minor',color='k', linestyle='--',linewidth = 1,alpha = 0.6)
            
            
            
            if not r2corr:
                vn = r2variable_names[j]
            else:
                vn = corrvariable_names[j]
            title = vn + f': \u03C3={sigma}'
            ax.set_title(title)
            ax.set_xlabel('test depths (m)')
        target_folder = 'paper_images/depth'
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        r2corr_str = 'en'
        plt.subplots_adjust(bottom=0.12, right=0.98, top=0.91, left= 0.05)
        fig.savefig(os.path.join(target_folder,f'{r2corr_str}_sigma_{sigma}.png'),transparent=True)
        fig.savefig(os.path.join(target_folder,f'{r2corr_str}_sigma_{sigma}_.png'),transparent=False)
        print(os.path.join(target_folder,f'{r2corr_str}_sigma_{sigma}.png'))

def main():
    root = DISTS
    filename = os.path.join(root,'all.nc')
    ds = xr.open_dataset(filename,mode = 'r')
    depth_plot(ds)

if __name__=='__main__':
    main()