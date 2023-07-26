import itertools
import logging
import os
import matplotlib.pyplot as plt
from constants.paths import  all_eval_path
from plots.basic import Multiplier, ReadMergedMetrics
from plots.for_paper.saliency import SubplotAxes
from utils.xarray_oper import drop_unused_coords, skipna_mean
import xarray as xr
import numpy as np
import matplotlib
logging.basicConfig(level = logging.INFO)
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

def plot(stats,lsrp,filenametag):
    r2corr = 0
    fcnn = group_by_extension(stats,'r2 corr'.split())[r2corr]
    lsrp = group_by_extension(lsrp,'r2 corr'.split())[r2corr]
    r2corr_str = 'r2' if not r2corr else 'corr'        

    fcnn,lsrp = drop_unused_coords(fcnn),drop_unused_coords(lsrp)

    nrows = 1
    ncols = 2
    figsizesc = 6
    # fig,axs = plt.subplots(ncols,nrows,figsize = (figsizesc*nrows,figsizesc/3*2*ncols))
    matplotlib.rcParams.update({'font.size': 14})
    fig = plt.figure(figsize = (ncols*figsizesc,nrows*figsizesc/1.5))
    spaxes = SubplotAxes(1,ncols,sizes = ((1,),(5,5)),ymargs=(0.15,0.005,0.09),xmargs = (0.04,0.03,0.03))
    r2variable_names = dict(zip('Su Sv Stemp'.split(), '$R^2_u$ $R^2_v$ $R^2_T$'.split()))
    corrvariable_names = dict(zip('Su Sv Stemp'.split(), '$C_u$ $C_v$ $C_T$'.split()))
    axs = {}
    for i,chan in zip(np.arange(ncols),'Su Stemp'.split()):
        ax = fig.add_axes(spaxes.get_ax_dims(0,i))
        axs[i] = ax
        # logging.info(f'stats = {fcnn}')
        # logging.info(f'lsrp = {lsrp}')
        y = fcnn[chan]
        ylsrp = lsrp[chan]
        ixaxis = np.arange(len(y.training_depth))
        
        minimumr2 = np.amin(y.values)
        minimumr2 = np.minimum(minimumr2,np.amin(ylsrp.values ))
        negative_r2_management = minimumr2 < 0
        ymin = -0.05
        ymax  = 1.05
        
        if negative_r2_management:
            ymin = np.floor(minimumr2)
            ymax = 1.05
            y = xr.where(y < 0, -y/ymin/5,y)
            ylsrp = xr.where(ylsrp < 0, -ylsrp/ymin/5,ylsrp)
            yticks = [-.2,-.1,0,0.2,0.4,0.6,0.8,1.]
            yticklabels = [
                str(int(ymin)),str(ymin/2),'0','0.2','0.4','0.6','0.8','1.'
            ]
            ymin = -.2
        
        import matplotlib as mpl
        color_offset = 10
        norm = mpl.colors.Normalize(vmin=0, vmax=len(stats.depth) + color_offset-1)
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)
        cmap.set_array([])

        colors = [cmap.to_rgba(i )for i in range(color_offset,len(stats.depth)+color_offset)]
        markers = ['^','v','<','>','s','p','D']
        npdtph = len(stats.depth)+1
        width = 0.8/npdtph
        
        for l in range(len(stats.depth)):
            yl = y.isel(training_depth =l)
            # ixaxisoff = ixaxis + width*(l - npdtph/2)
            # ax.bar(ixaxisoff,yl,color = colors[l],label = f'{str(int(stats.depth.values[l]))} m',width = width)
            ax.plot(ixaxis,yl,color = colors[l],label = f'{str(int(stats.depth.values[l]))} m',\
                    linestyle = 'dotted',marker = markers[l],markersize = 6)
        # ax.bar(ixaxis + width*(npdtph-1 - npdtph/2),ylsrp,color = 'red',width = width,label = 'linear')
        ax.plot(ixaxis,ylsrp,color = 'r',label = f'Linear',\
                        linestyle = 'dotted',marker = markers[-1],markersize = 6,fillstyle= 'none')
        ax.set_ylim([ymin,ymax])
        if negative_r2_management:            
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels)
        ax.set_xticks(ixaxis)
        xaxis = stats.depth.values
        xaxis = [int(v) for v in xaxis]#["{:.2e}".format(v) for v in xaxis]
        
        ax.set_xticklabels(xaxis)
        # ax.legend()
        
        ax.grid(which = 'major',color='k', linestyle='--',linewidth = 1,alpha = 0.8)
        ax.grid(which = 'minor',color='k', linestyle='--',linewidth = 1,alpha = 0.6)
        
        
        
        if not r2corr:
            vn = r2variable_names[chan] 
        else:
            vn = corrvariable_names[chan]
        title = vn# + f': $\kappa$={sigma}'
        # ax.set_title(title)
        # ax.set_xlabel('Test depth (m)')
    target_folder = 'paper_images/depth'
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    # fig.savefig(os.path.join(target_folder,f'depth_{r2corr_str}_{filenametag}_xlabel.png'),transparent=False)
    axs[0].legend(loc="lower left",fontsize = 10,\
        handlelength =1,ncol= 5,title="Training depth (m)",title_fontsize = 10)
    if 'heteroscedastic' in filenametag:
        axs[0].set_xlabel('Test depths (m)')
        axs[1].set_xlabel('Test depths (m)')
    fig.savefig(os.path.join(target_folder,f'depth_{r2corr_str}_{filenametag}.png'),transparent=True)
    print(os.path.join(target_folder,f'depth_{r2corr_str}_{filenametag}.png'))
    plt.close()
    # raise Exception

def main():
    linear = ReadMergedMetrics(model = 'linear',date='2023-07-18')
    fcnn = ReadMergedMetrics(model = 'fcnn',date='2023-07-18')

    linear.reduce_coord('filtering','ocean_interior')
    
    fcnn.reduce_usual_coords()

    fcnn.pick_training_value('filtering',)
    fcnn.pick_matching_ocean_interiority()

    fcnn.diagonal_slice(dict(
        sigma = (4,8,12,16),
        stencil = (21,11,7,7)
    ),)
    fcnn.sel(domain = 'global')
    fcnn.to_integer_depths()
    linear.to_integer_depths()
    fcnn.pick_matching_depths()
    coords = [c for c in fcnn.remaining_coords_list() if c not in ('depth','training_depth')]
    multip = Multiplier(*coords)
    multip.recognize_values(fcnn.metrics)
    for seldict,(ds0,ds1) in multip.iterate_fun(fcnn.metrics,linear.metrics):        
        def value_transform(val):
            if isinstance(val,float):
                return '{:.2f}'.format(val).replace('.','p')
            return val
        
        filenametag = '_'.join([f'{key}_{value_transform(val)}' for key,val in seldict.items()])
        plot(ds0,ds1,filenametag)
if __name__=='__main__':
    main()