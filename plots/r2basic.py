import itertools
import os
import matplotlib.pyplot as plt
from constants.paths import  all_eval_path
from utils.xarray import skipna_mean
import xarray as xr
import numpy as np

def class_functions(Foo):
    return [func for func in dir(Foo) if callable(getattr(Foo, func)) and not func.startswith("__")]

MODELCODES = {
    'lossfun' : {'MSE':'','heteroscedastic':'htr-','MVARE':''},
    'model': {'fcnn':'','lsrp:0':'lsrp','lsrp:1':'lsrp1','dfcnn':''},
    'domain': {'four_regions':'R4','global':'G'},
    'latitude':{1:'L',0:''},
    'temperature':{1:'T',0:''},
    'lsrp':{0:'',1:'-lsr'},
}
namecut = dict(
    model =  ['lsrp:0','lsrp:1'],
)
def build_name(ckeys,cvals):
    name = ''
    for mk in MODELCODES:
        if mk not in ckeys:
            continue
        subf = cvals[ckeys.index(mk)]
        nd = MODELCODES[mk][subf]
        if mk in namecut:
            if subf in namecut[mk]:
                return  name + nd
        else:
            name =  name + nd
    return name


def append_names(stats):
    coords = stats.coords
    cvals = [coords[key].values for key in coords.keys()]

    ckeys = list(coords.keys())
    mkeys = list(MODELCODES.keys())
    mkeys = [mk for mk in mkeys if mk in ckeys]
    names = []
    for valp in itertools.product(*cvals):
        name = build_name(ckeys,valp)
        names.append(name)
    uniqnames = np.unique(names)
    shp = [len(cv) for cv in cvals]
    names = np.reshape(names,shp)
    
    for name, datavar in stats.data_vars.items():
        dims = datavar.dims
        break
    
    dimsi = [ckeys.index(dm) for dm in dims]
    names = np.transpose(names,dimsi)
    names = xr.DataArray(dims = dims, data = names,coords = stats.coords)

    stats['name'] = names
    return stats
    modelsdict = []
    for uname in uniqnames:
        _stats = xr.where(names == uname,stats,np.nan)
        _nancount= xr.where(np.isnan(_stats),0,1)
        _values = xr.where(np.isnan(_stats),0,_stats)
        st = _values.sum(dim = mkeys)/_nancount.sum(dim = mkeys)
        st = st.expand_dims(dim = {'modelname':[uname]})
        modelsdict.append(st.copy())
    # models = xr.merge(modelsdict)
    # models = xr.where(models.training_depth == models.depth,models,np.nan)
    # models = models.isel(training_depth = 0,sigma = 3,co2 = 0,depth = 0).drop(['co2','depth','training_depth'])
    # print(models.ST_r2)
def prepare_stats():
    all_eval_filename = '/scratch/cg3306/climate/outputs/evals/all20230615.nc' #all_eval_path()
    stats = xr.open_dataset(all_eval_filename)
    # lsrp = stats.isel(model = 1)
    # skipna_mean(lsrp,)

    stats = stats.isel(training_depth = 0,depth = 0, co2 = 0).drop(['training_depth','depth','co2'])
    kernels_dict = {
        4:21,
        8:11,
        12:7,
        16:7
    }
    # for sigma,kernel_size in kernels_dict.items():
    #     if sigma == 4:
    #         mask = (stats.sigma == sigma) *(stats.kernel_size == kernel_size)
    #     else:
    #         mask += (stats.sigma == sigma) *(stats.kernel_size == kernel_size)
    # stats = xr.where(mask>0,stats,np.nan )
    stats = append_names(stats)
    
    keepcoords = ['sigma','seed']
    cnames =[k for k in stats.coords.keys() if k not in keepcoords]
    
    # for dc in dropcoords:
    #     cnames.pop(dc)
    # cnames = list(cnames.keys())
    unames = np.unique(stats.name.values)
    stats_ns = []
    for un in unames:
        stats_n = xr.where(stats.name == un,stats,np.nan).drop('name')
        stats_n = skipna_mean(stats_n,dim = cnames)
        stats_n = stats_n.expand_dims(dim = {'name' : [un]},axis= 0)
        stats_ns.append(stats_n)
    stats_ns = xr.merge(stats_ns)#.isel(co2 = 0,training_depth = 0,depth = 0).drop(['training_depth','depth','co2'])
    return stats_ns
def generate_plots():
    stats_ns = prepare_stats()
    keepnames = []
    for name in stats_ns.name.values:
        if not np.all(np.isnan(stats_ns.sel(name = name).Su_r2.values)):
            if 'htr-' in name and name != 'htr-lsrp':
                continue
            keepnames.append(name)
    
    stats_ns = stats_ns.sel(name = keepnames)

    
    plot(stats_ns,'G GT GLT R4 R4T htr-lsrp'.split(),'base')
    # print(stats_ns.name.values,'R4 R4T R4LT','r4')
    return
    
def plot(stats_ns_,name_select,imgtag):
    stats_ns = stats_ns_.sel(name = name_select)
    print(stats_ns.sel(name = ['R4','G'],seed = 0,sigma = 4))
    return
    ranks = {}
    for name in stats_ns.name.values:
        ranks[name] = 0
        if 'R4' in name:
            ranks[name] -= 1e6
        if 'lsr' in name:
            ranks[name] += 1e4
        if 'lsrp' in name:
            ranks[name] -= 1e7
        ranks[name] +=  len(name.replace('R4','').replace('G',''))
    namesort = np.argsort(list(ranks.values()))

    varnames = list(stats_ns.data_vars.keys())
    vartypes = np.unique([vn.split('_')[1] for vn in varnames])
    colors = 'r b g k'.split()
    markers = 'o ^ v < >'.split()
    renames = {'htr-lsrp':'LSRP'}
    for vtype in vartypes:
        vnselect = [vn for vn in varnames if vn.split('_')[1] == vtype]
        ncols = len(vnselect)
        nrows = 1
        fig,axs = plt.subplots(nrows,ncols, figsize = (ncols*5,nrows*5))
        for i in range(ncols):
            vname = vnselect[i]
            ax = axs[i]
            for j in range(len(stats_ns.sigma)):
                vals = stats_ns.isel(sigma = j).mean(dim = 'seed')
                y = vals[vname].values[namesort]
                if np.all(np.isnan(y)):
                    continue
                x = np.arange(len(y))
                nonans = [not np.all(np.isnan(stats_ns[vname].sel(name = x,).values)) for x in stats_ns.name.values[namesort].tolist()]
                nonans = np.array(nonans)
                xticklabels =  [str(x) for x in stats_ns.name.values[namesort].tolist()]
                for rnk,rnv in renames.items():
                    xtickind = xticklabels.index(rnk)
                    xticklabels[xtickind] = rnv
                x = x[nonans]
                y = y[nonans]
                y = np.where(y < 0,y/16,y)
                xticklabels = [xticklabels[i] for i in range(len(xticklabels)) if nonans[i]]
                if vtype in ['r2','corr']:
                    ax.plot(x,y,\
                        label = f"\u03C3 = {vals.sigma.values.item()}",\
                        color = colors[j],\
                        marker = markers[j],linestyle = 'None')
                else:
                    ax.semilogy(x,y,\
                        label = f"\u03C3 = {vals.sigma.values.item()}",\
                        color = colors[j],\
                        marker = markers[j],linestyle = 'None')
                ax.set_xticks(x)
                ax.set_xticklabels(xticklabels,rotation=45)
            if vtype in ['r2','corr']:
                ax.set_ylim([-0.25,1.01])
                yticks = [-0.25,-0.125,0,0.25,0.5,0.75,1]
                yticklabels = ['-4','-2','0','0.25','0.5','0.75','1.']
                ax.set_yticks(yticks)
                ax.set_yticklabels(yticklabels)
            ax.grid( which='major', color='k', linestyle='--',alpha = 0.5)
            ax.grid( which='minor', color='k', linestyle='--',alpha = 0.5)
            if vtype in ['r2','corr']:
                ax.legend(loc = 'lower right')
            else:
                ax.legend(loc = 'upper right')
            ax.set_title(vname.replace('_',' ').capitalize())
            # ax.set_ylabel(vtype)
        filename = f"{vtype}.png"
        
        target_folder = 'paper_images/hierarchy'
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        fig.savefig(os.path.join(target_folder,filename))
        print(os.path.join(target_folder,filename))
        plt.close()





def main():
    
    
    generate_plots()

    # print(statsmax)
    return
    ylim = [0,1]
    colsep = {'latitude_features':[0,1,0,1],'CNN_LSRP':[0,0,1,1]}
    title_naming = ['latitude','LSRP']
    linsep = 'training_depth'
    xaxisname = 'depth'
    ncol = 4
    rowsep = list(r2vals.data_vars)
    nrow = len(rowsep)
    fig,axs = plt.subplots(nrow,ncol,figsize = (9*ncol,6*nrow))
    for i,j in itertools.product(range(nrow),range(ncol)):
        ax = axs[i,j]
        colsel = {key:val[j] for key,val in colsep.items()}
        rowsel = rowsep[i]
        y = r2vals[rowsel]
        ylsrp = y.sel(LSRP = 1).isel(**{key:0 for key in colsel})
        ylsrp = ylsrp.isel({linsep : 0})
        y = y.sel(**colsel)        
        y = y.sel(LSRP = 0)
        ixaxis = np.arange(len(ylsrp))
        for l in range(len(y[linsep])):
            yl = y.isel({linsep : l})
            ax.plot(ixaxis,yl,label = str(yl[linsep].values))
            ax.plot(ixaxis[l],yl.values[l],'k.',markersize = 12)
        ax.plot(ixaxis,ylsrp,'--',label = 'LSRP')
        ax.set_ylim(ylim)
        ax.set_xticks(ixaxis)
        xaxis = ylsrp[xaxisname].values
        xaxis = ["{:.2e}".format(v) for v in xaxis]
        ax.set_xticklabels(xaxis)
        ax.legend()
        ax.grid(which = 'major',color='k', linestyle='--',linewidth = 1,alpha = 0.8)
        ax.grid(which = 'minor',color='k', linestyle='--',linewidth = 1,alpha = 0.6)
        if j==0:
            ax.set_ylabel(rowsel)
        if i==0:
            title = ", ".join([f"{n}:{bool(v)}" for n,v in zip(title_naming,colsel.values())])
            ax.set_title(title)
    fig.savefig(os.path.join(target,'depth_comparison.png'))
if __name__=='__main__':
    main()