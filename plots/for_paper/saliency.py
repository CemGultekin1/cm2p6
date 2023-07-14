import itertools
import os
from typing import List
import matplotlib.pyplot as plt
from constants.paths import  all_eval_path
from models.load import load_model
from utils.xarray_oper import drop_unused_coords, plot_ds, skipna_mean
import xarray as xr
import numpy as np
from constants.paths import SALIENCY
from run.analysis.saliency import MultiAdaptiveHistograms

def get_median_from_frequency(freqs,edges):
    cdf = np.cumsum(freqs)
    cdf = cdf/cdf[-1]
    i = np.where(cdf > 0.5)[0][0]
    p0 = cdf[i-1]
    p1 = cdf[i]
    x0 = edges[i]
    x1 = edges[i+1]
    # p0 + (p1-p0)/(x1-x0)*(x - x0) = 0.5 
    xm = (0.5  - p0)*(x1-x0)/(p1-p0) + x0
    
    m0 = (xm - x0)/(x1 - x0)
    m1 = (x1 - xm)/(x1 - x0)
    freqs0,freqs1 = freqs.copy(),freqs.copy()
    freqs0[i] = m0 * freqs0[i]
    freqs1[i] = m1 * freqs1[i]
    freqs0[i+1:] = 0
    freqs1[:i] = 0
    return xm,(freqs0,freqs1)
class EnergyDecayWithRadii:
    def __init__(self,ds:xr.Dataset,):        
            self.empirical_cdf = ds.values
            self.midpoints = ds['midpts'].values
    def compute_central_energy(self,):
        quarters = np.zeros((3,self.empirical_cdf.shape[1]))
        for i in range(1,self.empirical_cdf.shape[1]):
            for j,p in enumerate([0.25,0.5,0.75]):
                k = np.where(self.empirical_cdf[:,i]>p)[0][0]
                quarters[j,i] = self.midpoints[k]
        return quarters
    def plot_quarters(self,ax:plt.Axes,):
        quarters = self.compute_central_energy()
        quarters = -quarters[:,1:]
        xaxis = range(1,10)
        color = 'b'
        # ax.semilogy(xaxis,quarters[1],'o',color = color,)#label = tag)
        ax.errorbar(xaxis,quarters[1],\
            yerr = (np.abs(np.abs(quarters[2]-quarters[1]),quarters[1] - quarters[0])),\
            fmt='x',linestyle = 'dotted', color=color,ecolor='black',capsize=5,)#mew = 4)
        # ax.set_yscale('log')
        ax.grid(which = 'major',color='k', linestyle='--',linewidth = 1,alpha = 0.4)
        ax.grid(which = 'minor',color='k', linestyle='--',linewidth = 1,alpha = 0.3)
        
        ax.set_xticks(xaxis)
        xticklabels = [f'{np.maximum(0,2*g -1)}x{np.maximum(0,2*g -1)}' for g in xaxis]
        ax.set_xticklabels(xticklabels)#[str(xa -1 ) for xa in xaxis])
        # ax.legend()
        ax.set_ylim([-.5,3.5])
        ax.set_xlim([1.5,5.5])
        
class SubplotAxes:
    def __init__(self,nrows,ncols,xmargs = (0.05,0.03,0.),ymargs = (0.05,0.01,0.01),sizes = None):
        self.nrows = nrows
        self.ncols = ncols
        self.xmargs = xmargs
        self.ymargs = ymargs
        if sizes is None:
            sizes = (np.ones(nrows),np.ones(ncols))
        self.sizes =sizes
    def get_ax_dims(self,i,j):
        dy = (1-self.ymargs[0] - self.ymargs[2] - 2*self.ymargs[1]*(self.nrows - 1))/np.sum(self.sizes[0])
        dx = (1-self.xmargs[0] - self.xmargs[2] - 2*self.xmargs[1]*(self.ncols - 1))/np.sum(self.sizes[1])
        i = self.nrows - i - 1
        xloc = dx*np.sum(self.sizes[1][:j]) + self.xmargs[0] + self.xmargs[1]*2*j
        xw = dx*self.sizes[1][j]
        yloc = dy*np.sum(self.sizes[0][:i]) + self.ymargs[0] + self.ymargs[1]*2*i
        yw = dy*self.sizes[0][i]
        return [xloc,yloc,xw,yw]
def main():
    from utils.slurm import read_args    
    
    # fig,axs = plt.subplots(2,3,figsize = (14/1.5,8/1.8))
    # fig = plt.figure(figsize = (10,6))
    # subaxes = SubplotAxes(2,3,xmargs=(0.05,0.03,0.05),ymargs = (0.05,0.05,0.05))
    sigmas = [4,8,12]
    colors = 'r g b y'.split()
    
    varrename = dict(
        Su = '$S_u$',\
        Sv = '$S_v$',\
        Stemp = '$S_T$',\
    )
    targetfolder = 'paper_images/saliency'
    if not os.path.exists(targetfolder):
        os.makedirs(targetfolder)
    for i,sigma in enumerate(sigmas):
        args = read_args(i+1,filename = 'saliency.txt')
        modelid,_,_,_,_,_,_,_=load_model(args)
        path = os.path.join(SALIENCY,modelid + '.nc')
        mah = MultiAdaptiveHistograms()
        mah.load_from_file(os.path.join(SALIENCY,f'{modelid}.npz'))
        
        ahists = [mah.ahists[i] for i in [0,2]]
        for j,(mahi,varname) in enumerate(zip(ahists,'Su Stemp'.split())):
            # ax = axs[j,i]
            ds = mahi.to_xarray(varname)
            edr = EnergyDecayWithRadii(ds)
            # dims = subaxes.get_ax_dims(j,i)
            # ax = fig.add_axes(dims)
            import matplotlib
            matplotlib.rcParams.update({'font.size': 14})
            fig = plt.figure(figsize = (5,4))
            ax = fig.add_axes([0.15,0.13,0.8,0.85])
            # fig,ax = plt.subplots(1,1,figsize = (4,4))
            edr.plot_quarters(ax,)
            title = f'{varrename[varname]}: $\kappa$={sigma}'
            # ax.set_title(title)
            yticks = [1,2,3]
            ax.set_yticks(yticks)
            yticklabels = ['0.' + '9'*i for i in yticks]
            ax.set_yticklabels(yticklabels,rotation='vertical')
            if j == 1:
                ax.set_xlabel('Input stencils')
            if i == 0:
                ax.set_ylabel('$L^2$ concentration')
            
            title = title.replace('$','').replace('\kappa','kappa')
            fig.savefig(os.path.join(targetfolder,f'distribution-{title}.png'),transparent=False)
            plt.close()
    # plt.subplots_adjust(bottom=0.05, right=0.96, top=0.95, left= 0.09)
    # fig.savefig(os.path.join(targetfolder,'distribution_.png'),transparent=False)
    
    
    return
    fig,axs = plt.subplots(1,3,figsize = (24,8))
    radii = [4,5,6]
    for ax,mahi,varname in zip(axs,mah.ahists,'Su Sv Stemp'.split()):
        ds = mahi.to_xarray(varname)
        ds['midpts'] = np.power(10.,ds['midpts'])
        colors = 'r g b'.split()
        for rad,clr in zip(radii,colors):
            y = ds.sel(radius = rad)
            # y = y/np.sum(y)
            ax.semilogx(ds.midpts.values,y.values,'o',label = f'radius > {rad-1}',color = clr)
        ax.set_xlim([1e-2,5e-1])
        ax.legend()
        ax.grid(which = 'major',color='k', linestyle='--',linewidth = 1,alpha = 0.8)
        ax.grid(which = 'minor',color='k', linestyle='--',linewidth = 1,alpha = 0.6)
        ax.set_title(varname)
    fig.savefig('distributions.png')
    plt.close()
    return

    stats = xr.open_dataset(path).isel(co2 = 0,time = 0,depth = 0)
    mean,std = separate_mean_std(stats)
    moment2 = np.square(std) + np.square(mean)
    moment1 = mean
    ws = DatasetStacker(10,'lat lon'.split())
    moment1 = ws.stack(moment1).mean('index').drop(['lat','lon'])
    moment2 = ws.stack(moment2).mean('index').drop(['lat','lon'])
    moment1 = EnergyDecay(moment1).compute()
    moment2 = EnergyDecay(moment2).compute()
    mean = moment1
    std = np.sqrt(moment2-moment1**2)
    fig,axs = plt.subplots(3,1,figsize = (12,18))
    for v,ax in zip(mean.data_vars,axs):
        scl = np.amax(mean[v])
        ax.plot((mean[v] - std[v]/2)/scl, 'o',color = 'blue')
        ax.plot((mean[v]/scl) , 'o',color = 'blue')
        ax.plot((mean[v] + std[v]/2)/scl, 'o',color = 'blue')
        ax.set_title(v)
        ax.grid(which = 'major',color='k', linestyle='--',linewidth = 1,alpha = 0.8)
        ax.grid(which = 'minor',color='k', linestyle='--',linewidth = 1,alpha = 0.6)
    fig.savefig('saliency.png')
    return
    lsrp_ = get_lsrp_values(stats).isel(sigma= range(4))

    stats_ = stats.isel(seed = 0,latitude = 1, domain = 1, temperature = 1, sigma = range(4),co2= 0,training_depth = 0,depth = 0)
    sigma_vals = [4,8,12,16]

    
    for r2corr in range(2):#itertools.product(range(3),range(2)):
        # sigma = stats_.sigma.values[sigma_i]
        stats = stats_.copy()#isel(sigma = sigma_i)
        lsrp = lsrp_.copy()#isel(sigma = sigma_i)

        stats = group_by_extension(stats,'r2 corr'.split())[r2corr]
        lsrp = group_by_extension(lsrp,'r2 corr'.split())[r2corr]
        r2corr_str = 'r2' if not r2corr else 'corr'
        fcnn = stats.isel(model = 0)
        fcnn_lsrp = fcnn.isel(lsrp = 1)
        fcnn = fcnn.isel(lsrp = 0)

        fcnn,fcnn_lsrp,lsrp = drop_unused_coords(fcnn),drop_unused_coords(fcnn_lsrp),drop_unused_coords(lsrp)

        ylim = [0,1]
        nrows = 3
        ncols = 1#4
        fig,axs = plt.subplots(nrows,ncols,figsize = (9*ncols,6*nrows))

        fcnn_by_sigma = [fcnn.isel(sigma = ss) for ss in range(4)]
        fcnn_lsrp_by_sigma = [fcnn_lsrp.isel(sigma = ss) for ss in range(4)]
        lsrp_by_sigma = [lsrp.isel(sigma = ss) for ss in range(4)]

        for i,j in itertools.product(range(nrows),range(ncols)):
            # ax = axs[i,j]
            ax = axs[i]
            for j in range(4):
                yfcnn,rowsel = ax_sel_data(fcnn_by_sigma,i,j)
                # yfcnn_lsrp,_ = ax_sel_data(fcnn_lsrp_by_sigma,i,j)
                # ylsrp,_ = ax_sel_data(lsrp_by_sigma,i,j)
                ixaxis = np.arange(len(yfcnn.kernel_size))
                markers = 'o v ^ <'.split()
                colors = [f'tab:{x}' for x in 'blue orange green red'.split()]
                
                ax.plot(ixaxis,yfcnn,\
                    color = colors[j], marker = markers[j],label = f"\u03C3 = {sigma_vals[j]}",)#linestyle = 'None')
            # ax.plot(ixaxis,yfcnn_lsrp,\
            #     color = colors[1], marker = markers[1],label = f'FCNN+LSRP',linestyle = 'None')
            # ax.axhline(y = ylsrp.values.item(),color = colors[3], label = f'LSRP')

            ax.set_ylim(ylim)
            ax.set_xticks(ixaxis)
            xaxis = yfcnn.kernel_size.values
            xaxis = [str(v) for v in xaxis]
            ax.set_xticklabels(xaxis)
            ax.legend()
            ax.grid(which = 'major',color='k', linestyle='--',linewidth = 1,alpha = 0.8)
            ax.grid(which = 'minor',color='k', linestyle='--',linewidth = 1,alpha = 0.6)
            if j==0:
                ax.set_ylabel(rowsel)
            if i==0:
                title =f'\u03C3 = {stats.sigma.values[j]}'
                title = r2corr_str.upper() + ', ' + title
                ax.set_title(title)
            ax.set_xlabel('kernel_size')
        if not os.path.exists('saves/plots/kernel_size_comparison/'):
            os.makedirs('saves/plots/kernel_size_comparison/')
        
        fig.savefig(f'kernel_size_{r2corr_str}.png')
        print(f'kernel_size_{r2corr_str}.png')



    
if __name__=='__main__':
    main()