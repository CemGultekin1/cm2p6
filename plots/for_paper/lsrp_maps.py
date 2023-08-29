import itertools
import os
from data.paths import get_high_res_grid_location
from models.load import get_statedict
from plots.metrics_ import metrics_dataset
from constants.paths import EVALS
from plots.murray import MurrayPlotter
from transforms.grids import fix_grid
import xarray as xr
import numpy as np
from utils.slurm import read_args
import cartopy.crs as ccrs
import cartopy as cart
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import os
import matplotlib
import matplotlib.pyplot as plt
from plots.metrics_ import metrics_dataset
from constants.paths import EVALS
import xarray as xr
from models.load import get_statedict
import numpy as np
from plots.for_paper.saliency import SubplotAxes
def load_r2map():
    args = '--model lsrp:0 --sigma 4'.split()
    _,_,_,modelid = get_statedict(args)
    path =  os.path.join(EVALS,modelid + '_.nc')
    assert os.path.exists(path)
    ds = xr.open_dataset(path).isel(depth = 0,co2 = 0)
    return metrics_dataset(ds,dim = [])

def load_grid(coarse_graining_factor:int):
    grid = xr.open_dataset(get_high_res_grid_location())
    names = list(grid.data_vars.keys())
    names.pop(names.index('geolon_t'))
    names.pop(names.index('geolat_t'))
    grid = grid.drop(names)
    grid = grid.rename({'xt_ocean':'lon','yt_ocean':'lat'})
    names = list(grid.coords.keys())
    names.pop(names.index('lat'))
    names.pop(names.index('lon'))
    grid = grid.drop(names)
    wetmask = xr.where(grid == 0,0,1)
    
    
    coarse_kwargs = dict(lat = coarse_graining_factor,lon = coarse_graining_factor,boundary = 'trim')
    wetmask = wetmask.coarsen(**coarse_kwargs).sum().compute()
    grid = grid.coarsen(**coarse_kwargs).sum().compute()/wetmask
    grid = xr.where(np.isnan(grid),0,grid)
    return grid
def plot(xrvar,kwargs,target_folder,target_file,grid):
    # ax = plt.subplot(**kwargs)
    plt.margins(0.1)
    fig=plt.figure()
    ax = plt.axes(projection = ccrs.PlateCarree())
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                    linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.yrotation = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    ax.add_feature(cart.feature.LAND, zorder=100, edgecolor='k',facecolor='gray')#,alpha=.4)
    
    
    grid = fix_grid(grid,(xrvar.lat.values,xrvar.lon.values))
    def constrain_by_index(lgrid,sgrid,dim):
        ndim = len(sgrid[dim])
        lndim = len(lgrid[dim])
        if ndim == lndim:
            return lgrid,sgrid
        if ndim > lndim:
            sgrid,lgrid =  constrain_by_index(sgrid,lgrid,dim)
            return lgrid,sgrid
        xleft = np.maximum((lndim - ndim)//2,0)
        xright = np.maximum(lndim - ndim - xleft,0)
        
        return lgrid.isel({dim : slice(xleft,lndim-xright )}),sgrid
    
    grid,xrvar = constrain_by_index(grid,xrvar,'lon')
    grid,xrvar = constrain_by_index(grid,xrvar,'lat')
    im = ax.pcolormesh(grid.geolon_t.values,grid.geolat_t.values,xrvar.values,vmin = 0,vmax = 1,)#,edgecolors = None)
    cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
    
    plt.colorbar(im, cax=cax,)

    ax.set_title('')
    plt.savefig(os.path.join(target_folder,target_file), transparent=True)
    plt.close()
def main():
        
    lsrp = load_r2map()
    
    target_folder = 'paper_images/r2maps'
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    mp = MurrayPlotter(sigma=4,nrows= 1, ncols = 2,figsize = (10,3),)#leftxmarg=0.03,interxmarg=0.025,ymarg=0.05)
    ax_sizer = SubplotAxes(1,3,xmargs = (0.05,0.02,0.05),ymargs = (0.01,0.01,0.01),sizes = ((1,),(30,30,1)))
    kwargs = dict(
        vmin = 0,
        vmax = 1,
        cbar_label = None,
        set_bad_alpha = 0.,
        colorbar = None,
        no_colorbar=True,
        grid_lines = {'alpha' : 0.5,'linewidth': 1.5},
        cmap = matplotlib.cm.magma
    )
    
    titles = dict(
        Su = '(a) LSRP: $R^2_u$',
        Stemp = '(b) LSRP: $R^2_T$'
    )
    path = os.path.join(target_folder,'lsrp.png')
    for i,(svar,rc) in enumerate(itertools.product('Su Stemp'.split(),'r2'.split())):
        ax_dims = ax_sizer.get_ax_dims(0,i)
        print(ax_dims)
        ax,cs = mp.plot(0,i,lsrp[f'{svar}_{rc}'],title =  titles[svar],ax_dims = ax_dims,**kwargs)
        # mp.plot(0,i,lsrp[f'{svar}_{rc}'],title =  titles[svar],**kwargs)
    ax_dims = ax_sizer.get_ax_dims(0,2)    
    ax_dims[1] = 0.18
    ax_dims[3] = 0.64
    print(ax_dims)
    cbar_ax = mp.fig.add_axes(ax_dims)
    cbar=mp.fig.colorbar(cs, cax=cbar_ax,orientation='vertical')
    
    # mp.save(path.replace('.png','_.png'),transparent=False)
    mp.save(path,transparent=True)

if __name__ == '__main__':
    main()