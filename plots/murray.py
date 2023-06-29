import os
from data.paths import get_high_res_grid_location
from models.load import get_statedict
from plots.metrics import metrics_dataset
from constants.paths import EVALS
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
from plots.metrics import metrics_dataset
from constants.paths import EVALS
import xarray as xr
from models.load import get_statedict
import numpy as np




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

    
class MurrayPlotter:
    def __init__(self,sigma:int = 4,nrows:int = 1,ncols :int = 1,figsize = (10,4.5)):
        grid = load_grid(sigma)
        def jump_continuation(grid,key,dim,):
            val = grid[key]
            
            nnzpts = xr.where(val == 0,0,1)
            avgval = val.sum(dim = dim)/nnzpts.sum(dim = dim)
            val = xr.where(nnzpts == 0,avgval,val)
            grid[key] = (val.dims,val.values)
            return grid
        grid = jump_continuation(grid,'geolat_t','lon')
        grid = jump_continuation(grid,'geolon_t','lat')
        self.grid = grid        
        
        # ax = plt.axes(projection = ccrs.PlateCarree())
        # fig, axs = plt.subplots(nrows=nrows,ncols=ncols,
        #                 subplot_kw={'projection': ccrs.PlateCarree()},
        #                 figsize=figsize)
        
        fig = plt.figure(figsize = figsize)
        # udmarg = 0.
        # lrmarg = (0.05,0.07)
        # fig.subplots_adjust(left=lrmarg[0], right=1-lrmarg[1], top=1 - udmarg, bottom=udmarg)
        
        
        self.nrows,self.ncols = nrows,ncols
        # self.fig,self.axs =  fig,axs
        self.fig =  fig
        self.plotted = 0
    def get_ax(self,i,j,colorbar:bool = False):
        
        # southwestern edge, width,height
        colorbarxmarg = 0.08
        
        leftxmarg = 0.03
        interxmarg = 0.01
        width_multiplier = 1 - leftxmarg - colorbarxmarg
        ymarg = 0.05
        dy = 1/self.ncols
        dx = 1/self.nrows        
        i = self.ncols - i - 1
        
        if colorbar:
            j = self.ncols
            xloc = 1 - colorbarxmarg + interxmarg*width_multiplier
            xw = colorbarxmarg/4
            yloc = ymarg
            yw = 1 - 2*ymarg
            return self.fig.add_axes([xloc,yloc,xw,yw])
        
        xloc = leftxmarg + (dx*j + interxmarg)*width_multiplier
        xw = (dx - 2*interxmarg)*width_multiplier
        yloc = dy*i + ymarg
        yw = dy - 2*ymarg
        return self.fig.add_axes([xloc,yloc,xw,yw],projection = ccrs.PlateCarree())
    def plot(self,irow:int,icol:int,xrvar,vmin = 0,vmax = 1,title:str = None):
        grid = self.grid.copy()
        # ax = plt.subplot(**kwargs)
        # plt.margins(0.1)
        ax = self.get_ax(irow,icol)
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                        linewidth=2, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.left_labels = True if icol == 0 else False
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
        cs = ax.pcolormesh(grid.geolon_t.values,grid.geolat_t.values,xrvar.values,vmin = vmin,vmax = vmax,)#,edgecolors = None)
        # plt.colorbar(im, cax=cax,)
        if title is not None:
            ax.set_title(title)
        if irow == self.nrows -1 and icol == self.ncols -1:
            self.fig.subplots_adjust(bottom=0.05, top=0.95, left=0.05, right=0.8,
                    wspace=0.1, hspace=0.5)

            # Add a colorbar axis at the bottom of the graph
            # cbar_ax = self.fig.add_axes([0.2, 0.15, 0.6, 0.02])
            cbar_ax = self.get_ax(0,0,colorbar = True)
            cbar=self.fig.colorbar(cs, cax=cbar_ax,orientation='vertical')
            
    def save(self,path):
        plt.savefig(path, transparent=True)
        plt.close()