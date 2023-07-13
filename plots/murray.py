import os
from typing import Tuple, Union
from data.paths import get_high_res_grid_location
from models.load import get_statedict
from plots.metrics_ import metrics_dataset
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
from plots.metrics_ import metrics_dataset
from constants.paths import EVALS
import xarray as xr
from models.load import get_statedict
import numpy as np
import matplotlib




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
    if coarse_graining_factor == 1:
        return grid
    wetmask = xr.where(grid == 0,0,1)
    
    
    coarse_kwargs = dict(lat = coarse_graining_factor,lon = coarse_graining_factor,boundary = 'trim')
    wetmask = wetmask.coarsen(**coarse_kwargs).sum().compute()
    grid = grid.coarsen(**coarse_kwargs).sum().compute()/wetmask
    grid = xr.where(np.isnan(grid),0,grid)
    return grid

class Grid:
    def __init__(self,sigma:int = 4) -> None:
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
class MurrayPlotter:
    def __init__(self,sigma:Union[int,Tuple[int]] = (4,),\
                nrows:int = 1,ncols :int = 1,figsize = (10,4.5),\
                colorbarxmarg = 0.10,
                leftxmarg = 0.03,
                interxmarg = 0.01,
                ymarg = 0.05,
                colorbarwidth = None,
                ):
        if isinstance(sigma,int):
            sigma = (sigma,)
        self.grids = {
            sigma_ : Grid(sigma=sigma_) for sigma_ in sigma
        }
        if colorbarwidth is None:
            self.colorbarwidth = colorbarxmarg/6
        else:
            self.colorbarwidth = colorbarwidth
        self.colorbarxmarg = colorbarxmarg
        self.leftxmarg = leftxmarg
        self.interxmarg = interxmarg
        self.ymarg = ymarg
        
        # ax = plt.axes(projection = ccrs.PlateCarree())
        # fig, axs = plt.subplots(nrows=nrows,ncols=ncols,
        #                 subplot_kw={'projection': ccrs.PlateCarree()},
        #                 figsize=figsize)
        
        fig = plt.figure(figsize = figsize)
        # udmarg = 0.
        # lrmarg = (0.05,0.07)
        # fig.subplots_adjust(left=lrmarg[0], right=1-lrmarg[1], top=1 - udmarg, bottom=udmarg)
        
        self.grid_lines = dict(
            crs=ccrs.PlateCarree(), draw_labels=True,
                        linewidth=1, color='gray', alpha=0, linestyle='--'
        )
        self.nrows,self.ncols = nrows,ncols
        # self.fig,self.axs =  fig,axs
        self.fig =  fig
        self.plotted = 0
    def get_ax(self,i,j,colorbar = None,ax_dims = None,**kwargs):
        if ax_dims is not None:
            return self.fig.add_axes(ax_dims,projection = ccrs.PlateCarree())
        # southwestern edge, width,height
        colorbarxmarg = self.colorbarxmarg
        leftxmarg = self.leftxmarg
        interxmarg = self.interxmarg 
        ymarg = self.ymarg
        
        width_multiplier = 1 - leftxmarg - colorbarxmarg
        
        dy = 1/self.nrows
        dx = 1/self.ncols        
        i = self.nrows - i - 1
        
        if colorbar is not None:
            j = self.nrows
            xloc = 1 - colorbarxmarg + interxmarg*width_multiplier #+ 0.01
            xw = self.colorbarwidth #colorbarxmarg/6
            yloc = ymarg + (colorbar[1]  - colorbar[0] - 1)/colorbar[1] 
            yw = 1/colorbar[1] - 2*ymarg
            return self.fig.add_axes([xloc,yloc,xw,yw])
        
        xloc = kwargs.get('xloc',leftxmarg + (dx*j + interxmarg)*width_multiplier)
        xw = kwargs.get('xw',(dx - 2*interxmarg)*width_multiplier)
        yloc = dy*i + ymarg
        yw = dy - 2*ymarg
        return self.fig.add_axes([xloc,yloc,xw,yw],projection = ccrs.PlateCarree())

    def get_grid(self,sigma:int):
        if sigma == 0:
            sgms = list(self.grids.keys())
            sigma = sgms[0]
        if sigma in self.grids:
            return self.grids[sigma].grid
        raise Exception
    def plot(self,irow:int,icol:int,xrvar,\
        vmin = 0,vmax = 1,title:str = None,coord_sel:dict = {},\
        sigma :int = 0,projection_flag:bool = True,colorbar = None,\
        cmap = matplotlib.cm.viridis,grid_lines = {},\
        cbar_label:str = None,\
        set_bad_alpha:float = 1.,\
        no_colorbar:bool = False,**kwargs):
        grid = self.get_grid(sigma)
        # ax = plt.subplot(**kwargs)
        # plt.margins(0.1)
        grid_lines = {key:grid_lines[key] if key in grid_lines else val for key,val in self.grid_lines.items()}
        ax = self.get_ax(irow,icol,**kwargs)
        if grid_lines is not None:
            gl = ax.gridlines(**grid_lines)
            gl.top_labels = False
            gl.right_labels = False
            gl.left_labels = True if icol == 0 else False
            gl.yrotation = False
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
        
        if projection_flag:
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
        if bool(coord_sel):
            grid = grid.sel(**coord_sel)
            xrvar = xrvar.sel(**coord_sel)
            grid,xrvar = constrain_by_index(grid,xrvar,'lon')
            grid,xrvar = constrain_by_index(grid,xrvar,'lat')
        kwargs = dict(vmin = vmin,vmax = vmax)
        keys = list(kwargs.keys())
        for key in keys:
            if kwargs[key] is None:
                kwargs.pop(key)
        cmap.set_bad('gray',set_bad_alpha)
        cs = ax.pcolormesh(grid.geolon_t.values,grid.geolat_t.values,xrvar.values,cmap = cmap,**kwargs)
        if title is not None:
            ax.set_title(title)
        if no_colorbar:
            return ax,cs
        if colorbar is None:
            if irow == self.nrows -1 and icol == self.ncols -1:
                cbar_ax = self.get_ax(0,0,colorbar = colorbar)
                cbar=self.fig.colorbar(cs, cax=cbar_ax,orientation='vertical')
                if cbar_label is not None:
                    cbar.set_label(cbar_label)
        else:
            if icol == self.ncols - 1:
                cbar_ax = self.get_ax(0,0,colorbar = colorbar)
                cbar=self.fig.colorbar(cs, cax=cbar_ax,orientation='vertical')
                if cbar_label is not None:
                    cbar.set_label(cbar_label)
        return ax
            
    def save(self,path,transparent = True):
        plt.savefig(path, transparent=transparent)
        plt.close()
        
        
        
        

    
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
    
class MurrayPlotter2:
    def __init__(self,sigma:Tuple[int] = (4,8,12,16),):
        self.grids = {
            sigma_ : Grid(sigma=sigma_) for sigma_ in sigma
        }
        
        self.grid_lines = dict(
            crs=ccrs.PlateCarree(), draw_labels=True,
                        linewidth=1, color='gray', alpha=0, linestyle='--'
        )
    def get_ax(self,dims,fig,projection_flag:bool = True):
        # dims = self.spa.get_ax_dims(i,j)
        if projection_flag:
            return fig.add_axes(dims,projection = ccrs.PlateCarree())
        else:
            return fig.add_axes(dims)

    def get_grid(self,sigma:int):
        if sigma == 0:
            sgms = list(self.grids.keys())
            sigma = sgms[0]
        if sigma in self.grids:
            return self.grids[sigma].grid
        raise Exception
    def plot(self,xrvar,ax,\
            vmin = 0,vmax = 1,title:str = None,coord_sel:dict = {},\
            sigma:int = 0,\
            projection_flag:bool = True,\
            cmap = matplotlib.cm.viridis,grid_lines = {},\
            set_bad_alpha:float = 1.):
        grid = self.get_grid(sigma)
        grid_lines = {key:grid_lines[key] if key in grid_lines else val for key,val in self.grid_lines.items()}
        if grid_lines is not None:
            gl = ax.gridlines(**grid_lines)
            gl.top_labels = False
            gl.right_labels = False
            gl.left_labels = True #if icol == 0 else False
            gl.yrotation = False
            gl.xformatter = LONGITUDE_FORMATTER
            gl.yformatter = LATITUDE_FORMATTER
        
        if projection_flag:
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
        if bool(coord_sel):
            grid = grid.sel(**coord_sel)
            xrvar = xrvar.sel(**coord_sel)
            grid,xrvar = constrain_by_index(grid,xrvar,'lon')
            grid,xrvar = constrain_by_index(grid,xrvar,'lat')
        kwargs = dict(vmin = vmin,vmax = vmax)
        keys = list(kwargs.keys())
        for key in keys:
            if kwargs[key] is None:
                kwargs.pop(key)
        cmap.set_bad('gray',set_bad_alpha)
        cs = ax.pcolormesh(grid.geolon_t.values,grid.geolat_t.values,xrvar.values,cmap = cmap,**kwargs)
        if title is not None:
            ax.set_title(title)
        return cs
    def put_colorbar(self,cs,cbar_ax,fig,cbar_label:str = None,orientation='vertical'):
        cbar=fig.colorbar(cs, cax=cbar_ax,orientation=orientation)
        if cbar_label is not None:
            cbar.set_label(cbar_label)
        return cbar


class MurrayWithSubplots(SubplotAxes):
    def __init__(self, nrows, ncols, xmargs=(0.05, 0.03, 0), ymargs=(0.05, 0.01, 0.01), sizes=None,figsize = (10.5,8)):
        super().__init__(nrows, ncols, xmargs, ymargs, sizes)
        self.murrayplt = MurrayPlotter2()
        self.fig = plt.figure(figsize = figsize)
    def plot(self,irow,icol,xrvar,**kwargs):
        dims = self.get_ax_dims(irow,icol)        
        # dims = [0, 0.05, 0.9, 0.9 ]
        ax = self.murrayplt.get_ax(dims,self.fig,projection_flag=True)
        cs = self.murrayplt.plot(xrvar,ax,**kwargs)
        return dims,ax,cs
    def colorbar_dim_correction(self,dims:list):
        xloc,y0,dx,dy = dims
        y1 = y0 + dy
        cbar_ybeta = (0.05,0.05)
        y0 = y0 + dy*cbar_ybeta[0]
        y1 = y1 - dy*cbar_ybeta[1]
        dy = y1 - y0
        return (xloc,y0,dx,dy)
    def plot_colorbar(self,irow,icol,colorbar_cs,cbar_label:str = None,orientation='vertical',):
        dims = self.get_ax_dims(irow,icol)        
        # dims = [0.95, 0.05, 0.05,0.9 ]
        dims = self.colorbar_dim_correction(dims)
        cbar_ax = self.murrayplt.get_ax(dims,self.fig,projection_flag=False)
        cbar = self.murrayplt.put_colorbar(colorbar_cs,cbar_ax,self.fig,cbar_label = cbar_label,orientation = orientation)
        return cbar
    def save(self,path,transparent = True):
        plt.savefig(path, transparent=transparent)
        plt.close()