from data.load import load_filter_weights, load_xr_dataset
from transforms.coarse_graining import GcmFiltering,GreedyCoarseGrain
from transforms.grids import get_grid_vars
from utils.arguments import options
import numpy as np
from utils.xarray_oper import plot_ds
import xarray as xr 
from transforms.multi_gmres import MultiGmres,MultiLinearOps
from transforms.gcm_compression import MultiMatmult2DFilter


class GcmInversion(MultiMatmult2DFilter,MultiLinearOps):
    def __init__(self, sigma, grid, filter_weights, rank=np.inf) -> None:
        super().__init__(sigma, grid, filter_weights, rank)
        self.filtering, self.coarse_grain = GcmFiltering(sigma,grid,),GreedyCoarseGrain(sigma,grid)
    def __call__(self, x, inverse=False, separated=False, special: int = -1):
        # if not inverse and not separated and special < 0:
        #     xrx = self.np2xr(x.copy(),True,fine_grid=True)            
        #     cx = self.coarse_grain(self.filtering(xrx)).fillna(0)
        #     return cx.values
        return super().__call__(x, inverse, separated, special)
    def fit(self,cu:xr.DataArray,maxiter :int = 2,sufficient_decay_limit = np.inf):
        rhs = cu.fillna(0).values.flatten()
        gmres = MultiGmres(self,rhs,maxiter = maxiter,reltol = 1e-15,sufficient_decay_limit=sufficient_decay_limit)
        uopt,_,_ = gmres.solve()
        uopt = uopt.reshape(self.fine_shape)
        uopt = xr.DataArray(
            data = uopt,
            dims = self.grid.dims,
            coords = self.grid.coords
        )
        uopt = xr.where(self.grid.wet_mask == 0,np.nan,uopt)
        return uopt

        


    

def filtering_test():
    sigma = 4
    args = f'--sigma {sigma} --filtering gcm --lsrp 1 --mode data'.split()
    filter_weights = load_filter_weights(args,utgrid='u').load()
    datargs,_ = options(args,key = 'data')
    ds,_ = load_xr_dataset(args)
    ugrid,_ = get_grid_vars(ds.isel(time = 0))
    
    # i,j = 155,305

    # x = filter_weights.filters.isel(lat = i,lon = j).values
    
    
    utrue = ds.u.isel(time = 0).load().rename(
        {f'u{dim}':dim for dim in 'lat lon'.split()}
    ).drop('time')
    cds,_ = load_xr_dataset(args,high_res=False)
    ubar = cds.isel(time = 0,depth = 0).u.drop('time')
    
    
    
    m2df = MultiMatmult2DFilter(sigma,ugrid,filter_weights,)
    ubar1 = m2df(utrue,)
    import matplotlib.pyplot as plt
    fig,axs = plt.subplots(1,3,figsize = (30,10))
    ubar.plot(ax = axs[0])
    ubar1.plot(ax = axs[1])
    err = np.abs(ubar1 - ubar)/np.abs(ubar)
    err.plot(ax = axs[2])
    fig.savefig('filtered.png')
    

def plot_weight_maps():
    import itertools
    args = '--sigma 4 --filtering gcm --lsrp 1 --mode data'.split()
    for ut in 'u t'.split():
        fw = load_filter_weights(args,utgrid = ut).load()
        rank = 8        
        wmaps = {}
        for i,j in itertools.product(range(rank),range(rank)):
            fwij = fw.isel(lat_degree= i,lon_degree = j).drop('lat_degree').drop('lon_degree')
            wmaps[str((i,j))] =  fwij.weight_map
        plot_ds(wmaps,f'{ut}_wet_maps.png',ncols = 2)

    
def gcm_inversion_test():
    args = '--sigma 4 --filtering gcm --lsrp 1 --mode data'.split()
    fw = load_filter_weights(args).load()
    cisel = dict()
    fisel = dict()

    # dx = 200
    # x = 150
    # dy = 200
    # y = 150
    # cisel = dict(lat = slice(y,dy+y),lon= slice(x,dx+x))
    # fisel = dict(lat = slice(y*4,(dy+y)*4),lon = slice(x*4,(x+dx)*4))
    fw = fw.isel(**cisel)
    datargs,_ = options(args,key = 'data')
    ds,_ = load_xr_dataset(args)
    ugrid,_ = get_grid_vars(ds.isel(time = 0))
    ugrid = ugrid.isel(**fisel)
    rank = 4
    gcminv = GcmInversion(datargs.sigma,ugrid,fw,rank = rank)#,rank = 101)
    cds,_ = load_xr_dataset(args,high_res=False)
    ubar = cds.u.isel(time = 0).load().isel(**cisel)
    utrue = ds.u.isel(time = 0).load().rename(
        {f'u{dim}':dim for dim in 'lat lon'.split()}
    ).isel(**fisel)
    uopt = gcminv.fit(ubar,maxiter = 16,sufficient_decay_limit=0.95)#,initial_operator=0)
    
    
    coords = (
        (-60,0,-40,20),
        (0,60,-40,20),
        (-60,0,-150,-90),
        (0,60,-150,-90),
        (-60,0,-90,-30),
        (0,60,-90,-30),
    )
    sel_dicts = [
        dict(lat = slice(c[0],c[1]),lon = slice(c[2],c[3])) for c in coords
    ]
    udict = dict(utrue = utrue,usolv = uopt, uerr = np.log10(np.abs(utrue - uopt)))
    plot_ds(udict,f'gcm_inversion_{rank}.png',ncols = 3)
    for i,sel_dict in enumerate(sel_dicts):
        udict_ = {key:val.sel(**sel_dict) for key,val in udict.items()}
        plot_ds(udict_,f'gcm_inversion_{i}_{rank}.png',ncols = 3)
def gcm_inversion_test_temp():
    args = '--sigma 4 --filtering gcm --lsrp 1 --mode data'.split()
    fw = load_filter_weights(args,utgrid='t').load()
    cisel = dict()
    fisel = dict()
    
    # dx = 200
    # x = 150
    # dy = 200
    # y = 150
    # cisel = dict(lat = slice(y,dy+y),lon= slice(x,dx+x))
    # fisel = dict(lat = slice(y*4,(dy+y)*4),lon = slice(x*4,(x+dx)*4))
    
    fw = fw.isel(**cisel)
    datargs,_ = options(args,key = 'data')
    ds,_ = load_xr_dataset(args)
    _,tgrid = get_grid_vars(ds.isel(time = 0))
    tgrid = tgrid.isel(**fisel)
    rank = 1
    gcminv = GcmInversion(datargs.sigma,tgrid,fw,rank = rank)
    cds,_ = load_xr_dataset(args,high_res=False)
    ubar = cds.temp.isel(time = 0).load()
    ubar = ubar.isel(**cisel)
    utrue = ds.temp.isel(time = 0).load().rename(
        {f't{dim}':dim for dim in 'lat lon'.split()}
    ).isel(**fisel)
    # return
    uopt = gcminv.fit(ubar,maxiter = 16,sufficient_decay_limit=0.98)
    
    coords = (
        (-60,0,-40,20),
        (0,60,-40,20),
        (-60,0,-150,-90),
        (0,60,-150,-90),
        (-60,0,-90,-30),
        (0,60,-90,-30),
    )
    sel_dicts = [
        dict(lat = slice(c[0],c[1]),lon = slice(c[2],c[3])) for c in coords
    ]
    udict = dict(utrue = utrue,usolv = uopt, uerr = np.log10(np.abs(utrue - uopt)))
    plot_ds(udict,f'gcm_inversion_{rank}.png',ncols = 3)
    for i,sel_dict in enumerate(sel_dicts):
        udict_ = {key:val.sel(**sel_dict) for key,val in udict.items()}
        plot_ds(udict_,f'gcm_inversion_{i}_{rank}.png',ncols = 3)
def main():
    gcm_inversion_test_temp()
    
if __name__ == '__main__':
    main()
