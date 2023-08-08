from data.load import load_xr_dataset
import numpy as np
from transforms.grids import get_grid_vars
from constants.paths import OUTPUTS_PATH
import os
import xarray as xr
from data.coords import DEPTHS

class GridLoader:
    def __init__(self,depth = 0,cpuinds = [0]) -> None:
        self.depth = depth
        args = f'--sigma 4 --depth {depth} --mode data --filtering gcm'.split()    
        self.args = args
        self.cpuinds = cpuinds
        self.grids =(None,None)
        root = os.path.join(OUTPUTS_PATH,'filter_weights','grids')
        if not os.path.exists(root):
            os.makedirs(root)
        filename = f'dpth-{depth}-ugrid'
        self.paths = [0,0]
        self.paths[0] = os.path.join(root,filename)
        filename = f'dpth-{depth}-tgrid'
        self.paths[1] = os.path.join(root,filename)
    def load_xr_grids(self,):
        grid,_ = load_xr_dataset(self.args)
        grid = grid.isel(time = 0).sel(depth = [self.depth],method = 'nearest')
        self.grids = get_grid_vars(grid)
    def write2npz(self,):
        for i in range(2):
            gl = self.grids[i]
            data_vars_dict = {key:gl[key].values for key in gl.data_vars.keys()}
            coords_dict = {key:gl[key].values for key in gl.coords}
            data_vars_dict.update(**coords_dict)
            if 'time' in data_vars_dict:
                data_vars_dict.pop('time')
            for cpuind in self.cpuinds:                
                path = self.paths[i] +'-'+ str(cpuind) + '.npz'
                print(path)
                np.savez(path,**data_vars_dict)
    def load_npz_grids(self,):
        grids = [0,0]
        for i in range(2):
            cpuind = self.cpuinds[0]
            path = self.paths[i]+'-'+ str(cpuind) + '.npz'
            agrid = dict(np.load(path))
            coords = {key:val for key,val in agrid.items() if key in 'depth lat lon'.split()}
            datavars = {key:('depth lat lon'.split(),val) for key,val in agrid.items() if key in 'dx dy area wet_mask'.split()}
            grids[i] = xr.Dataset(
                data_vars = datavars,coords = coords
            )
        return grids
    
def main():
    for depth in DEPTHS:
        gl = GridLoader(depth = int(depth),cpuinds=list(range(20)))
        gl.load_xr_grids()
        gl.write2npz()
        grids = gl.load_npz_grids()
        print(grids[0])
        
    


if __name__ == '__main__':
    main()