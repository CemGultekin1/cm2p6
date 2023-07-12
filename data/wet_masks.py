


import itertools
from data.load import get_data, load_xr_dataset,get_wet_mask_location
import xarray as xr
from data.coords import SIGMAS,DEPTHS
from utils.xarray import plot_ds

def generate_save_wet_masks():
    for sigma,depth in itertools.product(SIGMAS,DEPTHS[:2]):
        datargs = f'--sigma {sigma} --depth {depth} --mode data --minibatch 1 --filtering gaussian'.split()
        print(' '.join(datargs))
        generator,= get_data(datargs,half_spread = 0, torch_flag = False, data_loaders = False,groups = ('all',))
        wet_mask_address = get_wet_mask_location(datargs)
        for i,masks in enumerate(generator.iterate_wet_masks_through_depths()):
            masks = masks.chunk({k:len(masks[k]) for k in list(masks.coords)})            
            if i==0:
                masks.to_zarr(wet_mask_address,mode = 'w')
            else:
                masks.to_zarr(wet_mask_address,mode = 'a',append_dim = 'depth')
            print(f'masks.depth = {masks.depth.values.item()}')
        masks = xr.open_zarr(wet_mask_address)
        plot_ds(masks,f'masks-{sigma}-{depth}.png',ncols = 2)
    
def main():
    for sigma,depth in itertools.product(SIGMAS,DEPTHS):
        datargs = f'--sigma {sigma} --depth {depth} --mode data --minibatch 1 --filtering gcm --co2 True'.split()
        ds,_= load_xr_dataset(datargs,high_res = False)
        plot_ds(
            dict(
                interior_wet_mask = ds.interior_wet_mask,
                wet_density = ds.wet_density
            ),
            f'wet_density_{sigma}_{depth}.png',
            ncols = 2
        )

if __name__=='__main__':
    main()