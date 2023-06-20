import xarray as xr
import matplotlib.pyplot as plt
def main():
    path = '/scratch/zanna/data/cm2.6/surface_1pct_co2.zarr'
    ds = xr.open_zarr(path).isel(time = 300)
    print(ds)
    ds.usurf.plot()
    plt.savefig('usurf.png')


if __name__ == '__main__':
    main()