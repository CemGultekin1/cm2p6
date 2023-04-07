from data.coords import REGIONS
from data.load import get_data, load_xr_dataset
from utils.xarray import plot_ds


def main():
    from utils.slurm import read_args
    args = read_args(1,)
    from params import replace_params
    args = replace_params(args,'num_workers','1','minibatch','4','domain','four_regions','temperature','True','latitude','True')
    training_generator,=get_data(args,half_spread = 10,torch_flag = True,data_loaders = True,groups = ('train',))
    for fields,forcings,masks in training_generator:
        print(fields.shape,forcings.shape,masks.shape)
        return


if __name__ == '__main__':
    main()