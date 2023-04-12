from data.load import get_data
from models.load import load_model
import torch
from torch.nn import functional as F
from torch.nn import Sequential
from torch import nn
import numpy as np
import math
from utils.arguments import options
from constants.paths import OUTPUTS_PATH,ONLINE_MODELS
import os
import matplotlib.pyplot as plt

from utils.xarray import fromtorchdict2tensor


def main():
    from data.load import load_xr_dataset

    args = '--filtering gaussian --num_workers 1 --interior False --mode eval'.split()
    ds,_ = load_xr_dataset(args,high_res = False)

    ds = ds.isel(time = 0,).fillna(0)

    u_scale = 1/0.09439346225350978
    v_scale = 1/0.07252696573672539
    uv = np.stack([np.stack([ds.u.values,ds.v.values],axis = 0)],axis = 0)
    uv[:,0,:,:]= uv[:,0,:,:]*u_scale
    uv[:,1,:,:]= uv[:,1,:,:]*v_scale
    inputs = uv
    test_generator, = get_data(args,half_spread = 10, torch_flag = False, data_loaders = True,groups = ('train',))
    for fields,forcings,forcing_mask,_,forcing_coords in test_generator:  
        fields_tensor = fromtorchdict2tensor(fields).type(torch.float32)
        inputs2 = fields_tensor.numpy().astype(np.float64)
        break
    
    
    args = '--filtering gaussian --num_workers 1 --interior True --domain four_regions --batchnorm 1 1 1 1 1 1 1 0 --widths 2 128 64 32 32 32 32 32 4 --kernels 5 5 3 3 3 3 3 3 --minibatch 4 --mode train'.split()
    args_legacy = args#get_legacy_args(args)
    runargs,_ = options(args,key = "run")

    _,_,net,_,_,_,_,_=load_model(args)
    net.eval()
    with torch.set_grad_enabled(False):
        mean,prec =  net.forward(fields_tensor)
    
    path = os.path.join(ONLINE_MODELS,'cem_20230405.pth')
    print(path)
    
    
if __name__ == '__main__':
    main()