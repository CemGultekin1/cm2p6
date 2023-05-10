#!/bin/env python
from collections import OrderedDict
import torch
from torch.nn import  Parameter
from torch import nn
import numpy as np
import math
from torch.nn.functional import softplus

# GPU setup
args_no_cuda = False #True when manually turn off cuda
use_cuda = not args_no_cuda and torch.cuda.is_available()
if use_cuda:
    print('device for inference on',torch.cuda.device_count(),'GPU(s)')
else:
    print('device for inference on CPU')


cem_testing_flag = False
if cem_testing_flag:
    import os
    from constants.paths import ONLINE_MODELS
    fn = f'cem_20230418.pth'
    path = os.path.join(ONLINE_MODELS,fn)
    nn_load_main_file = path
else:
    nn_load_main_file= '/scratch/cimes/cz3321/MOM6/MOM6-examples/src/MOM6/config_src/external/ML_Forpy/Forpy_CNN_GZ21/global_model.pt'

nn_load_name = 'gaussian_four_regions'

u_scale = 1/0.1 #0.09439346225350978
v_scale = 1/0.1 #0.07252696573672539
Su_scale = 1e-7 #4.9041400042653195e-08
Sv_scale = 1e-7 #4.8550991806254025e-08

    

class ConvLayer(nn.Sequential):
    def __init__(self,width0:int,width1:int,kernel0:int,kernel1:int = None,batchnorm:bool = False,nnlnr:bool = True):
        if kernel1 is None:
            kernel1 = kernel0
        d = []
        d.append(('conv',nn.Conv2d(width0,width1,(kernel0,kernel1))))
        if batchnorm and nnlnr:
            assert nnlnr
            d.append(('bnorm', nn.BatchNorm2d(width1)))
        if nnlnr:
            d.append(('nnlnr',nn.ReLU(inplace = True)))
        super().__init__(OrderedDict(d))
        
class SoftPlusLayer_(nn.Module):
    def __init__(self,min_value = 0) -> None:
        super().__init__()
        self._min_value = Parameter(torch.tensor(min_value))
    @property
    def min_value(self,):
        return softplus(self._min_value)
    def forward(self, x_):
        x = torch.clone(x_)
        x0,x1 = torch.split(x,x.shape[1]//2,dim=1)
        x1 = softplus(x1) + self.min_value
        return x0,x1
    def __repr__(self) -> str:
        return f'SoftPlusLayer({self.min_value.item()})'
    
class PartialSoftPlusLayer(nn.Module):
    def forward(self, x_):
        x = torch.clone(x_)
        x0,x1 = torch.split(x,x.shape[1]//2,dim=1)
        x1 = softplus(x1)
        return x0,x1
    def __repr__(self) -> str:
        return self.__class__.__name__

class CNN(nn.Sequential):
    def __init__(self,widths = None,kernels = None,batchnorm = None,seed = None,**kwargs):
        d = []
        zipwidths = zip(widths[:-1],widths[1:])
        nlayers = len(kernels)
        torch.manual_seed(seed)
        for i,((w0,w1),k,b,) in enumerate(zip(zipwidths,kernels,batchnorm)):
            d.append(
                (f'layer-{i}',ConvLayer(w0,w1,k,k,b,nnlnr = i < nlayers - 1))
            )
        d.append(
            (f'layer-{i+1}',PartialSoftPlusLayer())
        )
        super().__init__(OrderedDict(d))
        spread = 0
        for k in kernels:
            spread += (k-1)/2
        self.spread = int(spread)


statedict = torch.load(nn_load_main_file)
modeldict,statedict = statedict[nn_load_name]
cnn=CNN(**modeldict)
cnn.load_state_dict(statedict)
cnn.eval()



def MOM6_testNN(uv,pe,pe_num,index):
    global cnn,gpu_id,u_scale,v_scale,Su_scale,Sv_scale
    # start_time = time.time()
    # print('PE number is',pe_num)
    # print('PE is',pe)
    # print(u.shape,v.shape)
    #normalize the input by training scaling
    u= uv[0,:,:,:]*u_scale
    v= uv[1,:,:,:]*v_scale
    x = np.array([np.squeeze(u),np.squeeze(v)])
    if x.ndim==3:
        x = x[:,:,:,np.newaxis]
    x = x.astype(np.float32)
    print(x.shape)
    x = x.transpose((3,0,1,2)) # new the shape is (nk,2,ni,nj)
    x = torch.from_numpy(x) # quite faster than x = torch.tensor(x)
    if use_cuda:
        if not next(cnn.parameters()).is_cuda:
            gpu_id = int(pe/math.ceil(pe_num/torch.cuda.device_count()))
            print('GPU id is:',gpu_id)
            cnn = cnn.cuda(gpu_id)
        x = x.cuda(gpu_id)
    with torch.no_grad():
        # start_time = time.time()
        outs = cnn.forward(x)
        if isinstance(outs,tuple):
            mean,precision = outs
        else:
            mean,precision = torch.split(outs,2,dim = 1)
        # end_time = time.time()
    if use_cuda:
        mean = mean.to('cpu')
        precision = precision.to('cpu')
    mean = mean.numpy().astype(np.float64)
    std = np.sqrt(1/precision.numpy().astype(np.float64))
    # At this point, python out shape is (nk,4,ni,nj)
    # Comment-out is tranferring arraies into F order
    # convert out to (ni,nj,nk)
    mean = mean.transpose((1,2,3,0)) # new the shape is (4,ni,nj,nk)
    std = std.transpose((1,2,3,0))
    dim = np.shape(mean)
    Sxy = np.zeros((6,dim[1],dim[2],dim[3])) # the shape is (2,ni,nj,nk)
    epsilon_x = np.random.normal(0, 1, size=(dim[1],dim[2]))
    epsilon_x = np.dstack([epsilon_x]*dim[3])
    epsilon_y = np.random.normal(0, 1, size=(dim[1],dim[2]))
    epsilon_y = np.dstack([epsilon_y]*dim[3])				  
    Sxy[0,:,:,:] = (mean[0,:,:,:] )*Su_scale
    Sxy[1,:,:,:] = (mean[1,:,:,:] )*Sv_scale
    Sxy[2,:,:,:] = mean[0,:,:,:]*Su_scale
    Sxy[3,:,:,:] = mean[1,:,:,:]*Sv_scale
    #    Sxy[4,:,:,:] = std[0,:,:,:]*Su_scale
    #    Sxy[5,:,:,:] = std[1,:,:,:]*Sv_scale
    """
    np.savetxt('Sx_mean_cem.txt',Sxy[2,:,:,0])
    np.savetxt('Sy_mean_cem.txt',Sxy[3,:,:,0])
    np.savetxt('Sx_std_cem.txt',Sxy[4,:,:,0])
    np.savetxt('Sy_std_cem.txt',Sxy[5,:,:,0])
    np.savetxt('WH_u_cem.txt',uv[0,:,:,0])
    np.savetxt('WH_v_cem.txt',uv[1,:,:,0])
    """
    return Sxy 
