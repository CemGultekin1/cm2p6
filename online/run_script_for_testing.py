#!/bin/env python

import torch
from torch.nn import functional as F
from torch import nn
import numpy as np
import math
import time

# GPU setup
args_no_cuda = False #True when manually turn off cuda
use_cuda = not args_no_cuda and torch.cuda.is_available()
if use_cuda:
    print('device for inference on',torch.cuda.device_count(),'GPU(s)')
else:
    print('device for inference on CPU')

nn_load_file='/scratch/cimes/cz3321/MOM6/MOM6-examples/src/MOM6/config_src/external/ML_Forpy/Forpy_CNN_GZ21/global_model.pt'

kernels=[5, 5, 3, 3, 3, 3, 3, 3]
widths=[2,128, 64, 32, 32, 32, 32, 32, 4]
batchnorm = [1]*7 + [0]


u_scale = 1/0.09439346225350978
v_scale = 1/0.07252696573672539
Su_scale = 4.9041400042653195e-08
Sv_scale = 4.8550991806254025e-08
class Layer:
    def __init__(self,nn_layers:list,device) -> None:
        self.device = device
        self.nn_layers =nn_layers
        self.section = []
    def add(self,nn_obj):
        self.section.append(len(self.nn_layers))
        self.nn_layers.append(nn_obj.to(self.device))
    def __call__(self,x):
        for j in self.section:
            x = self.nn_layers[j](x)
        return x

class CNN_Layer(Layer):
    def __init__(self,nn_layers:list,device,widthin,widthout,kernel,batchnorm,nnlnr) -> None:
        super().__init__(nn_layers,device)
        self.add(nn.Conv2d(widthin,widthout,kernel))
        if batchnorm:
            self.add(nn.BatchNorm2d(widthout))
        if nnlnr:
            self.add(nn.ReLU(inplace = True))
class Softmax_Layer(Layer):
    def __init__(self,nn_layers:list,device,split) -> None:
        super().__init__(nn_layers,device)
        self.add(nn.Softplus())
        self.split = split
    def __call__(self, x):
        if self.split>1:
            xs = list(torch.split(x,x.shape[1]//self.split,dim=1))
            xs[-1] = super().__call__(xs[-1])
            return tuple(xs)
        return super().__call__(x)
        

class Sequential(Layer):
    def __init__(self,nn_layers,device,widths,kernels,batchnorm,softmax_layer = False,split = 1):
        super().__init__(nn_layers,device)
        self.sections = []
        spread = 0
        self.nlayers = len(kernels)
        for i in range(self.nlayers):
            spread+=kernels[i]-1
        self.spread = spread//2
        for i in range(self.nlayers):
            self.sections.append(CNN_Layer(nn_layers,device,widths[i],widths[i+1],kernels[i],batchnorm[i], i < self.nlayers - 1))
        if softmax_layer:
            self.sections.append(Softmax_Layer(nn_layers,device,split))
    def __call__(self, x):
        for lyr in self.sections:
            x = lyr.__call__(x)
        return x

class CNN(nn.Module):
    def __init__(self,cuda_flag = False,widths = None,kernels = None,batchnorm = None,seed = 0,**kwargs):
        super(CNN, self).__init__()
        device = "cpu" if not cuda_flag else "gpu:0"
        self.device = device
        torch.manual_seed(seed)
        self.nn_layers = nn.ModuleList()        
        self.sequence = \
            Sequential(self.nn_layers,device, widths,kernels,batchnorm,softmax_layer=True,split = 2)
        spread = 0
        for k in kernels:
            spread += (k-1)/2
        self.spread = int(spread)
    def forward(self,x):
        x1 = x.to(self.device)
        x1 = self.sequence(x1)
        return torch.cat(x1,axis = 1)
    

nn=CNN(cuda_flag=False,widths = widths,kernels=kernels,batchnorm=batchnorm)

from utils.paths import ONLINE_MODELS
import os
filename =os.listdir(ONLINE_MODELS)[-1]
path = os.path.join(ONLINE_MODELS,filename)
statedict = torch.load(path)
modelid,statedict = statedict['gaussian_four_regions']
nn.load_state_dict(statedict)

nn.eval()
# raise Exception

# example_forward_input = torch.ones((1,2,42,40))
# second_example_input = torch.ones((4,2,42,40))

# module = torch.jit.trace(nn, example_forward_input)

# onn=nn(second_example_input).detach().numpy()
# jittrace=module(second_example_input).detach().numpy()

# np.savetxt('Sx_mean_up_nn1.txt',onn[0,0,:,:])
# np.savetxt('Sx_mean_up_md1.txt',jittrace[0,0,:,:])
# np.savetxt('Sx_std_up_nn1.txt',onn[0,2,:,:])
# np.savetxt('Sx_std_up_md1.txt',jittrace[0,2,:,:])


def MOM6_testNN(uv,pe,pe_num,index):
   global nn,gpu_id,u_scale,v_scale,Su_scale,Sv_scale
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
   x = x.transpose((3,0,1,2)) # new the shape is (nk,2,ni,nj)
   x = torch.from_numpy(x) # quite faster than x = torch.tensor(x)
   if use_cuda:
       if not next(nn.parameters()).is_cuda:
          gpu_id = int(pe/math.ceil(pe_num/torch.cuda.device_count()))
          print('GPU id is:',gpu_id)
          nn = nn.cuda(gpu_id)
       x = x.cuda(gpu_id)
   with torch.no_grad():
       # start_time = time.time()
       out = nn.forward(x)
       # end_time = time.time()
   if use_cuda:
       out = out.to('cpu')
   out = out.numpy().astype(np.float64)
   # At this point, python out shape is (nk,4,ni,nj)
   # Comment-out is tranferring arraies into F order
   """
   print(out.shape)
   dim = np.shape(out)
   out = out.flatten(order='F')
   out = out.reshape(dim[0],dim[1],dim[2],dim[3], order='F')
   """
   # convert out to (ni,nj,nk)
   out = out.transpose((1,2,3,0)) # new the shape is (4,ni,nj,nk)
   dim = np.shape(out)
   # print(dim)
   Sxy = np.zeros((6,dim[1],dim[2],dim[3])) # the shape is (2,ni,nj,nk)
   epsilon_x = np.random.normal(0, 1, size=(dim[1],dim[2]))
   epsilon_x = np.dstack([epsilon_x]*dim[3])
   epsilon_y = np.random.normal(0, 1, size=(dim[1],dim[2]))
   epsilon_y = np.dstack([epsilon_y]*dim[3])
   # if pe==0:
   #   print(scaling)
   """
   # mean output
   Sxy[0,:,:,:] = (out[0,:,:,:])*Su_scale
   Sxy[1,:,:,:] = (out[1,:,:,:])*Sv_scale
   # std output
   Sxy[0,:,:,:] = (epsilon_x/out[2,:,:,:])*Su_scale
   Sxy[1,:,:,:] = (epsilon_y/out[3,:,:,:])*Sv_scale
   """
   # full output
   Sxy[0,:,:,:] = (out[0,:,:,:] + epsilon_x*np.sqrt(1/out[2,:,:,:]))*Su_scale
   Sxy[1,:,:,:] = (out[1,:,:,:] + epsilon_y*np.sqrt(1/out[3,:,:,:]))*Sv_scale
#    Sxy[0,:,:,:] = out[0,:,:,:]*Su_scale
#    Sxy[1,:,:,:] = out[1,:,:,:]*Sv_scale
   Sxy[2,:,:,:] = out[0,:,:,:]*Su_scale
   Sxy[3,:,:,:] = out[1,:,:,:]*Sv_scale
   Sxy[4,:,:,:] = np.sqrt(1/out[2,:,:,:])*Su_scale
   Sxy[5,:,:,:] = np.sqrt(1/out[3,:,:,:])*Sv_scale
   """
   # scaling the parameters for upper and lower layers
   Sxy[:,:,:,0]=Sxy[:,:,:,0]*0.8
   Sxy[:,:,:,1]=Sxy[:,:,:,1]*1.5
   """

#    np.savetxt('u.txt',uv[0,:,:,0])
#    np.savetxt('v.txt',uv[1,:,:,0])
#    np.savetxt('Sx_mean.txt',Sxy[2,:,:,0])
#    np.savetxt('Sx_std.txt',Sxy[4,:,:,0])

   # end_time = time.time()
   # print("--- %s seconds for CNN ---" % (end_time - start_time))
   # print(nn)
   # print(Sxy.shape)

#    exit()
   return Sxy 


from data.load import load_xr_dataset

args = '--filtering gaussian --interior False'
ds,scs = load_xr_dataset(args.split(),high_res = False)

ds = ds.isel(time = 3).fillna(0)
inputs = np.stack([np.stack([ds.u.values*u_scale,ds.v.values*v_scale],axis = 0)],axis = 0)
true_vals = np.stack([ds.Su.values,ds.Sv.values],axis = 0)
true_vals = true_vals[:,10:-10,10:-10]

with torch.set_grad_enabled(False):
    outputs = nn.forward(torch.tensor(inputs,dtype = torch.float32))
    mean,std = outputs[0,:2],outputs[0,2:]
ssc = np.array([Su_scale,Sv_scale]).reshape([-1,1,1])
mean,std =mean.numpy()*ssc,std.numpy()*ssc

mean[true_vals == 0] = 0
std[true_vals == 0] = 0

import matplotlib.pyplot as plt
fig,axs = plt.subplots(2,3,figsize = (35,12))
for i in range(2):
    for j,vec in zip(range(3),[true_vals,mean,std],):
        val = np.log10(np.abs(vec[i,::-1]))
        if j == 1:
            vmax = np.amax(np.log10(np.abs(true_vals[i])))
            vmin = -14#np.amin(np.log10(np.abs(true_vals[i])))
        else:
            vmax = np.amax(np.log10(np.abs(vec[i])))
            vmin = -14#np.amin(np.log10(np.abs(vec[i])))
        print(i,j,vmax,vmin)
        pos = axs[i,j].imshow(val,cmap = 'bwr',vmax = vmax,vmin = vmin)
        fig.colorbar(pos, ax=axs[i,j])
fig.savefig('model_outputs.png')
    


