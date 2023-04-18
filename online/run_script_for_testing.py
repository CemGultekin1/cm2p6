#!/bin/env python

import torch
from torch.nn import functional as F
from torch.nn import Sequential
from torch import nn
import numpy as np
import math
from constants.paths import OUTPUTS_PATH,ONLINE_MODELS
import os
import matplotlib.pyplot as plt
from online.run_script import u_scale,v_scale,Su_scale,Sv_scale,CNN

class DetectOutputSizeMixin:
    def output_width(self, input_height, input_width):
        x = torch.zeros((1, self.n_in_channels, input_height, input_width))
        x = x.to(device=self.device)
        y = self(x)
        # temporary fix for student loss
        if isinstance(y, tuple):
            y = y[0]
        return y.size(3)

    def output_height(self, input_height, input_width):
        x = torch.zeros((1, self.n_in_channels, input_height, input_width))
        x = x.to(device=self.device)
        y = self(x)
        # temporary fix for student loss
        if isinstance(y, tuple):
            y = y[0]
        return y.size(2)

    @property
    def device(self):
        p = list(self.parameters())[0]
        return p.device

class FinalTransformationMixin:
    @property
    def final_transformation(self):
        return self._final_transformation

    @final_transformation.setter
    def final_transformation(self, transformation):
        self._final_transformation = transformation

    def forward(self, x):
        x = super().forward(x)
        return self.final_transformation(x)

class FullyCNN(DetectOutputSizeMixin, Sequential):

    def __init__(self, n_in_channels: int = 2, n_out_channels: int = 4,
                 padding=None, batch_norm=False):
        if padding is None:
            padding_5 = 0
            padding_3 = 0
        elif padding == 'same':
            padding_5 = 2
            padding_3 = 1
        else:
            raise ValueError('Unknow value for padding parameter.')
        self.n_in_channels = n_in_channels
        self.batch_norm = batch_norm
        conv1 = torch.nn.Conv2d(n_in_channels, 128, 5, padding=padding_5)
        block1 = self._make_subblock(conv1)
        conv2 = torch.nn.Conv2d(128, 64, 5, padding=padding_5)
        block2 = self._make_subblock(conv2)
        conv3 = torch.nn.Conv2d(64, 32, 3, padding=padding_3)
        block3 = self._make_subblock(conv3)
        conv4 = torch.nn.Conv2d(32, 32, 3, padding=padding_3)
        block4 = self._make_subblock(conv4)
        conv5 = torch.nn.Conv2d(32, 32, 3, padding=padding_3)
        block5 = self._make_subblock(conv5)
        conv6 = torch.nn.Conv2d(32, 32, 3, padding=padding_3)
        block6 = self._make_subblock(conv6)
        conv7 = torch.nn.Conv2d(32, 32, 3, padding=padding_3)
        block7 = self._make_subblock(conv7)
        conv8 = torch.nn.Conv2d(32, n_out_channels, 3, padding=padding_3)
        Sequential.__init__(self, *block1, *block2, *block3, *block4, *block5,
                            *block6, *block7, conv8)

    @property
    def final_transformation(self):
        return self._final_transformation

    @final_transformation.setter
    def final_transformation(self, transformation):
        self._final_transformation = transformation

    def forward(self, x):
        x = super().forward(x)
        x[:,2:] = torch.maximum(x[:,2:],0.1*torch.ones_like(x[:,2:]))
        # for i in range(4):
        #     xi = x[0,i].numpy()
        #     print(f'ARTHUR_{i} = ',np.mean(np.abs(xi)))
        
        return x #self.final_transformation(x)

    def _make_subblock(self, conv):
        subbloc = [conv, nn.ReLU()]
        if self.batch_norm:
            subbloc.append(nn.BatchNorm2d(conv.out_channels))
        return subbloc



def MOM6_testNN(nn,uv,pe,pe_num,u_scale,v_scale,Su_scale,Sv_scale):
   global gpu_id
   use_cuda = False
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
       outs = nn.forward(x)
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
   Sxy[0,:,:,:] = (mean[0,:,:,:] + epsilon_x*std[0,:,:,:])*Su_scale
   Sxy[1,:,:,:] = (mean[1,:,:,:] + epsilon_y*std[1,:,:,:])*Sv_scale
   Sxy[2,:,:,:] = mean[0,:,:,:]*Su_scale
   Sxy[3,:,:,:] = mean[1,:,:,:]*Sv_scale
   Sxy[4,:,:,:] = std[0,:,:,:]*Su_scale
   Sxy[5,:,:,:] = std[1,:,:,:]*Sv_scale
   return Sxy 


def run_model(cnns_:dict,):
    from data.load import load_xr_dataset

    args = '--filtering gaussian --num_workers 1 --interior False'
    ds,_ = load_xr_dataset(args.split(),high_res = False)

    ds = ds.isel(time = 0,).fillna(0)

    inputs = np.stack([np.stack([ds.u.values,ds.v.values],axis = 0)],axis = 0)
    
    
    inputs = inputs.transpose([1,2,3,0])
    outputs = {}
    for name,cnn_ in cnns_.items():
        scales = [u_scale,v_scale,Su_scale,Sv_scale]
        print(name)
        if 'GZ21' in name:
            scales = [10,10,1e-7,1e-7]
        Sxy = MOM6_testNN(cnn_,inputs,0,1,*scales)
        cnn_.to('cpu')
        Sxy = np.squeeze(Sxy)        
        Sxy[4:] = np.log10(Sxy[4:])
        Sxy = np.where(np.isnan(Sxy),np.nan,Sxy)
        outputs[name] = Sxy[2:]

    true_vals = np.stack([ds.Su.values,ds.Sv.values],axis = 0)
    true_vals = true_vals[:,10:-10,10:-10]
    
    outputs['true_outputs'] = true_vals
    
    return outputs
def imshow_on_ax(fig,ax,val,title,zero_centered:bool,**kwargs):
    val = np.squeeze(val)
    vmax = kwargs.get('vmax')
    vmin = kwargs.get('vmin')
    if zero_centered:
        vmax = np.maximum(np.abs(vmax),np.abs(vmin))
        vmin = -vmax
    pos = ax.imshow(val[::-1],cmap = 'bwr',vmax = vmax,vmin = vmin)
    ax.set_title(title)
    fig.colorbar(pos, ax=ax)
def get_vmin_vmax(outputs:dict):
    vmaxs = [-np.inf]*3
    vmins = [np.inf]*3
    vmaxs_ = [0]*3
    vmins_ = [0]*3
    
    true_value = outputs['true_outputs']
    true_value = true_value[0]
    true_value = np.where(np.isnan(true_value),0,true_value)
    for key,val in outputs.items():
        if key == 'true_outputs':
            continue
        val = np.where(np.isnan(val),0,val)        
        vmaxs_[0] = np.amax(val[0])
        vmaxs_[1] = np.amax(val[0] - true_value)
        vmaxs_[2] = np.amax(val[2])

        vmins_[0] = np.amin(val[0])
        vmins_[1] = np.amin(val[0] - true_value)
        vmins_[2] = np.amin(val[2])
        for i in range(3):
            vmaxs[i] = np.maximum(vmaxs[i],vmaxs_[i])    
            vmins[i] = np.minimum(vmins[i],vmins_[i])    
    return vmaxs,vmins
def plot_std_distribution(outputs:dict):
    outputs.pop('true_outputs',None)
    fig,axs = plt.subplots(1,2,figsize = (15,7))
    for key,val in outputs.items():
        std = np.sort(val[2].flatten())[::-1]
        axs[0].semilogy(std,label = key)
        std = np.sort(val[3].flatten())[::-1]
        axs[1].semilogy(std,label = key)
    axs[0].legend()
    axs[1].legend()
    axs[0].set_title('Su forcing std outputs sorted')
    axs[1].set_title('Sv forcing std outputs sorted')
    fig.savefig('std_distributions.png')
        
        
def plot_outputs(outputs:dict):
    vmaxs,vmins = get_vmin_vmax(outputs)
    num_models = len(outputs)
    fig,axs = plt.subplots(3,num_models,figsize = (12*num_models,24))
    true_value = outputs.pop('true_outputs')
    model_names = tuple(outputs.keys())
    imshow_on_ax(fig,axs[0,0],true_value[0],'true Su',True,vmax = vmaxs[0],vmin = vmins[0])
    imshow_on_ax(fig,axs[1,0],true_value[0]*0,'*',True,vmax = vmaxs[1],vmin = vmins[1])
    imshow_on_ax(fig,axs[2,0],true_value[0]*0,'*',False,vmax = vmaxs[2],vmin = vmins[2])
    true_value = true_value[0]
    for r,i in enumerate(range(1,num_models)):
        mn = model_names[r]
        tv = outputs[mn].squeeze()
        imshow_on_ax(fig,axs[0,i],tv[0],f'{mn} Subgrid-u',True,vmax = vmaxs[0],vmin = vmins[0])
        imshow_on_ax(fig,axs[1,i],tv[0] - true_value,f'{mn} Subgrid-err',True,vmax = vmaxs[1],vmin = vmins[1])
        imshow_on_ax(fig,axs[2,i],tv[2],f'{mn} log10|std-u|',False,vmax = vmaxs[2],vmin = vmins[2])

    root = os.path.join(OUTPUTS_PATH,'tobedeleted_')
    if not os.path.exists(root):
        os.makedirs(root)
    fig.savefig(os.path.join(root,f'crowded_comparison.png'))
    print(os.path.join(root,f'crowded_comparison.png'))

def load_model(path,old_model_flag,fully_cnn_flag:bool = False):
    statedict = torch.load(path,map_location=torch.device('cpu'))
    if old_model_flag:
        statedict = {'x':statedict}
    else:
        modelsdict = {}
    for name,sttdict in statedict.items():
        if not old_model_flag:
            cnn_arguments,_ = sttdict
        if  not fully_cnn_flag:
            nn=CNN(**cnn_arguments)#cuda_flag=False,relu_flag= 1 - old_model_flag)
        else:        
            nn = FullyCNN()
        
        if not old_model_flag:
            _,sttdict = sttdict
        nn.load_state_dict(sttdict,strict = False)
        nn.eval()
        if old_model_flag:
            return nn
        else:
            modelsdict[name] = nn
        break
    return modelsdict
    
def load_models():
    models = {}
    
    path = os.path.join(OUTPUTS_PATH,'GZ21.pth')
    models['GZ21'] = load_model(path,True,fully_cnn_flag=True)
    

    filenames = ['20230418']
    for fn in filenames:
        path = os.path.join(ONLINE_MODELS,'cem_' + fn + '.pth')
        for name,model in load_model(path,False).items():
            models[name] =  model
    return models

def main():
    models = load_models()
    outputs = run_model(models)
    plot_outputs(outputs)
    # plot_std_distribution(outputs)

if '__main__' == __name__:
    main()
    

    

