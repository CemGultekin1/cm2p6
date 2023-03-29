#!/bin/env python

import torch
from torch.nn import functional as F
from torch import nn
import numpy as np
import math

u_scale = 1/0.09439346225350978
v_scale = 1/0.07252696573672539
Su_scale = 4.9041400042653195e-08
Sv_scale = 4.8550991806254025e-08

class CNN(nn.Module):
    def __init__(self,filter_size=[5, 5, 3, 3, 3, 3, 3, 3],\
                     width=[128, 64, 32, 32, 32, 32, 32, 4],\
                        inchan=2,cuda_flag=False,relu_flag = True):
        super(CNN, self).__init__()
        self.nn_layers = nn.ModuleList()
        self.filter_size=filter_size
        self.num_layers=len(filter_size)
        
        if cuda_flag:
            device = "cuda:0" 
        else:  
            device = "cpu"  
        self.relu_flag = relu_flag
        self.nn_layers.append(nn.Conv2d(inchan, width[0], filter_size[0]).to(device) )
        for i in range(1,self.num_layers):
            self.nn_layers.append(nn.BatchNorm2d(width[i-1]).to(device) )
            if relu_flag:
                self.nn_layers.append(nn.ReLU(inplace = True))
            self.nn_layers.append(nn.Conv2d(width[i-1], width[i], filter_size[i]).to(device) )
        self.nn_layers.append(nn.Softplus().to(device))
    def forward(self, x):
        cn=0
        x = self.nn_layers[cn](x) # conv2d
        cn+=1
        while cn<len(self.nn_layers)-1:
            x = self.nn_layers[cn](x) # batch
            cn+=1
            if self.relu_flag:
                x = self.nn_layers[cn](x) # relu #torch.relu(x)#
                cn+=1
            else:
                x = torch.relu(x)
            x = self.nn_layers[cn](x) # conv2d 
            cn+=1
        mean,precision=torch.split(x,x.shape[1]//2,dim=1)
        precision=self.nn_layers[-1](precision) # softplus
        out=torch.cat([mean,precision],dim=1)
        return out
    



def MOM6_testNN(nn,uv,pe,pe_num,index):
   global gpu_id,u_scale,v_scale,Su_scale,Sv_scale
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
       out = nn.forward(x)
       # end_time = time.time()
   if use_cuda:
       out = out.to('cpu')
   out = out.numpy().astype(np.float64)
   # At this point, python out shape is (nk,4,ni,nj)
   # Comment-out is tranferring arraies into F order
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
   # full output
   Sxy[0,:,:,:] = (out[0,:,:,:] + epsilon_x*np.sqrt(1/out[2,:,:,:]))*Su_scale
   Sxy[1,:,:,:] = (out[1,:,:,:] + epsilon_y*np.sqrt(1/out[3,:,:,:]))*Sv_scale
#    Sxy[0,:,:,:] = out[0,:,:,:]*Su_scale
#    Sxy[1,:,:,:] = out[1,:,:,:]*Sv_scale
   Sxy[2,:,:,:] = out[0,:,:,:]*Su_scale
   Sxy[3,:,:,:] = out[1,:,:,:]*Sv_scale
   Sxy[4,:,:,:] = np.sqrt(1/out[2,:,:,:])*Su_scale
   Sxy[5,:,:,:] = np.sqrt(1/out[3,:,:,:])*Sv_scale
   return Sxy 


def run_model(cnns_:dict,):
    from data.load import load_xr_dataset

    args = '--filtering gaussian --interior False'
    ds,scs = load_xr_dataset(args.split(),high_res = False)

    ds = ds.isel(time = 3,lat = slice(100,240),lon = slice(100,240)).fillna(0)

    inputs = np.stack([np.stack([ds.u.values,ds.v.values],axis = 0)],axis = 0)
    inputs = inputs.transpose([1,2,3,0])#*0 + 1
    outputs = {}
    for name,cnn_ in cnns_.items():
        Sxy = MOM6_testNN(cnn_,inputs,0,0,0)
        Sxy = np.squeeze(Sxy)
        outputs[name] = Sxy[2:]

    true_vals = np.stack([ds.Su.values,ds.Sv.values],axis = 0)
    true_vals = true_vals[:,10:-10,10:-10]
    
    # outputs['inputs'] = inputs
    outputs['true_outputs'] = true_vals
    
    return outputs
def imshow_on_ax(fig,ax,val,title,pos_valued:bool):
    val = np.squeeze(val)
    if not pos_valued:
        vmax = np.amax(np.abs(val))
        vmin = -vmax
    else:
        vmax = np.amax(val)
        vmin = 0
    pos = ax.imshow(val,cmap = 'bwr',vmax = vmax,vmin = vmin)
    ax.set_title(title)
    fig.colorbar(pos, ax=ax)
    
def plot_outputs(outputs:dict):
    import matplotlib.pyplot as plt
    num_models = len(outputs)
    fig,axs = plt.subplots(3,num_models,figsize = (12*num_models,24))
    true_value = outputs.pop('true_outputs')
    model_names = tuple(outputs.keys())
    imshow_on_ax(fig,axs[0,0],true_value[0],'true Su',False)
    imshow_on_ax(fig,axs[1,0],true_value[0]*0,'*',True)
    imshow_on_ax(fig,axs[2,0],true_value[0]*0,'*',True)
    true_value = true_value[0]
    for r,i in enumerate(range(1,num_models)):
        mn = model_names[r]
        tv = outputs[mn].squeeze()
        imshow_on_ax(fig,axs[0,i],tv[0],f'{mn} Su',False)
        imshow_on_ax(fig,axs[1,i],tv[0] - true_value,f'{mn} Su-err',False)
        imshow_on_ax(fig,axs[2,i],tv[2],f'{mn} Pu',True)
    import os
    from utils.paths import OUTPUTS_PATH
    root = os.path.join(OUTPUTS_PATH,'tobedeleted_')
    if not os.path.exists(root):
        os.makedirs(root)
    fig.savefig(os.path.join(root,f'comparison.png'))

def load_model(path,old_model_flag):
    nn=CNN(cuda_flag=False,relu_flag= 1 - old_model_flag)
    statedict = torch.load(path,map_location=torch.device('cpu'))
    if not old_model_flag:
        _,statedict = statedict['gaussian_four_regions']
    nn.load_state_dict(statedict)
    nn.eval()
    return nn
    
def load_models():

    from utils.paths import OUTPUTS_PATH,ONLINE_MODELS
    import os
    models = {}
    
    path = os.path.join(OUTPUTS_PATH,'best-model')
    models['best-model'] = load_model(path,True)
    
    filenames = ['20230327','20230329']
    for fn in filenames:
        path = os.path.join(ONLINE_MODELS,'cem_' + fn + '.pth')
        models[fn] = load_model(path,False)
    return models

def main():
    models = load_models()
    outputs = run_model(models)
    plot_outputs(outputs)

if '__main__' == __name__:
    main()
    

    

