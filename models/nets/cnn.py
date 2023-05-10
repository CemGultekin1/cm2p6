from collections import OrderedDict
from typing import Dict
import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.nn.functional import softplus
import numpy as np

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
    def __init__(self) -> None:
        super().__init__()
        self._min_value = Parameter(torch.tensor(0.1))
    def forward(self, x):
        x0,x1 = torch.split(x,x.shape[1]//2,dim=1)
        x1 = softplus(x1) + softplus(self._min_value)        
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
    
class DoubleCNN(CNN):
    def __init__(self, cnn1:CNN,widths=None, kernels=None, batchnorm=None, seed=None, **kwargs):
        super().__init__(widths, kernels, batchnorm, seed, **kwargs)
        self.cnn1 = cnn1
        self.cnn1.eval()
        assert self.spread == cnn1.spread
    def forward(self,x):
        conditional_mean,_ = self.cnn1.forward(x)
        _,conditional_var = super().forward(x)
        return conditional_mean,conditional_var
        

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def adjustcnn(kernel_factor = 1.,width_factor =1., kernel_size = -1,constant_nparam = True,**kwargs):
    kernels = kwargs.pop('kernels')
    widths = kwargs.pop('widths')
    kernels = list(kernels)
    widths = list(widths)
    def compute_view_field(kernels):
        spread = 0
        for i in range(len(kernels)):
            spread+=kernels[i]-1
        return spread

    n0 = count_parameters(CNN(kernels = kernels,widths = widths,**kwargs))
    view_field = compute_view_field(kernels)
    def compare(view,):
        if kernel_size < 0 :
            return view > view_field*kernel_factor
        return view > kernel_size - 1
    i=0
    while compare(compute_view_field(kernels)):
        K = np.amax(np.array(kernels))
        if K ==1 :
            break
        I = np.where(np.array(kernels)== K)[0]
        i = I[-1]
        kernels[i]-=1
    if compute_view_field(kernels)%2 == 1 :
        kernels[i]+=1
    if constant_nparam:
        n1 = count_parameters(CNN(kernels = kernels,widths = widths,**kwargs))
        wd = np.array(widths[1:-1])
        wd = np.round(wd * np.sqrt(n0/n1) * width_factor).astype(int).tolist()
        widths = [widths[0]] + wd + [widths[-1]]
    return widths,kernels

def kernels2spread(kernels):
    spread = 0
    for i in range(len(kernels)):
        spread+=(kernels[i]-1)/2
    spread = int(spread)
    return spread




class LCNN(nn.Module):
    def __init__(self,widths = None,kernels = None,batchnorm = None,seed = None,**kwargs):
        super(LCNN, self).__init__()
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
        self.device = device

        self.skipcons = False
        # layers = OrderedDict()
        
        self.spread = kernels2spread(kernels)
        torch.manual_seed(seed)

        self.nn_layers = nn.ModuleList()
        initwidth = widths[0]
        width = widths[1:]

        filter_size = kernels
        self.nn_layers.append(nn.Conv2d(initwidth, width[0], filter_size[0]) )
        self.num_layers = len(filter_size)
        for i in range(1,self.num_layers):
            self.nn_layers.append(nn.BatchNorm2d(width[i-1]).to(device) )
            self.nn_layers.append(nn.Conv2d(width[i-1], width[i], filter_size[i]) )
        self.nn_layers.append(nn.Softplus())
        self.receptive_field=int(self.spread*2+1)


    def forward(self, x):
        cn=0
        for _ in range(self.num_layers-1):
            x = self.nn_layers[cn](x)
            cn+=1
            x = F.relu(self.nn_layers[cn](x))
            cn+=1
        x=self.nn_layers[cn](x)
        cn+=1
        mean,precision=torch.split(x,x.shape[1]//2,dim=1)
        precision=self.nn_layers[cn](precision)

        return mean,precision



class DoubleLCNNWrapper(LCNN):
    var_net :LCNN
    def add_var_part(self,net:LCNN):
        self.var_net = net
    def forward(self, x):
        mean,_ =  super().forward(x)
        _,var = self.var_net.forward(x)
        return mean,1/var
    def train(self,mean = True,var = True):
        if mean:
            super().train()
        if var:
            self.var_net.train()
    def eval(self,mean = True,var = True):
        if mean:
            super().eval()
        if var:
            self.var_net.eval()