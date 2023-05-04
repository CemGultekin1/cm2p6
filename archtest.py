from typing import List
from torch import nn
import torch
from collections import OrderedDict
class ConvLayer(nn.Sequential):
    def __init__(self,width0:int,width1:int,kernel0:int,kernel1:int = None,batchnorm:bool = False,nnlnr:bool = True):
        if kernel1 is None:
            kernel1 = kernel0
        d = []
        d.append(('conv',nn.Conv2d(width0,width1,(kernel0,kernel1))))
        if batchnorm:
            assert nnlnr
            d.append(('bnorm', nn.BatchNorm2d(width1)))
        if nnlnr:
            d.append(('nnlnr',nn.ReLU(inplace = True)))
        super().__init__(OrderedDict(d))
class Model(nn.Sequential):
    def __init__(self,widths:List[int],kernels:List[int],batchnorm:List[bool],nnlnr:List[bool]) -> None:
        d = []
        zipwidths = zip(widths[:-1],widths[1:])
        for i,((w0,w1),k,b,nnl) in enumerate(zip(zipwidths,kernels,batchnorm,nnlnr)):
            conv2d = (f'layer{i}', ConvLayer(w0,w1,k,k,b,nnl))
            d.append(conv2d)
        super().__init__(OrderedDict(d))
# model = Model([2,3,1],[2,3],[1,1],[1,1])
# x = torch.randn(1,2,5,5)
# y = model(x)
# loss = torch.sum(y)
# loss.backward()
# # print(loss)
# print(model.layer0.conv.weight.grad)

widths = [2,3,4]
kernels = [3,3]
batchnorm = [1,1]
min_precision = 0.1
from models.nets.cnn import CNN

cnn = CNN(widths,kernels,batchnorm,0,min_precision,)
print(cnn)