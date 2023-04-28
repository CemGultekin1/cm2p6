
import torch
from torch.nn import functional as F
from torch.nn import Sequential
from torch import nn

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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 19:38:09 2020

@author: arthur
In this file we define some transformations applied to the output of our 
models. This allows us to keep separate these from the models themselves.
In particular, when we use a heteroskedastic loss, we compare two
transformations that ensure that the precision is positive.
"""

from abc import ABC, abstractmethod
from torch.nn import Module, Parameter
import torch
from torch.nn.functional import softplus


class Transform(Module, ABC):
    """Abstract Base Class for all transforms"""

    def __init__(self):
        super().__init__()

    @abstractmethod
    def transform(self, input):
        pass

    def forward(self, input_):
        return self.transform(input_)

    @abstractmethod
    def __repr__(self):
        pass
    
    def __call__(self,input_):
        return self.forward(input_)

class PrecisionTransform(Transform):
    def __init__(self, min_value=0.1):
        super().__init__()
        self._min_value = Parameter(torch.tensor(min_value))
        self.indices = slice(2,4)

    @property
    def min_value(self):
        return softplus(self._min_value)

    @min_value.setter
    def min_value(self, value):
        self._min_value = Parameter(torch.tensor(value))

    @property
    def indices(self):
        """Return the indices transformed"""
        return self._indices

    @indices.setter
    def indices(self, values):
        self._indices = values

    def transform(self, input_):
        # Split in sections of size 2 along channel dimension
        # Careful: the split argument is the size of the sections, not the
        # number of them (although does not matter for 4 channels)
        result = torch.clone(input_)
        result[:, self.indices, :, :] = self.transform_precision(
            input_[:, self.indices, :, :]) + self.min_value
        return result

    @staticmethod
    @abstractmethod
    def transform_precision(precision):
        pass

class SquareTransform(PrecisionTransform):
    def __init__(self, min_value=0.1):
        super().__init__(min_value)

    @staticmethod
    def transform_precision(precision):
        return precision**2

    def __repr__(self):
        return ''.join(('SquareTransform(', str(self.min_value), ')'))
    
class SoftPlusTransform(PrecisionTransform):
    def __init__(self, min_value=0.1):
        super().__init__(min_value)

    @staticmethod
    def transform_precision(precision):
        return softplus(precision)

    def __repr__(self):
        return ''.join(('SoftPlusTransform(', str(self.min_value), ')'))
class FullyCNN(DetectOutputSizeMixin, Sequential):

    def __init__(self, n_in_channels: int = 2, n_out_channels: int = 4,
                 padding=None, batch_norm=False,final_activation:str = "softplus",**kwargs):
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
        self.spread = 10
        if final_activation == "softplus":
            self.final_transformation = SoftPlusTransform()
        elif final_activation == "square":
            self.final_transformation = SquareTransform()
        else:
            raise Exception
        
    @property
    def final_transformation(self):
        return self._final_transformation

    @final_transformation.setter
    def final_transformation(self, transformation):
        self._final_transformation = transformation

    def forward(self, x):
        x = super().forward(x)
        
        x = self.final_transformation(x)
        x,y = torch.split(x,2,dim = 1)
        return x,y**2

    def _make_subblock(self, conv):
        subbloc = [conv, nn.ReLU()]
        if self.batch_norm:
            subbloc.append(nn.BatchNorm2d(conv.out_channels))
        return subbloc
    
    