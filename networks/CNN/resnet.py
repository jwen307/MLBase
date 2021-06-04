#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 15:10:29 2021

@author: jeff
"""

import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.models.utils import load_state_dict_from_url
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


#ResNet class
class ResNet(nn.Module):
    
    #Initialize
    def __init__(self, num_layers):
        super().__init()
        
        
    #Forward pass
    def forward(self,x):
        dsf


class ResidualBlock(nn.Module): 

    #Initialize
    def __init__(self, downsample = False, in_channels, out_channels):
        super().__init()
        
         equal_size_feats = in_channels == out_channels
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        #First layer in the residual block has a stride of two if downsampling
        if downsample:
            self.layer1 = nn.Conv2d(in_channels, out_channels, 3, stride = 2, padding = 1)
        else:
            self.layer1 = nn.Conv2d(in_channels, out_channels, 3, stride = 1, padding = 1)
            
        self.layer2 = nn.Conv2d(in_channels, out_channels, 3, stride = 1, padding = 1)
        
    #Forward pass
    def forward(self,x):
        
        original_input = x * 1.0
        x = self.layer1(x)
        x = self.bn(x)
        x = self.relu(x)
        
        x = self.layer2(x)
        x = self.bn(x)
        
        #If the residual feature maps and original features maps are the same size, just do elementwise addition
        if equal_size_feats:
            output = x + original_input
        #Otherwise, do the identity mapping by adding zeros
        else:
            
