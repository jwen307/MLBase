#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 14:10:58 2021

@author: jeff
"""
import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class LeNet(nn.Module):
    
    def __init__(self,num_classes = 10, in_channels = 3, img_size = 28):
        super().__init__()
        
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels=6, kernel_size = 5, stride = 1, padding = 2),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size = 2),
            
            nn.Conv2d(in_channels = 6, out_channels=16, kernel_size = 5, stride = 1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size = 2),
            
            nn.Conv2d(in_channels = 16, out_channels=120, kernel_size = 5, stride = 1),
            nn.Tanh(),

            )
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features=int(120*((img_size/4)-6)**2), out_features=84),
            nn.Tanh(),
            nn.Linear(in_features = 84, out_features=num_classes),
            )
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x,1)
        
        out = self.classifier(x)
        
        return out
    
    
def lenet(pretrained: bool = False, pretrained_file: str = None, **kwargs) -> LeNet:
    
    model = LeNet(**kwargs)
    
    if pretrained:
        state_dict = torch.load(pretrained_file) #Should be a .pth file
        model.load_state_dict(state_dict)
    return model



if __name__ == '__main__':
    x = 0