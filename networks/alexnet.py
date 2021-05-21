#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 15:40:28 2021

@author: jeff
"""

import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url


class AlexNet(nn.Module):
    #AlexNet architecture from <https://arxiv.org/abs/1404.5997>
    #Based on PyTorch implementation from https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py
    
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        
        #Feature layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        #Fully Connected Classifier Network
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        
        
    #Forward pass through network
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    
   
#Function to load network
def alexnet(pretrained: bool = False, online: bool = True, pretrained_file: str = None, progress: bool = True, **kwargs) -> AlexNet:
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        online: (bool): If True, pulls the pre-trained ImageNet model from online
        pretrained_file: Location of the pretrained AlexNet model
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    if pretrained:
        #If online, pull the pretrained network from the URL
        if online:
            state_dict = load_state_dict_from_url('https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',progress=progress)
        else:
            state_dict = torch.load(pretrained_file) #Should be a .pth file
        model.load_state_dict(state_dict)
    return model
    
    