#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 13:38:57 2021

@author: jeff
"""

import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.models.utils import load_state_dict_from_url
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
}


class VGG(nn.Module):
    
    
    def __init__(self, num_layers: int = 11, num_classes: int = 1000):
        
        #Inherit the nn.Module propertiers
        super().__init__()
        
        #Determine the number of convolutions in each of the layers
        if num_layers == 11:
            num_conv_group1 = 1
            num_conv_group2 = 2
            
        elif num_layers == 13:
            num_conv_group1 = 2
            num_conv_group2 = 2
            
        else:
            num_conv_group1 = 2
            num_conv_group2 = 3
        
        
        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)  
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.group1 = MultiConv(3,64,num_layers = num_conv_group1)
        self.group2 = MultiConv(64,128,num_layers = num_conv_group1)

        self.group3 = MultiConv(128,256,num_layers = num_conv_group2)
        self.group4 = MultiConv(256,512,num_layers = num_conv_group2)
        self.group5 = MultiConv(512,512,num_layers = num_conv_group2)
        
        
        #Classification
        self.fully_connected = nn.Sequential(
            nn.Linear(7*7*512,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096,num_classes)
            )
        
        
        
    #Forward Pass
    def forward(self,x):
        x = self.group1(x)
        x = self.maxpool(x)
        x = self.group2(x)
        x = self.maxpool(x)
        x = self.group3(x)
        x = self.maxpool(x)
        x = self.group4(x)
        x = self.maxpool(x)
        x = self.group5(x)
        x = self.maxpool(x)
        x = self.avgpool(x)
        
        x = torch.flatten(x,1)
        
        out = self.fully_connected(x)
        
        return out
        

#Multiple convolutional layers with the same number of channels for each group     
class MultiConv(nn.Module):
    
    def __init__(self, in_channels: int = 3, num_channels: int = 64, num_layers = 1):
        #Inherit the nn.Module propertiers
        super().__init__()
        
        sequence = OrderedDict([])
        sequence['conv0'] = nn.Conv2d(in_channels, num_channels, kernel_size = 3, padding=1)
        sequence['ReLU0'] = nn.ReLU(inplace=True)
        
        #Add more layers if nedded
        for i in range(num_layers-1):
            sequence['conv{0}'.format(i+1)] = nn.Conv2d(num_channels, num_channels, kernel_size = 3, padding=1)
            sequence['ReLU{0}'.format(i+1)] = nn.ReLU(inplace=True)
        
        self.group = nn.Sequential(sequence)
        
    def forward(self,x):
        x = self.group(x)
        
        return x
    
#Get a vgg network pretrained or not 
def vgg(num_layers: int = 11, num_classes: int = 1000, pretrained: bool = False, online: bool = True, pretrained_file: str = None, progress: bool = True):
    
    #Initialize a model
    model = VGG(num_layers,num_classes)
    
    if pretrained:
        #If online, pull the pretrained network from the URL
        if online:
            state_dict = load_state_dict_from_url(model_urls['vgg{0}'.format(num_layers)],progress=progress)
        else:
            state_dict = torch.load(pretrained_file) #Should be a .pth file
            
        #Add the weights
        model = load_seq(model,state_dict)
        
    return model
    
#Sequentially load the vgg network from the pretrained network
def load_seq(model, state_dict):
    
    #Get the model state dict
    state_dict_mine = model.state_dict()
    
    #Get the weights of the pretrained network
    pretrained_weights = list(state_dict.values())
    
    #Sequentially copy the weights over
    for i, key in enumerate(state_dict_mine.keys()):
        state_dict_mine[key].copy_(pretrained_weights[i])
        
    return model
        
    
    
    
    
#%% Make sure results are the same for PyTorch implementation and mine
cifar_dir = '../../../Data/Datasets/'

if __name__ == '__main__':
    
    

    #Try on CIFAR10
    dataset = torchvision.datasets.CIFAR10(cifar_dir, train=False, transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
        ]))
    
    #Mine
    model = vgg(num_layers = 11, num_classes = 1000, pretrained = True)
    
    #Pass through model
    model.eval()
    softmax = torch.nn.Softmax()
    with torch.no_grad():
        probs = softmax(model(dataset[23][0].unsqueeze(0)))
        class_decision = probs.argmax()
        top5 = torch.topk(probs,5)
        
    print(top5)
    
    
    
    #PyTorch Implementation
    model = torchvision.models.vgg11(pretrained = True)
    
    #Pass through model
    model.eval()
    softmax = torch.nn.Softmax()
    with torch.no_grad():
        probs = softmax(model(dataset[23][0].unsqueeze(0)))
        class_decision = probs.argmax()
        top5 = torch.topk(probs,5)
        
    print(top5)
    
    