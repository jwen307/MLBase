#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 15:13:52 2021

@author: jeff

Implementation of the AlexNet architecture. Used the same naming scheme and convolutional channel sizes
so the pretrained AlexNet can be imported
"""

import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class AlexNet(nn.Module):
    
    #Constructor
    def __init__(self, num_classes=1000):
        
        #Inherit functions and properties of parent class
        super().__init__()
        
        self.num_classes = num_classes
        
        #Convolutional layers
        self.features = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = (11,11), stride = 4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size = (3,3), stride = 2),
    
            nn.Conv2d(64,192,[5,5], padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size = (3,3), stride = 2),
            
            nn.Conv2d(192,384,[3,3], padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384,256,[3,3], padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256,256,[3,3], padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size = (3,3), stride = 2),
            )
        
        self.avgpool = nn.AdaptiveAvgPool2d((6,6))
        
        
        #Fully Connected Layers
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(9216,4096),
            nn.ReLU(inplace=True),
            
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            
            nn.Linear(4096,self.num_classes),
            )
        
        
    #Define the forward pass
    def forward(self,x):
        
        #Pass through the convolutional layers
        x = self.features(x)
        x = self.avgpool(x)
        
        #Flatten
        x = x.view(-1, 9216) #could also use torch.flatten(x,1)
        
        #Pass through dense layers
        x = self.classifier(x)
        
        return x
        
        
#Load a pretrained AlexNet
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

        



if __name__ == '__main__':
    
    cifar_dir = '../../../Data/Datasets/'
    
    model = alexnet(pretrained = True, num_classes = 1000)
    
    #Try on CIFAR10
    dataset = torchvision.datasets.CIFAR10(cifar_dir, train=False, transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
        ]))
    
    
    #View example images
    plt.figure()
    plt.imshow(dataset[23][0].permute(1,2,0).numpy())
    
    #Pass through model
    model.eval()
    softmax = torch.nn.Softmax()
    with torch.no_grad():
        probs = softmax(model(dataset[23][0].unsqueeze(0)))
        class_decision = probs.argmax()
        top5 = torch.topk(probs,5)
        
    print(top5)
    
    
    
    
    
    
    
    