#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 08:58:43 2021

@author: jeff
"""



import numpy as np
import cv2
import torch
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F



#Show multiple images in a grid from a tensor
def show_tensor_imgs(plot_imgs,nrow = 7, **kwargs):
    ''' 
    Show tensor images (with values between -1 and 1) in a grid
    
    plot_imgs: (batch_size, num_channels, height, width) [Tensor] tensor of imgs with values between -1 and 1
    nrows: Number of imgs to include in a row
    '''
    
    #Put the images in a grid and show them
    grid = torchvision.utils.make_grid(plot_imgs.clamp(min=-1, max=1), nrow = int(nrow), scale_each=True, normalize=True)
    f = plt.figure()
    f.set_figheight(15)
    f.set_figwidth(15)
    plt.imshow(grid.permute(1, 2, 0).numpy())
    
    
#For images with values from 0 to 255
def show_imgs(imgs):
    ''' 
    Show tensor images (with values between 0 and 255)
    
    imgs: [List of Tensors or single tensor] tensor of imgs with values between -1 and 1
    nrows: Number of imgs to include in a row
    '''
    
    if not isinstance(imgs, list):
        imgs = [imgs]
        
    #Plot the imgs
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    
#Show the image with the given bounding box
def show_bounding_box(img_tensor, bounding_boxes):
    ''' 
    Show tensor images (with values between 0 and 1) with bounding boxes
    
    tensor_imgs: [Tensor] tensor img with values between 0 and 1
    bounding_boxes: [list or list of lists] bounding boxes [xmin,ymin,xmax,ymax]
    '''

    #Get the image with the bounding boxes
    img_bb = torchvision.utils.draw_bounding_boxes((img_tensor*255).type(torch.uint8), boxes = bounding_boxes)
    
    #Show the img
    show_imgs(img_bb)