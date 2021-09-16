#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 12:11:20 2021

@author: jeff
"""

import sys
import os
import os.path as osp
import argparse
import numpy as np
import cv2
import math
import torch
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
from torch.nn.parallel.data_parallel import DataParallel
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as F
from tqdm import tqdm
import open3d as o3d
import time

sys.path.insert(0, osp.join('posenet', 'posenet_common'))
from posenet_utils.vis import vis_keypoints, vis_3d_multiple_skeleton


#Save the 2d pose imposed on the actual image
def vis_2d_pose(imgs, output_pose_2d_list, skeleton, joint_num, abs_counts, output_dir):
    
    for i in range(len(imgs)):
        
        if isinstance(output_pose_2d_list[i],np.ndarray):
            
            # visualize 2d poses
            vis_img = (imgs[i]*255).cpu().permute(1,2,0).numpy().copy()
            vis_kps = np.zeros((3,joint_num))
            vis_kps[0,:] = output_pose_2d_list[i][:,0]
            vis_kps[1,:] = output_pose_2d_list[i][:,1]
            vis_kps[2,:] = 1
            vis_img = vis_keypoints(cv2.cvtColor(vis_img,cv2.COLOR_RGB2BGR), vis_kps, skeleton)
            cv2.imwrite(output_dir + 'output_pose_2d{0}.jpg'.format(abs_counts[i]), vis_img)
            
            #Note: CV2 uses BGR color format by default so need to doc v2.cvtColor(vis_img,cv2.COLOR_RGB2BGR)

        

def scene_4d_human(mesh_dir, extrinsics, world_poses, skeleton):
    
    #Get the mesh of the scene
    mesh= o3d.io.read_triangle_mesh(mesh_dir)
    mesh.compute_vertex_normals()
    
    #Visualize the scene
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh)
    
    first_pose = []
    for i, pose in enumerate(world_poses):
    
        if isinstance(pose,np.ndarray):
            
            first_pose = pose
            
            break
    
    
    #Initialize the line_set geometry
    line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(first_pose),
                lines=o3d.utility.Vector2iVector(skeleton),
                )
    vis.add_geometry(line_set)
    
    
    #Sphere is the camera position
    cam = o3d.geometry.TriangleMesh.create_cone(radius=0.07, height=0.2)
    cam.paint_uniform_color([0.0, 0.2, 1.0])
    vis.add_geometry(cam)
    
    
    for i, pose in enumerate(world_poses):
    
        if isinstance(pose,np.ndarray):
            
            extrinsic = extrinsics[i]
            #Need to at 1.5 to the z component to match xy plane of scannet
            extrinsic[2,3] += 1.5
            
            #Update skeleton and camera position
            line_set.points = o3d.utility.Vector3dVector(pose)
            cam.transform(extrinsic)
        
            vis.update_geometry(line_set)
            vis.update_geometry(cam)
            vis.poll_events()
            vis.update_renderer()
            
            #Put sphere back at world center
            cam.transform(np.linalg.inv(extrinsic))
            
            time.sleep(0.05)
     
  
    #vis.destroy_window()
        