import open3d as o3d
import numpy as np
import os
import sys

sys.path.append('')

# 1. Point Cloud Outlier Removal 

print("Load a ply point cloud, print it, and render it")
dataset = o3d.data.PLYPointCloud()
pcd = o3d.io.read_point_cloud(dataset.path)
#For initial view
o3d.visualization.draw_geometries([pcd],
                                  zoom = 1.1,
                                  front = [0.412, -0.21, -0.87],
                                  lookat = [2.6, 2.04, 1.53],
                                  up = [-0.009, -0.9, 0.2])

print("Downsample the pointcloud with a voxel value")

voxel_down_pcd = pcd.voxel_down_sample(voxel_size = 0.02) #To reduce the point. Try and change this ! 
o3d.visualization.draw_geometries([voxel_down_pcd],
                                  zoom = 1.1,
                                  front = [0.412, -0.21, -0.87],
                                  lookat = [2.6, 2.04, 1.53],
                                  up = [-0.009, -0.9, 0.2])

# Another way to downsample Point Cloud
print("Every 5th points are selected")
uni_down_pcd = pcd.uniform_down_sample(every_k_points = 5) #Change this value ! 
o3d.visualization.draw_geometries([uni_down_pcd],
                                  zoom = 1.1,
                                  front = [0.412, -0.21, -0.87],
                                  lookat = [2.6, 2.04, 1.53],
                                  up = [-0.009, -0.9, 0.2])


# Select down sample. Takes a binary masks to output only selected points. 

def display_inlier_outlier (cloud, idx):
    inlier_cloud = cloud.select_by_index(idx)
    outlier_cloud = cloud.select_by_index (idx, invert= True)
    
    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1,0,0])
    inlier_cloud.paint_uniform_color([0.8,0.8,0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                  zoom = 1.1,
                                  front = [0.412, -0.21, -0.87],
                                  lookat = [2.6, 2.04, 1.53],
                                  up = [-0.009, -0.9, 0.2])

'''

Statistical Outlier Removal 

Removes points that are further away from their neighbors compared to the average for the PC. 
Takes two inputs : 
1. nb_neighbors - specifies how many neighbors are taken into account to calc the average distance
2. std_ratio - sets threshold level based on standard deviation for average distances across PC. 

'''

print("statistical outlier removal")
cl, idx = voxel_down_pcd.remove_statistical_outlier(nb_neighbors = 20 , #we can play with this values
                                                     std_ratio= 2.0)
display_inlier_outlier(voxel_down_pcd, idx)


'''

Radius Outlier Removal 

removes points that have few neighbors in a given sphere around them
Needs two params: 
1. nb_points : 
2. radius  :

'''
print("Radius Outlier removal")
cl2, idx = voxel_down_pcd.remove_radius_outlier(nb_points = 16, radius = 0.09)
display_inlier_outlier(voxel_down_pcd, idx)