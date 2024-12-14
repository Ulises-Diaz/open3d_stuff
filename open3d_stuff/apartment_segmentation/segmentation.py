import numpy as np
import open3d as o3d
import matplotlib as plt
import matplotlib.cm as cm

# 2 . Point Cloud Data preparation

DATANAME = "/home/uli/Desktop/tec/Open3D/004 - 3D Shape Recognition/appartment_cloud.ply"
pcd = o3d.io.read_point_cloud (DATANAME)

# 3 . Data Pre-Processing

pcd_center = pcd.get_center ()
pcd.translate(-pcd_center)
print("Point Cloud Center:", pcd_center)

# 3.1 Statistical outlier filter
nn = 16 

std_multiplier = 10  #to consider which noise to eliminate 

filtered_pcd = pcd.remove_statistical_outlier(nn, std_multiplier)

outliers = pcd.select_by_index (filtered_pcd [1], invert = True)
outliers.paint_uniform_color([1, 0 , 0])

filtered_pcd = filtered_pcd [0]

#o3d.visualization.draw_geometries([filtered_pcd])

# 3.2 Voxel Downsampling

voxel_size= 0.01 
pcd_downsampled = filtered_pcd.voxel_down_sample(voxel_size = voxel_size)
#o3d.visualization.draw_geometries([pcd_downsampled])

# 3.3 Estimating normals 

max_nn = 16  # Set the number of nearest neighbors
pcd_downsampled.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamKNN(knn=max_nn),  # Using KNN search parameter
    fast_normal_computation=True
)

# 3.4 Computing nearest neighbor distances
nn_distance = pcd_downsampled.compute_nearest_neighbor_distance()
mean_distance = np.mean(nn_distance)

pcd_downsampled.paint_uniform_color([0.6, 0.6, 0.6 ])
#o3d.visualization.draw_geometries([pcd_downsampled, outliers])

# 4. E xtracting and setting parameters

front = [ 0.968 , 0.247, -0.0353]

lookat =[0.008497, -0.247, -0.3531]

up = [ 0.011, 0.0955, 0.99]

zoom = 0.2399

pcd = pcd_downsampled

#o3d.visualization.draw_geometries([pcd], zoom=zoom, front=front, lookat = lookat, up =up)

# 5. RANSAC Planar segmentation

pt_to_plane_dist =0.01
planel_model , inliers = pcd.segment_plane(distance_threshold = pt_to_plane_dist, ransac_n = 3,
                                           num_iterations = 1000) 

[a, b, c, d] = planel_model
print(f' plane equation : {a : .2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0')

inlier_cloud = pcd.select_by_index(inliers)
outlier_cloud = pcd.select_by_index(inliers, invert= True)
inlier_cloud.paint_uniform_color([1.0, 0.0 , 0.0])
outlier_cloud.paint_uniform_color ([0.6, 0.6, 0.6])

#o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud], zoom=zoom, front=front, lookat = lookat, up =up)


# 6. Multiorder RANSAC

max_plane_idx = 6
pt_to_plane_dist = 0.02

segment_models = {} 
segments = {}
rest = pcd 

for i in range (max_plane_idx) : 
    colors = cm.get_cmap("tab20")(i)
    segment_models[i], inliers =rest.segment_plane(distance_threshold = pt_to_plane_dist, ransac_n=3, num_iterations= 1000)
    segments [i] = rest.select_by_index(inliers)
    segments [i].paint_uniform_color(list(colors[:3]))
    rest = rest.select_by_index(inliers,invert = True) 
    print ("pass", i , "/" , max_plane_idx, "done.")
    
#o3d.visualization.draw_geometries([segments [i] for i in range (max_plane_idx)] + [rest], zoom=zoom, front=front, lookat = lookat, up =up)


# 7. DBSCAN sur rest. Euclidean Clustering Refine 

labels = np.array (rest.cluster_dbscan(eps=0.05, min_points =5))
max_labels = labels.max ()
print (f' point cloud has {max_labels + 1} clusters ')

colors = cm.get_cmap('tab10')(labels/ (max_labels if max_labels > 0 else 1))
colors[labels < 0] = 0
rest.colors = o3d.utility.Vector3dVector(colors[:, : 3])
o3d.visualization.draw_geometries([segments [i] for i in range (max_plane_idx)] + [rest], zoom=zoom, front=front, lookat = lookat, up =up)
