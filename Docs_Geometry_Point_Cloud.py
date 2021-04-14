import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import python.open3d_tutorial as o3dtut

# http://www.open3d.org/docs/release/tutorial/geometry/pointcloud.html

########################################################################################################################
# 1. Visualize point cloud
########################################################################################################################
print("Load a ply point cloud, print it, and render it")
'''
reads a point cloud from a file. It tries to decode the file based on the extension name
Format        Description
xyz           Each line contains [x, y, z], where x, y, z are the 3D coordinates
xyzn          Each line contains [x, y, z, nx, ny, nz], where nx, ny, nz are the normals
xyzrgb        Each line contains [x, y, z, r, g, b], where r, g, b are in floats of range [0, 1]
pts           The first line is an integer representing the number of points. Each subsequent line contains [x, y, z, i,
              r, g, b], where r, g, b are in uint8
ply           See Polygon File Format, the ply file can contain both point cloud and mesh data
              (http://paulbourke.net/dataformats/ply/)
pcd           See Point Cloud Data(http://pointclouds.org/documentation/tutorials/pcd_file_format.html)

It’s also possible to specify the file type explicitly. In this case, the file extension will be ignored.
pcd = o3d.io.read_point_cloud("../../test_data/my_points.txt", format='xyz')
'''
pcd = o3d.io.read_point_cloud("test_data/fragment.ply")

print(pcd)
print(np.asarray(pcd.points))
# visualizes the point cloud
o3d.visualization.draw_geometries([pcd],
                                  zoom=0.3412,
                                  width=800, height=600,
                                  window_name='Visualize point cloud',
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])

########################################################################################################################
# 2. Voxel downsampling
########################################################################################################################
'''
Voxel downsampling uses a regular voxel grid to create a uniformly downsampled point cloud
from an input point cloud. It is often used as a pre-processing step for many point cloud
processing tasks. The algorithm operates in two steps:
- Points are bucketed into voxels.
- Each occupied voxel generates exactly one point by averaging all points inside.
'''
print("Downsample the point cloud with a voxel of 0.05")
downpcd = pcd.voxel_down_sample(voxel_size=0.05)
o3d.visualization.draw_geometries([downpcd],
                                  zoom=0.3412,
                                  width=800, height=600,
                                  window_name='Voxel downsampling',
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])
########################################################################################################################
# 3. Vertex normal estimation
########################################################################################################################
'''
Another basic operation for point cloud is point normal estimation.
Press N to see point normals.
The keys - and + can be used to control the length of the normal.
'''
print("Recompute the normal of the downsampled point cloud")
'''
estimate_normals: computes the normal for every point.
The function finds adjacent points and calculates the principal axis of the adjacent points using covariance analysis.
The function takes an instance of KDTreeSearchParamHybrid class as an argument.
The two key arguments radius = 0.1 and max_nn = 30 specifies search radius and maximum nearest neighbor.
It has 10cm of search radius, and only considers up to 30 neighbors to save computation time.

The covariance analysis algorithm produces two opposite directions as normal candidates.
Without knowing the global structure of the geometry, both can be correct.
This is known as the normal orientation problem.
Open3D tries to orient the normal to align with the original normal if it exists.
Otherwise, Open3D does a random guess.
Further orientation functions such as orient_normals_to_align_with_direction and orient_normals_towards_camera_location
need to be called if the orientation is a concern.
'''
downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
o3d.visualization.draw_geometries([downpcd],
                                  zoom=0.3412,
                                  width=800, height=600,
                                  window_name='Vertex normal estimation',
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024],
                                  point_show_normal=True)
########################################################################################################################
# 4. Access estimated vertex normal
########################################################################################################################
print("Print a normal vector of the 0th point")
print(downpcd.normals[0])

# Normal vectors can be transformed as a numpy array using np.asarray.
print("Print the normal vectors of the first 10 points")
print(np.asarray(downpcd.normals)[:10, :])

########################################################################################################################
# 5. Crop point cloud
########################################################################################################################
print("Load a polygon volume and use it to crop the original point cloud")
'''
read_selection_polygon_volume reads a json file that specifies polygon selection area.
vol.crop_point_cloud(pcd) filters out points. Only the chair remains.
'''
vol = o3d.visualization.read_selection_polygon_volume("test_data/cropped.json")
chair = vol.crop_point_cloud(pcd)
o3d.visualization.draw_geometries([chair],
                                  zoom=0.7,
                                  width=800, height=600,
                                  window_name='Crop point cloud',
                                  front=[0.5439, -0.2333, -0.8060],
                                  lookat=[2.4615, 2.1331, 1.338],
                                  up=[-0.1781, -0.9708, 0.1608])

########################################################################################################################
# 6. Paint point cloud
########################################################################################################################
print("Paint chair")
# paint_uniform_color paints all the points to a uniform color. The color is in RGB space, [0, 1] range.
chair.paint_uniform_color([1, 0.706, 0])
o3d.visualization.draw_geometries([chair],
                                  zoom=0.7,
                                  width=800, height=600,
                                  window_name='Paint point cloud',
                                  front=[0.5439, -0.2333, -0.8060],
                                  lookat=[2.4615, 2.1331, 1.338],
                                  up=[-0.1781, -0.9708, 0.1608])

########################################################################################################################
# 7. Point cloud distance
########################################################################################################################
'''
Open3D provides the method compute_point_cloud_distance to compute the distance from a source point cloud to a
target point cloud. I.e., it computes for each point in the source point cloud the distance to the closest point in the
target point cloud.
'''
dists = pcd.compute_point_cloud_distance(chair)
dists = np.asarray(dists)
ind = np.where(dists > 0.01)[0]
pcd_without_chair = pcd.select_by_index(ind)
o3d.visualization.draw_geometries([pcd_without_chair],
                                  zoom=0.3412,
                                  width=800, height=600,
                                  window_name='Point cloud distance',
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024])

########################################################################################################################
# 8. Bounding volumes
########################################################################################################################
'''
The PointCloud geometry type has bounding volumes as all other geometry types in Open3D. Currently, Open3D implements an
AxisAlignedBoundingBox and an OrientedBoundingBox that can also be used to crop the geometry.
'''
aabb = chair.get_axis_aligned_bounding_box()
aabb.color = (1, 0, 0)
obb = chair.get_oriented_bounding_box()
obb.color = (0, 1, 0)
o3d.visualization.draw_geometries([chair, aabb, obb],
                                  zoom=0.7,
                                  width=800, height=600,
                                  window_name='Bounding volumes',
                                  front=[0.5439, -0.2333, -0.8060],
                                  lookat=[2.4615, 2.1331, 1.338],
                                  up=[-0.1781, -0.9708, 0.1608])

########################################################################################################################
# 9. Convex hull
########################################################################################################################
'''
The convex hull of a point cloud is the smallest convex set that contains all points. Open3D contains the method
compute_convex_hull that computes the convex hull of a point cloud. The implementation is based on
Qhull(http://www.qhull.org/).
'''
pcl = o3dtut.get_bunny_mesh().sample_points_poisson_disk(number_of_points=2000)
hull, _ = pcl.compute_convex_hull()
hull_ls = o3d.geometry.LineSet.create_from_triangle_mesh(hull)
hull_ls.paint_uniform_color((1, 0, 0))
o3d.visualization.draw_geometries([pcl, hull_ls],
                                  width=800, height=600,
                                  window_name='Convex hull')

########################################################################################################################
# 10. DBSCAN clustering
########################################################################################################################
'''
Given a point cloud from e.g. a depth sensor we want to group local point cloud clusters together. For this purpose, we
can use clustering algorithms. Open3D implements DBSCAN [Ester1996] that is a density based clustering algorithm. The
algorithm is implemented in cluster_dbscan and requires two parameters: eps defines the distance to neighbors in a
cluster and min_points defines the minimum number of points required to form a cluster. The function returns labels,
where the label -1 indicates noise.
'''
pcd = o3d.io.read_point_cloud("test_data/fragment.ply")
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))

max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
o3d.visualization.draw_geometries([pcd],
                                  zoom=0.455,
                                  width=800, height=600,
                                  window_name='DBSCAN clustering',
                                  front=[-0.4999, -0.1659, -0.8499],
                                  lookat=[2.1813, 2.0619, 2.0999],
                                  up=[0.1204, -0.9852, 0.1215])


########################################################################################################################
# 11. Plane segmentation
########################################################################################################################
'''
Open3D also supports segmentation of geometric primitives from point clouds using RANSAC.
To find the plane with the largest support in the point cloud, we can use segment_plane.
The method has three arguments: distance_threshold defines the maximum distance a point can have to an estimated plane
to be considered an inlier, ransac_n defines the number of points that are randomly sampled to estimate a plane, and
num_iterations defines how often a random plane is sampled and verified.

The function then returns the plane as (a,b,c,d) such that for each point (x,y,z) on the plane we have ax+by+cz+d=0.
The function further returns a list of indices of the inlier points.
'''
pcd = o3d.io.read_point_cloud("test_data/fragment.pcd")
plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                         ransac_n=3,
                                         num_iterations=1000)
[a, b, c, d] = plane_model
print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

inlier_cloud = pcd.select_by_index(inliers)
inlier_cloud.paint_uniform_color([1.0, 0, 0])
outlier_cloud = pcd.select_by_index(inliers, invert=True)
o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                  zoom=0.8,
                                  width=800, height=600,
                                  window_name='Plane segmentation',
                                  front=[-0.4999, -0.1659, -0.8499],
                                  lookat=[2.1813, 2.0619, 2.0999],
                                  up=[0.1204, -0.9852, 0.1215])


########################################################################################################################
# 12. Hidden point removal
########################################################################################################################
'''
Imagine you want to render a point cloud from a given view point, but points from the background leak into the
foreground because they are not occluded by other points. For this purpose we can apply a hidden point removal
algorithm. In Open3D the method by [Katz2007] is implemented that approximates the visibility of a point cloud from a
given view without surface reconstruction or normal estimation.
'''

print("Convert mesh to a point cloud and estimate dimensions")
pcd = o3dtut.get_armadillo_mesh().sample_points_poisson_disk(5000)
diameter = np.linalg.norm(
    np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
o3d.visualization.draw_geometries([pcd],
                                  width=800, height=600,
                                  window_name='Hidden point removal (Before)')

print("Define parameters used for hidden_point_removal")
camera = [0, 0, diameter]
radius = diameter * 100

print("Get all points that are visible from given view point")
_, pt_map = pcd.hidden_point_removal(camera, radius)

print("Visualize result")
pcd = pcd.select_by_index(pt_map)
o3d.visualization.draw_geometries([pcd],
                                  width=800, height=600,
                                  window_name='Hidden point removal (After)')
