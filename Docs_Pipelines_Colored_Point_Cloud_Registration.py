import open3d as o3d
import numpy as np
import copy

# http://www.open3d.org/docs/release/tutorial/pipelines/colored_pointcloud_registration.html

########################################################################################################################
# 1. Colored point cloud registration
########################################################################################################################
'''
This tutorial demonstrates an ICP variant that uses both geometry and color for registration. It implements the
algorithm of [Park2017]. The color information locks the alignment along the tangent plane. Thus this algorithm is more
accurate and more robust than prior point cloud registration algorithms, while the running speed is comparable to that
of ICP registration. This tutorial uses notations from ICP registration.
'''


########################################################################################################################
# 2. Helper visualization function
########################################################################################################################
def draw_registration_result_original_color(source, target, transformation):
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target],
                                      zoom=0.5, width=800, height=600,
                                      front=[-0.2458, -0.8088, 0.5342],
                                      lookat=[1.7745, 2.2305, 0.9787],
                                      up=[0.3109, -0.5878, -0.7468])


########################################################################################################################
# 3. Input
########################################################################################################################
print("1. Load two point clouds and show initial pose")
source = o3d.io.read_point_cloud("test_data/ColoredICP/frag_115.ply")
target = o3d.io.read_point_cloud("test_data/ColoredICP/frag_116.ply")

# draw initial alignment
current_transformation = np.identity(4)
draw_registration_result_original_color(source, target, current_transformation)

########################################################################################################################
# 4. Point-to-plane ICP
########################################################################################################################
'''
We first run Point-to-plane ICP as a baseline approach.
The visualization below shows misaligned green triangle textures. This is because a geometric constraint does not
prevent two planar surfaces from slipping.
'''
# point to plane ICP
current_transformation = np.identity(4)
print("2. Point-to-plane ICP registration is applied on original point")
print("   clouds to refine the alignment. Distance threshold 0.02.")
result_icp = o3d.pipelines.registration.registration_icp(
    source, target, 0.02, current_transformation,
    o3d.pipelines.registration.TransformationEstimationPointToPlane())
print(result_icp)
draw_registration_result_original_color(source, target, result_icp.transformation)

########################################################################################################################
# 5. Colored point cloud registration
########################################################################################################################
# colored pointcloud registration
# This is implementation of following paper
# J. Park, Q.-Y. Zhou, V. Koltun,
# Colored Point Cloud Registration Revisited, ICCV 2017
voxel_radius = [0.04, 0.02, 0.01]
max_iter = [50, 30, 14]
current_transformation = np.identity(4)
print("3. Colored point cloud registration")
for scale in range(3):
    iter = max_iter[scale]
    radius = voxel_radius[scale]
    print([iter, radius, scale])

    print("3-1. Downsample with a voxel size %.2f" % radius)
    source_down = source.voxel_down_sample(radius)
    target_down = target.voxel_down_sample(radius)

    print("3-2. Estimate normal.")
    source_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
    target_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

    print("3-3. Applying colored point cloud registration")
    result_icp = o3d.pipelines.registration.registration_colored_icp(
        source_down, target_down, radius, current_transformation,
        o3d.pipelines.registration.TransformationEstimationForColoredICP(),
        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                          relative_rmse=1e-6,
                                                          max_iteration=iter))
    current_transformation = result_icp.transformation
    print(result_icp)
draw_registration_result_original_color(source, target, result_icp.transformation)
