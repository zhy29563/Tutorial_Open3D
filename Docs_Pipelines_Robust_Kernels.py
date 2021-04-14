import open3d as o3d
import numpy as np
import copy

# http://www.open3d.org/docs/release/tutorial/pipelines/robust_kernels.html

########################################################################################################################
# 1. Robust kernels
########################################################################################################################
'''
This tutorial demonstrates the use of robust kernels in the context of outlier rejection. For this particular tutorial,
we will be using the ICP (Iterative Closest Point) registration algorithm as the target problem where we want to deal
with outliers. Even so, the theory applies to any given optimization problem and not just for ICP. For the moment the
robust kernels have been only implemented for the PointToPlane ICP.

The notation and some of the kernels implemented in Open3D has been inspired by the publication “Analysis of Robust
Functions for Registration Algorithms” [Babin2019]
'''


########################################################################################################################
# 2. Robust kernels
########################################################################################################################
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459, width=800, height=600,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])


source = o3d.io.read_point_cloud("test_data/ICP/cloud_bin_0.pcd")
target = o3d.io.read_point_cloud("test_data/ICP/cloud_bin_1.pcd")
trans_init = np.asarray([[0.862, 0.011, -0.507, 0.5],
                         [-0.139, 0.967, -0.215, 0.7],
                         [0.487, 0.255, 0.835, -1.4],
                         [0.0, 0.0, 0.0, 1.0]])
draw_registration_result(source, target, trans_init)


########################################################################################################################
# 3. Vanilla ICP vs Robust ICP
########################################################################################################################
def apply_noise(pcd, mu, sigma):
    noisy_pcd = copy.deepcopy(pcd)
    points = np.asarray(noisy_pcd.points)
    points += np.random.normal(mu, sigma, size=points.shape)
    noisy_pcd.points = o3d.utility.Vector3dVector(points)
    return noisy_pcd


mu, sigma = 0, 0.1  # mean and standard deviation
source_noisy = apply_noise(source, mu, sigma)

print("Source PointCloud + noise:")
o3d.visualization.draw_geometries([source_noisy],
                                  zoom=0.4459, width=800, height=600,
                                  front=[0.353, -0.469, -0.809],
                                  lookat=[2.343, 2.217, 1.809],
                                  up=[-0.097, -0.879, 0.467])


# Vanilla ICP
threshold = 0.02
print("Vanilla point-to-plane ICP, threshold={}:".format(threshold))
p2l = o3d.pipelines.registration.TransformationEstimationPointToPlane()
reg_p2l = o3d.pipelines.registration.registration_icp(source_noisy, target, threshold, trans_init, p2l)

print(reg_p2l)
print("Transformation is:")
print(reg_p2l.transformation)
draw_registration_result(source, target, reg_p2l.transformation)

# Tuning Vanilla ICP
threshold = 1.0
print("Vanilla point-to-plane ICP, threshold={}:".format(threshold))
p2l = o3d.pipelines.registration.TransformationEstimationPointToPlane()
reg_p2l = o3d.pipelines.registration.registration_icp(source_noisy, target, threshold, trans_init, p2l)

print(reg_p2l)
print("Transformation is:")
print(reg_p2l.transformation)
draw_registration_result(source, target, reg_p2l.transformation)

# Robust ICP
print("Robust point-to-plane ICP, threshold={}:".format(threshold))
loss = o3d.pipelines.registration.TukeyLoss(k=sigma)
print("Using robust loss:", loss)
p2l = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
reg_p2l = o3d.pipelines.registration.registration_icp(source_noisy, target,
                                                      threshold, trans_init,
                                                      p2l)
print(reg_p2l)
print("Transformation is:")
print(reg_p2l.transformation)
draw_registration_result(source, target, reg_p2l.transformation)
