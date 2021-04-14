import open3d as o3d
import numpy as np

# http://www.open3d.org/docs/release/tutorial/pipelines/rgbd_odometry.html

########################################################################################################################
# 1. Odometry
########################################################################################################################
'''
https://en.wikipedia.org/wiki/Odometry

Odometry is the use of data from motion sensors to estimate change in position over time. It is used in robotics by some
legged or wheeled robots to estimate their position relative to a starting location. This method is sensitive to errors
due to the integration of velocity measurements over time to give position estimates. Rapid and accurate data
collection, instrument calibration, and processing are required in most cases for odometry to be used effectively.

The word odometry is composed from the Greek words odos (meaning "route") and metron (meaning "measure").
'''
########################################################################################################################
# 2. RGBD Odometry
########################################################################################################################
'''
An RGBD odometry finds the camera movement between two consecutive RGBD image pairs. The input are two instances of
RGBDImage. The output is the motion in the form of a rigid body transformation. Open3D implements the method of
[Steinbrucker2011] and [Park2017].
'''
########################################################################################################################
# 3. Read camera intrinsic
########################################################################################################################
# We first read the camera intrinsic matrix from a json file.
pinhole_camera_intrinsic = o3d.io.read_pinhole_camera_intrinsic("test_data/camera_primesense.json")
print(pinhole_camera_intrinsic.intrinsic_matrix)
########################################################################################################################
# 4. Read RGBD image
########################################################################################################################
# This code block reads two pairs of RGBD images in the Redwood format. We refer to Redwood dataset for a comprehensive
# explanation.
source_color = o3d.io.read_image("test_data/RGBD/color/00000.jpg")
source_depth = o3d.io.read_image("test_data/RGBD/depth/00000.png")
target_color = o3d.io.read_image("test_data/RGBD/color/00001.jpg")
target_depth = o3d.io.read_image("test_data/RGBD/depth/00001.png")
source_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(source_color, source_depth)
target_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(target_color, target_depth)
target_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(target_rgbd_image, pinhole_camera_intrinsic)
########################################################################################################################
# 5. Compute odometry from two RGBD image pairs
########################################################################################################################
option = o3d.pipelines.odometry.OdometryOption()
odo_init = np.identity(4)
print(option)

[success_color_term, trans_color_term, info] = o3d.pipelines.odometry.compute_rgbd_odometry(
     source_rgbd_image, target_rgbd_image, pinhole_camera_intrinsic, odo_init,
     o3d.pipelines.odometry.RGBDOdometryJacobianFromColorTerm(), option)
[success_hybrid_term, trans_hybrid_term, info] = o3d.pipelines.odometry.compute_rgbd_odometry(
     source_rgbd_image, target_rgbd_image, pinhole_camera_intrinsic, odo_init,
     o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option)
'''
This code block calls two different RGBD odometry methods. The first one is from [Steinbrucker2011]. It minimizes photo
consistency of aligned images. The second one is from [Park2017]. In addition to photo consistency, it implements
constraint for geometry. Both functions run in similar speed, but [Park2017] is more accurate in our test on benchmark
datasets and is thus the recommended method.

Several parameters in OdometryOption():
    1. minimum_correspondence_ratio: After alignment, measure the overlapping ratio of two RGBD images. If overlapping
      region of two RGBD image is smaller than specified ratio, the odometry module regards that this is a failure case.
    2. max_depth_diff: In depth image domain, if two aligned pixels have a depth difference less than specified value,
       they are considered as a correspondence. Larger value induce more aggressive search, but it is prone to unstable
       result.
    3. min_depth and max_depth: Pixels that has smaller or larger than specified depth values are ignored.
'''
########################################################################################################################
# 6. Visualize RGBD image pairs
########################################################################################################################
'''
The RGBD image pairs are converted into point clouds and rendered together. Note that the point cloud representing the
first (source) RGBD image is transformed with the transformation estimated by the odometry. After this transformation,
both point clouds are aligned.
'''
if success_color_term:
    print("Using RGB-D Odometry")
    print(trans_color_term)
    source_pcd_color_term = o3d.geometry.PointCloud.create_from_rgbd_image(
        source_rgbd_image, pinhole_camera_intrinsic)
    source_pcd_color_term.transform(trans_color_term)
    o3d.visualization.draw_geometries([target_pcd, source_pcd_color_term],
                                      zoom=0.48,
                                      front=[0.0999, -0.1787, -0.9788],
                                      lookat=[0.0345, -0.0937, 1.8033],
                                      up=[-0.0067, -0.9838, 0.1790], width=800, height=600)
if success_hybrid_term:
    print("Using Hybrid RGB-D Odometry")
    print(trans_hybrid_term)
    source_pcd_hybrid_term = o3d.geometry.PointCloud.create_from_rgbd_image(source_rgbd_image, pinhole_camera_intrinsic)
    source_pcd_hybrid_term.transform(trans_hybrid_term)
    o3d.visualization.draw_geometries([target_pcd, source_pcd_hybrid_term],
                                      zoom=0.48,
                                      front=[0.0999, -0.1787, -0.9788],
                                      lookat=[0.0345, -0.0937, 1.8033],
                                      up=[-0.0067, -0.9838, 0.1790], width=800, height=600)



