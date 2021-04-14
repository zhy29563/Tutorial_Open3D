from os.path import join

import open3d as o3d
import numpy as np
import os
import re
import python.open3d_tutorial as o3dtut

# http://www.open3d.org/docs/release/tutorial/pipelines/rgbd_odometry.html

########################################################################################################################
# 1. Color Map Optimization
########################################################################################################################
'''
Consider color mapping to the geometry reconstructed from depth cameras. As color and depth frames are not perfectly
aligned, the texture mapping using color images is subject to results in blurred color map. Open3D provides color map
optimization method proposed by [Zhou2014]. The following script shows an example of color map optimization.
'''
########################################################################################################################
# 2. Input
########################################################################################################################
'''
This code below reads color and depth image pairs and makes rgbd_image. Note that convert_rgb_to_intensity flag is
False. This is to preserve 8-bit color channels instead of using single channel float type image.

It is always good practice to visualize the RGBD image before applying it to the color map optimization. The debug_mode
switch can be set to True to visualize the RGBD image.
'''
def sorted_alphanum(file_list_ordered):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(file_list_ordered, key=alphanum_key)


def get_file_list(path, extension=None):
    if extension is None:
        file_list = [
            path + f for f in os.listdir(path) if os.path.isfile(join(path, f))
        ]
    else:
        file_list = [
            path + f
            for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f)) and
            os.path.splitext(f)[1] == extension
        ]
    file_list = sorted_alphanum(file_list)
    return file_list

path = o3dtut.download_fountain_dataset()
debug_mode = False

rgbd_images = []
depth_image_path = get_file_list(os.path.join(path, "depth/"), extension=".png")
color_image_path = get_file_list(os.path.join(path, "image/"), extension=".jpg")
assert (len(depth_image_path) == len(color_image_path))
for i in range(len(depth_image_path)):
    depth = o3d.io.read_image(os.path.join(depth_image_path[i]))
    color = o3d.io.read_image(os.path.join(color_image_path[i]))
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, convert_rgb_to_intensity=False)
    if debug_mode:
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,
            o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
        o3d.visualization.draw_geometries([pcd])
    rgbd_images.append(rgbd_image)


camera = o3d.io.read_pinhole_camera_trajectory(os.path.join(path, "scene/key.log"))
mesh = o3d.io.read_triangle_mesh(os.path.join(path, "scene", "integrated.ply"))

'''
To visualize how the camera poses are not good for color mapping, this code intentionally sets the iteration number to
0, which means no optimization. color_map_optimization paints a mesh using corresponding RGBD images and camera poses.
Without optimization, the texture map is blurred.
'''
# Before full optimization, let's just visualize texture map
# with given geometry, RGBD images, and camera poses.
option = o3d.pipelines.color_map.ColorMapOptimizationOption()
option.maximum_iteration = 0
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    o3d.pipelines.color_map.color_map_optimization(mesh, rgbd_images, camera, option)

o3d.visualization.draw_geometries([mesh],
                                  zoom=0.5399, width=800, height=600,
                                  front=[0.0665, -0.1107, -0.9916],
                                  lookat=[0.7353, 0.6537, 1.0521],
                                  up=[0.0136, -0.9936, 0.1118])

########################################################################################################################
# 3. Rigid Optimization
########################################################################################################################
# The next step is to optimize camera poses to get a sharp color map.
#
# The code below sets maximum_iteration = 300 for actual iterations.
# Optimize texture and save the mesh as texture_mapped.ply
# This is implementation of following paper
# Q.-Y. Zhou and V. Koltun,
# Color Map Optimization for 3D Reconstruction with Consumer Depth Cameras,
# SIGGRAPH 2014
is_ci = True
option.maximum_iteration = 100 if is_ci else 300
option.non_rigid_camera_coordinate = False
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    o3d.pipelines.color_map.color_map_optimization(mesh, rgbd_images, camera,option)

o3d.visualization.draw_geometries([mesh],
                                  zoom=0.5399, width=800, height=600,
                                  front=[0.0665, -0.1107, -0.9916],
                                  lookat=[0.7353, 0.6537, 1.0521],
                                  up=[0.0136, -0.9936, 0.1118])

# The residual error implies inconsistency of image intensities. Lower residual leads to a better color map quality.
# By default, ColorMapOptimizationOption enables rigid optimization. It optimizes 6-dimentional pose of every cameras.

########################################################################################################################
# 4. Non-rigid Optimization
########################################################################################################################
'''
For better alignment quality, there is an option for non-rigid optimization. To enable this option, simply set
option.non_rigid_camera_coordinate to True before calling color_map_optimization. Besides 6-dimentional camera poses,
non-rigid optimization even considers local image warping represented by anchor points. This adds even more flexibility
and leads to an even higher quality color mapping. The residual error is smaller than the case of rigid optimization.
'''
option.maximum_iteration = 100 if is_ci else 300
option.non_rigid_camera_coordinate = True
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    o3d.pipelines.color_map.color_map_optimization(mesh, rgbd_images, camera, option)


