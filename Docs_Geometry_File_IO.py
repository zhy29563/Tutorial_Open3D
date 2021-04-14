import open3d as o3d

# http://www.open3d.org/docs/release/tutorial/geometry/file_io.html

########################################################################################################################
# 1. Point cloud
########################################################################################################################
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
print("Testing IO for point cloud ...")
pcd = o3d.io.read_point_cloud("test_data/fragment.pcd")
print(pcd)
o3d.io.write_point_cloud("copy_of_fragment.pcd", pcd)

########################################################################################################################
# 2. Mesh
########################################################################################################################
'''
By default, Open3D tries to infer the file type by the filename extension.
The following mesh file types are supported:

    Format          Description
    ply             See Polygon File Format, the ply file can contain both point cloud and mesh data
    stl             See StereoLithography
    obj             See Object Files
    off             See Object File Format
    gltf/glb        See GL Transmission Format
'''
print("Testing IO for meshes ...")
mesh = o3d.io.read_triangle_mesh("test_data/knot.ply")
print(mesh)
o3d.io.write_triangle_mesh("copy_of_knot.ply", mesh)

########################################################################################################################
# 3. Image
########################################################################################################################
'''Both jpg and png image files are supported.'''
print("Testing IO for images ...")
img = o3d.io.read_image("test_data/lena_color.jpg")
print(img)
o3d.io.write_image("copy_of_lena_color.jpg", img)
