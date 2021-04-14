import open3d as o3d
import numpy as np
import python.open3d_tutorial as o3dtut
import copy

# http://www.open3d.org/docs/release/tutorial/geometry/mesh.html

########################################################################################################################
# 1. Mesh
########################################################################################################################
'''
Open3D has a data structure for 3D triangle meshes called TriangleMesh.
The code below shows how to read a triangle mesh from a ply file and print its vertices and triangles.

The TriangleMesh class has a few data fields such as vertices and triangles.
Open3D provides direct memory access to these fields via numpy.
'''
print("Testing mesh in Open3D...")
mesh = o3dtut.get_knot_mesh()
print(mesh)
print('Vertices:')
print(np.asarray(mesh.vertices))
print('Triangles:')
print(np.asarray(mesh.triangles))

########################################################################################################################
# 2. Visualize a 3D mesh
########################################################################################################################
print("Try to render a mesh with normals (exist: " +
      str(mesh.has_vertex_normals()) + ") and colors (exist: " +
      str(mesh.has_vertex_colors()) + ")")
o3d.visualization.draw_geometries([mesh],
                                  width=800, height=600,
                                  window_name='Visualize a 3D mesh')
print("A mesh with no normals and no colors does not look good.")
'''
You can rotate and move the mesh but it is painted with uniform gray color and does not look “3d”.
The reason is that the current mesh does not have normals for vertices or faces.
So uniform color shading is used instead of a more sophisticated Phong shading.
'''

########################################################################################################################
# 3. Surface normal estimation
########################################################################################################################
print("Computing normal and rendering it.")
mesh.compute_vertex_normals()
print(np.asarray(mesh.triangle_normals))
o3d.visualization.draw_geometries([mesh],
                                  width=800, height=600,
                                  window_name='Surface normal estimation')

########################################################################################################################
# 4. Crop mesh
########################################################################################################################
'''
We remove half of the surface by directly operating on the triangle and triangle_normals data fields of the mesh.
This is done via numpy.
'''
print("We make a partial mesh of only the first half triangles.")
mesh1 = copy.deepcopy(mesh)
mesh1.triangles = o3d.utility.Vector3iVector(np.asarray(mesh1.triangles)[:len(mesh1.triangles) // 2, :])
mesh1.triangle_normals = o3d.utility.Vector3dVector(
    np.asarray(mesh1.triangle_normals)[:len(mesh1.triangle_normals) // 2, :])
print(mesh1.triangles)
o3d.visualization.draw_geometries([mesh1],
                                  width=800, height=600,
                                  window_name='Crop mesh')

########################################################################################################################
# 5. Paint mesh
########################################################################################################################
# paint_uniform_color paints the mesh with a uniform color. The color is in RGB space, [0, 1] range.
print("Painting the mesh")
mesh1.paint_uniform_color([1, 0.706, 0])
o3d.visualization.draw_geometries([mesh1],
                                  width=800, height=600,
                                  window_name='Paint mesh')

########################################################################################################################
# 6. Mesh properties
########################################################################################################################
'''
https://zh.wikipedia.org/wiki/%E6%B5%81%E5%BD%A2
流形（英語：Manifolds）是可以局部欧几里得空间化的一个拓扑空间，是欧几里得空间中的曲线、曲面等概念的推广。欧几里得空间就是最简单的流
形的实例。

地球表面这样的球面则是一个稍微复杂的例子。一般的流形可以通过把许多平直的片折弯并粘连而成。

流形在数学中用于描述几何形体，它们为研究形体的可微性提供了一个自然的平台。物理上，经典力学的相空间和构造广义相对论的时空模型的四维伪
黎曼流形都是流形的实例。位形空间中也可以定义流形。环面就是双摆的位形空间。

一般可以把几何形体的拓扑结构看作是完全“柔软”的，因为所有变形（同胚）会保持拓扑结构不变；而把解析几何结构看作是“硬”的，因为整体的结构
都是固定的。例如，当一个多项式在 {displaystyle (0,1)}(0,1) 区间的取值确定了，则其在整个实数范围的值都被固定，可见局部的变动会导致
全局的变化。光滑流形可以看作是介于两者之间的模型：其无穷小的结构是“硬”的，而整体结构则是“柔软”的。这也许是中文译名“流形”的原因（整体
的形态可以流动）。该译名由著名数学家和数学教育学家江泽涵引入。这样，流形的硬度使它能够容纳微分结构，而它的软度使得它可以作为很多需要
独立的局部扰动的数学和物理的模型。

A triangle mesh has several properties that can be tested with Open3D. One important property is the manifold property,
where we can test the triangle mesh if it is edge manifold is_edge_manifold and if it is is_vertex_manifold. A triangle
mesh is edge manifold, if each edge is bounding either one or two triangles. The function is_edge_manifold has the bool
parameter allow_boundary_edges that defines if boundary edges should be allowed. Further, a triangle mesh is vertex
manifold if the star of the vertex is edge‐manifold and edge‐connected, e.g., two or more faces connected only by a
vertex and not by an edge.

Another property is the test of self-intersection.
The function is_self_intersecting returns True if there exists a triangle in the mesh that is intersecting another mesh.
A watertight mesh can be defined as a mesh that is edge manifold, vertex manifold and not self intersecting. The
function is_watertight implements this check in Open3D.

We also can test the triangle mesh, if it is orientable, i.e. the triangles can be oriented in such a way that all
normals point towards the outside. The corresponding function in Open3D is called is_orientable.

The code below tests a number of triangle meshes against those properties and visualizes the results. Non-manifold
edges are shown in red, boundary edges in green, non-manifold vertices are visualized as green points, and
self-intersecting triangles are shown in pink.
'''


def check_properties(name, mesh):
    mesh.compute_vertex_normals()

    edge_manifold = mesh.is_edge_manifold(allow_boundary_edges=True)
    edge_manifold_boundary = mesh.is_edge_manifold(allow_boundary_edges=False)
    vertex_manifold = mesh.is_vertex_manifold()
    self_intersecting = mesh.is_self_intersecting()
    watertight = mesh.is_watertight()
    orientable = mesh.is_orientable()

    print(name)
    print(f"  edge_manifold:          {edge_manifold}")
    print(f"  edge_manifold_boundary: {edge_manifold_boundary}")
    print(f"  vertex_manifold:        {vertex_manifold}")
    print(f"  self_intersecting:      {self_intersecting}")
    print(f"  watertight:             {watertight}")
    print(f"  orientable:             {orientable}")

    geoms = [mesh]
    if not edge_manifold:
        edges = mesh.get_non_manifold_edges(allow_boundary_edges=True)
        geoms.append(o3dtut.edges_to_lineset(mesh, edges, (1, 0, 0)))
    if not edge_manifold_boundary:
        edges = mesh.get_non_manifold_edges(allow_boundary_edges=False)
        geoms.append(o3dtut.edges_to_lineset(mesh, edges, (0, 1, 0)))
    if not vertex_manifold:
        verts = np.asarray(mesh.get_non_manifold_vertices())
        pcl = o3d.geometry.PointCloud(
            points=o3d.utility.Vector3dVector(np.asarray(mesh.vertices)[verts]))
        pcl.paint_uniform_color((0, 0, 1))
        geoms.append(pcl)
    if self_intersecting:
        intersecting_triangles = np.asarray(mesh.get_self_intersecting_triangles())
        intersecting_triangles = intersecting_triangles[0:1]
        intersecting_triangles = np.unique(intersecting_triangles)
        print("  # visualize self-intersecting triangles")
        triangles = np.asarray(mesh.triangles)[intersecting_triangles]
        edges = [
            np.vstack((triangles[:, i], triangles[:, j]))
            for i, j in [(0, 1), (1, 2), (2, 0)]
        ]
        edges = np.hstack(edges).T
        edges = o3d.utility.Vector2iVector(edges)
        geoms.append(o3dtut.edges_to_lineset(mesh, edges, (1, 0, 1)))
    o3d.visualization.draw_geometries(geoms, mesh_show_back_face=True,
                                      width=800, height=600,
                                      window_name='Mesh properties: ' + name)


check_properties('Knot', o3dtut.get_knot_mesh())
check_properties('Moebius', o3d.geometry.TriangleMesh.create_moebius(twists=1))
check_properties("non-manifold edge", o3dtut.get_non_manifold_edge_mesh())
check_properties("non-manifold vertex", o3dtut.get_non_manifold_vertex_mesh())
check_properties("open box", o3dtut.get_open_box_mesh())
check_properties("intersecting_boxes", o3dtut.get_intersecting_boxes_mesh())
########################################################################################################################
# 7. Mesh filtering - Average filter
########################################################################################################################
'''
The simplest filter is the average filter. A given vertex vi is given by the average of the adjacent vertices N
This filter can be used to denoise meshes as demonstrated in the code below. The parameter number_of_iterations in the
function filter_smooth_simple defines the how often the filter is applied to the mesh.
'''
print('create noisy mesh')
mesh_in = o3dtut.get_knot_mesh()
vertices = np.asarray(mesh_in.vertices)
noise = 5
vertices += np.random.uniform(0, noise, size=vertices.shape)
mesh_in.vertices = o3d.utility.Vector3dVector(vertices)
mesh_in.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh_in],
                                  width=800, height=600,
                                  window_name='Average filter with noise')

print('filter with average with 1 iteration')
mesh_out = mesh_in.filter_smooth_simple(number_of_iterations=1)
mesh_out.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh_out],
                                  width=800, height=600,
                                  window_name='Average filter with noise 1 iteration')

print('filter with average with 5 iterations')
mesh_out = mesh_in.filter_smooth_simple(number_of_iterations=5)
mesh_out.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh_out],
                                  width=800, height=600,
                                  window_name='Average filter with noise 5 iteration')
########################################################################################################################
# 8. Mesh filtering - Laplacian
########################################################################################################################
'''
Another important mesh filter is the Laplacian defined as vi=vi⋅λ∑wnvn−vi,

where λ is the strength of the filter and wn are normalized weights that relate to the distance of the neighboring
vertices. The filter is implemented in filter_smooth_laplacian and has the parameters number_of_iterations and lambda.
'''
print('filter with Laplacian with 10 iterations')
mesh_out = mesh_in.filter_smooth_laplacian(number_of_iterations=10)
mesh_out.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh_out],
                                  width=800, height=600,
                                  window_name='Laplacian with noise 10 iteration')

print('filter with Laplacian with 50 iterations')
mesh_out = mesh_in.filter_smooth_laplacian(number_of_iterations=50)
mesh_out.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh_out],
                                  width=800, height=600,
                                  window_name='Laplacian with noise 50 iteration')

########################################################################################################################
# 9. Mesh filtering - Taubin filter
########################################################################################################################
'''
The problem with the average and Laplacian filter is that they lead to a shrinkage of the triangle mesh. [Taubin1995]
showed that the application of two Laplacian filters with different λ parameters can prevent the mesh shrinkage. The
filter is implemented in filter_smooth_taubin.
'''
print('filter with Taubin with 10 iterations')
mesh_out = mesh_in.filter_smooth_taubin(number_of_iterations=10)
mesh_out.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh_out],
                                  width=800, height=600,
                                  window_name='Taubin with noise 10 iteration')

print('filter with Taubin with 100 iterations')
mesh_out = mesh_in.filter_smooth_taubin(number_of_iterations=100)
mesh_out.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh_out],
                                  width=800, height=600,
                                  window_name='Taubin with noise 100 iteration')

########################################################################################################################
# 10. Sampling
########################################################################################################################
'''
Open3D includes functions to sample point clouds from a triangle mesh. The simplest method is sample_points_uniformly
that uniformly samples points from the 3D surface based on the triangle area. The parameter number_of_points defines
how many points are sampled from the triangle surface.
'''
mesh = o3d.geometry.TriangleMesh.create_sphere()
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh],
                                  width=800, height=600,
                                  window_name='Mesh with Normal')

pcd = mesh.sample_points_uniformly(number_of_points=500)
o3d.visualization.draw_geometries([pcd],
                                  width=800, height=600,
                                  window_name='Meshed Point Cloud')

mesh = o3dtut.get_bunny_mesh()
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh],
                                  width=800, height=600,
                                  window_name='Mesh with Normal')

pcd = mesh.sample_points_uniformly(number_of_points=500)
o3d.visualization.draw_geometries([pcd],
                                  width=800, height=600,
                                  window_name='Meshed Point Cloud')
'''
Uniform sampling can yield clusters of points on the surface, while a method called Poisson disk sampling can evenly
distribute the points on the surface. The method sample_points_poisson_disk implements sample elimination. It starts
with a sampled point cloud and removes points to satisfy the sampling criterion. The method supports two options to
provide the initial point cloud:
1. Default via the parameter init_factor: The method first samples uniformly a point cloud from the mesh with
   init_factor x number_of_points and uses this for the elimination.
2. One can provide a point cloud and pass it to the sample_points_poisson_disk method. Then, this point cloud is used
for elimination.
'''
mesh = o3d.geometry.TriangleMesh.create_sphere()
pcd = mesh.sample_points_poisson_disk(number_of_points=500, init_factor=5)
o3d.visualization.draw_geometries([pcd],
                                  width=800, height=600,
                                  window_name='Mesh with Normal')

pcd = mesh.sample_points_uniformly(number_of_points=2500)
pcd = mesh.sample_points_poisson_disk(number_of_points=500, pcl=pcd)
o3d.visualization.draw_geometries([pcd],
                                  width=800, height=600,
                                  window_name='Meshed Point Cloud')

mesh = o3dtut.get_bunny_mesh()
pcd = mesh.sample_points_poisson_disk(number_of_points=500, init_factor=5)
o3d.visualization.draw_geometries([pcd],
                                  width=800, height=600,
                                  window_name='Mesh with Normal')

pcd = mesh.sample_points_uniformly(number_of_points=2500)
pcd = mesh.sample_points_poisson_disk(number_of_points=500, pcl=pcd)
o3d.visualization.draw_geometries([pcd],
                                  width=800, height=600,
                                  window_name='Meshed Point Cloud')
########################################################################################################################
# 11. Mesh subdivision
########################################################################################################################
'''
In mesh subdivision, we divide each triangle into a number of smaller triangles. In the simplest case, we compute the
midpoint of each side per triangle and divide the triangle into four smaller triangles. This is implemented in the
subdivide_midpoint function. The 3D surface and area stays the same, but the number of vertices and triangles increases.
The parameter number_of_iterations defines how many times this process should be repeated.
'''
mesh = o3d.geometry.TriangleMesh.create_box()
mesh.compute_vertex_normals()
print(f'The mesh has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles')

o3d.visualization.draw_geometries([mesh], zoom=0.8, mesh_show_wireframe=True,
                                  width=800, height=600,
                                  window_name='Raw Mesh')

mesh = mesh.subdivide_midpoint(number_of_iterations=1)
print(f'After subdivision it has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles')
o3d.visualization.draw_geometries([mesh], zoom=0.8, mesh_show_wireframe=True,
                                  width=800, height=600,
                                  window_name='Mesh Subdivision')
'''
Open3D implements an additional subdivision method based on [Loop1987]. The method is based on a quartic box spline,
which generates C2 continuous limit surfaces everywhere except at extraordinary vertices where they are C1 continuous.
This leads to smoother corners.
'''
mesh = o3d.geometry.TriangleMesh.create_sphere()
mesh.compute_vertex_normals()
print(f'The mesh has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles')
o3d.visualization.draw_geometries([mesh], zoom=0.8, mesh_show_wireframe=True,
                                  width=800, height=600,
                                  window_name='Raw Mesh')

mesh = mesh.subdivide_loop(number_of_iterations=2)
print(f'After subdivision it has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles')
o3d.visualization.draw_geometries([mesh], zoom=0.8, mesh_show_wireframe=True,
                                  width=800, height=600,
                                  window_name='Mesh Subdivision')


mesh = o3dtut.get_knot_mesh()
mesh.compute_vertex_normals()
print(f'The mesh has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles')
o3d.visualization.draw_geometries([mesh], zoom=0.8, mesh_show_wireframe=True,
                                  width=800, height=600,
                                  window_name='Raw Mesh')

mesh = mesh.subdivide_loop(number_of_iterations=1)
print(f'After subdivision it has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles')
o3d.visualization.draw_geometries([mesh], zoom=0.8, mesh_show_wireframe=True,
                                  width=800, height=600,
                                  window_name='Mesh Subdivision')

########################################################################################################################
# 12. Mesh simplification
########################################################################################################################
'''
Sometimes we want to represent a high-resolution mesh with fewer triangles and vertices, but the low-resolution mesh
should still be close to the high-resolution mesh. For this purpose Open3D implements a number of mesh simplification
methods.
'''
########################################################################################################################
# 13. Mesh simplification - Vertex clustering
########################################################################################################################
'''
The vertex clustering method pools all vertices that fall into a voxel of a given size to a single vertex. The method is
implemented in simplify_vertex_clustering and has as parameters voxel_size that defines the size of the voxel grid and
contraction that defines how the vertices are pooled. o3d.geometry.SimplificationContraction.Average computes a simple
average.
'''
mesh_in = o3dtut.get_bunny_mesh()
print(f'Input mesh has {len(mesh_in.vertices)} vertices and {len(mesh_in.triangles)} triangles')
o3d.visualization.draw_geometries([mesh_in],
                                  width=800, height=600,
                                  window_name='Vertex clustering')

voxel_size = max(mesh_in.get_max_bound() - mesh_in.get_min_bound()) / 32
print(f'voxel_size = {voxel_size:e}')
mesh_smp = mesh_in.simplify_vertex_clustering(voxel_size=voxel_size,
                                              contraction=o3d.geometry.SimplificationContraction.Average)
print(f'Simplified mesh has {len(mesh_smp.vertices)} vertices and {len(mesh_smp.triangles)} triangles')
o3d.visualization.draw_geometries([mesh_smp],
                                  width=800, height=600,
                                  window_name='Vertex clustering: ' + str(voxel_size))

voxel_size = max(mesh_in.get_max_bound() - mesh_in.get_min_bound()) / 16
print(f'voxel_size = {voxel_size:e}')
mesh_smp = mesh_in.simplify_vertex_clustering(voxel_size=voxel_size,
                                              contraction=o3d.geometry.SimplificationContraction.Average)
print(f'Simplified mesh has {len(mesh_smp.vertices)} vertices and {len(mesh_smp.triangles)} triangles')
o3d.visualization.draw_geometries([mesh_smp],
                                  width=800, height=600,
                                  window_name='Vertex clustering: ' + str(voxel_size))
########################################################################################################################
# 14. Mesh decimation
########################################################################################################################
'''
Another category of mesh simplification methods is mesh decimation that operates in incremental steps. We select a
single triangle that minimizes an error metric and removes it. This is repeated until a required number of triangles is
achieved. Open3D implements simplify_quadric_decimation that minimizes error quadrics (distances to neighboring planes).
The parameter target_number_of_triangles defines the stopping critera of the decimation algorithm.
'''
mesh_smp = mesh_in.simplify_quadric_decimation(target_number_of_triangles=6500)
print(f'Simplified mesh has {len(mesh_smp.vertices)} vertices and {len(mesh_smp.triangles)} triangles')
o3d.visualization.draw_geometries([mesh_smp],
                                  width=800, height=600,
                                  window_name='Mesh decimation: ' + str(6500))

mesh_smp = mesh_in.simplify_quadric_decimation(target_number_of_triangles=1700)
print(f'Simplified mesh has {len(mesh_smp.vertices)} vertices and {len(mesh_smp.triangles)} triangles')
o3d.visualization.draw_geometries([mesh_smp],
                                  width=800, height=600,
                                  window_name='Mesh decimation: ' + str(1700))

########################################################################################################################
# 15. Connected components
########################################################################################################################
'''
The result of various reconstruction methods.
Open3D implements a connected components algorithm cluster_connected_triangles that assigns each triangle to a cluster
of connected triangles. It returns for each triangle the index of the cluster in triangle_clusters, and per cluster the
number of triangles in cluster_n_triangles and the surface area of the cluster in cluster_area.

This is useful in for instance RGBD Integration, which is not always a single triangle mesh, but a number of meshes.
Some of the smaller parts are due to noise and we most likely want to remove them.

The code below shows the application of cluster_connected_triangles and how it can be used to remove spurious triangles.
'''
print("Generate data")
mesh = o3dtut.get_bunny_mesh().subdivide_midpoint(number_of_iterations=2)
vert = np.asarray(mesh.vertices)
min_vert, max_vert = vert.min(axis=0), vert.max(axis=0)
for _ in range(30):
    cube = o3d.geometry.TriangleMesh.create_box()
    cube.scale(0.005, center=cube.get_center())
    cube.translate(
        (
            np.random.uniform(min_vert[0], max_vert[0]),
            np.random.uniform(min_vert[1], max_vert[1]),
            np.random.uniform(min_vert[2], max_vert[2]),
        ),
        relative=False,
    )
    mesh += cube
mesh.compute_vertex_normals()
print("Show input mesh")
o3d.visualization.draw_geometries([mesh],
                                  width=800, height=600,
                                  window_name='Input Mesh')

print("Cluster connected triangles")
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    triangle_clusters, cluster_n_triangles, cluster_area = (mesh.cluster_connected_triangles())
triangle_clusters = np.asarray(triangle_clusters)
cluster_n_triangles = np.asarray(cluster_n_triangles)
cluster_area = np.asarray(cluster_area)

print("Show mesh with small clusters removed")
mesh_0 = copy.deepcopy(mesh)
triangles_to_remove = cluster_n_triangles[triangle_clusters] < 100
mesh_0.remove_triangles_by_mask(triangles_to_remove)
o3d.visualization.draw_geometries([mesh_0],
                                  width=800, height=600,
                                  window_name='small clusters')

print("Show largest cluster")
mesh_1 = copy.deepcopy(mesh)
largest_cluster_idx = cluster_n_triangles.argmax()
triangles_to_remove = triangle_clusters != largest_cluster_idx
mesh_1.remove_triangles_by_mask(triangles_to_remove)
o3d.visualization.draw_geometries([mesh_1],
                                  width=800, height=600,
                                  window_name='largest cluster')
