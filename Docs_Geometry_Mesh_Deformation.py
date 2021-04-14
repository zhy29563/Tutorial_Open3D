import open3d as o3d
import numpy as np
import python.open3d_tutorial as o3dtut

# http://www.open3d.org/docs/release/tutorial/geometry/mesh_deformation.html

########################################################################################################################
# 1. Mesh deformation
########################################################################################################################
'''
If we want to deform a triangle mesh according to a small number of constraints, we can use mesh deformation algorithms.
Open3D implements the as-rigid-as-possible method by [SorkineAndAlexa2007] that optimizes the following energy function
    ∑i∑j∈N(i)wij||(p′i−p′j)−Ri(pi−pj)||2,
where Ri are the rotation matrices that we want to optimize for, and pi and p′i are the vertex positions before and
after the optimization, respectively. N(i) is the set of neighbors of vertex i. The weights wij are cotangent weights.

Open3D implements this method in deform_as_rigid_as_possible. The first argument to this method is a set of
constraint_ids that refer to the vertices in the triangle mesh. The second argument constrint_pos defines at which
position those vertices should be after the optimization. The optimization process is an iterative scheme. Hence, we
also can define the number of iterations via max_iter.
'''
mesh = o3dtut.get_armadillo_mesh()

vertices = np.asarray(mesh.vertices)
static_ids = [idx for idx in np.where(vertices[:, 1] < -30)[0]]
static_pos = []
for id in static_ids:
    static_pos.append(vertices[id])
handle_ids = [2490]
handle_pos = [vertices[2490] + np.array((-40, -40, -40))]
constraint_ids = o3d.utility.IntVector(static_ids + handle_ids)
constraint_pos = o3d.utility.Vector3dVector(static_pos + handle_pos)

with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
    mesh_prime = mesh.deform_as_rigid_as_possible(constraint_ids, constraint_pos, max_iter=50)

print('Original Mesh')
R = mesh.get_rotation_matrix_from_xyz((0, np.pi, 0))
o3d.visualization.draw_geometries([mesh.rotate(R, center=mesh.get_center())])
print('Deformed Mesh')
mesh_prime.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh_prime.rotate(R, center=mesh_prime.get_center())], width=800, height=600)

########################################################################################################################
# 2. Smoothed ARAP
########################################################################################################################

'''
Open3D also implements a smoothed version of the ARAP objective defined as
∑i∑j∈N(i)wij||(p′i−p′j)−Ri(pi−pj)||2+αA||Ri−Rj||2,
that penalizes a deviation of neighboring rotation matrices. α is a trade-off parameter for the regularization term and
A is the surface area.

The smoothed objective can be used in deform_as_rigid_as_possible by using the argument energy with the parameter
Smoothed.
'''
