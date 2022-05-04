import numpy as np
import open3d as o3d
import copy

#take two point cloud scans and put them on the same voxelgrid

N = 2000
armadillo_data = o3d.data.ArmadilloMesh()
pcd = o3d.io.read_triangle_mesh(
    armadillo_data.path).sample_points_poisson_disk(N)
# Fit to unit cube.
pcd.scale(1 / np.max(pcd.get_max_bound() - pcd.get_min_bound()),
            center=pcd.get_center())
pcd.colors = o3d.utility.Vector3dVector(np.random.uniform(0, 1,
                                                            size=(N, 3)))
# print('Displaying input point cloud ...')
# o3d.visualization.draw([pcd])

pcd1 = copy.deepcopy(pcd)

T = np.eye(4)
T[0,3] = 1.0
pcd1.transform(T)

print('Displaying voxel grid ...')
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,
                                                            voxel_size=0.05)                                         
o3d.visualization.draw([voxel_grid])

vg = o3d.geometry.VoxelGrid()
voxels = []
vox = o3d.geometry.Voxel()
vox.grid_index = np.array([  7,  59, 101])
voxels.append(vox)

