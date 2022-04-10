import open3d as o3d
import numpy as np
# import matplotlib.pyplot as plt
wallWidth = 0.2
wallHeight = 2.0
wallColor = [0, 0.9, 0]

wall1 = o3d.geometry.TriangleMesh.create_box(width=wallWidth,height=wallHeight, depth=1.0)
wall1.paint_uniform_color(wallColor)
wall1.transform([0,0,0], relative=False)
# cube = o3d.t.geometry.TriangleMesh.from_legacy(cube)

# # scene = o3d.t.geometry.RaycastingScene()
# # cube_id = scene.add_triangles(cube)

o3d.visualization.draw_geometries([wall1])

# # vis = o3d.visualization.Visualizer()
# # vis.create_window()
# # vis.add_geometry(scene)
# # plt.pause(0.5)
# # vis.update_renderer()