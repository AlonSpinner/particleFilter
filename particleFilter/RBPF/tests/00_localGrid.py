from particleFilter.RBPF.gridmaps import gridmap2
import matplotlib.pyplot as plt
import numpy as np

g = gridmap2(100,100,0.1)
g.beta = np.radians(10)
x_grid, y_grid, bearing_grid, range_grid = g.localGrid()

rays = g.localgrid2raymap()

ray = rays[135]
raygrid = np.zeros_like(bearing_grid)
raygrid[ray.idx[:,0],ray.idx[:,1]] = range_grid[ray.idx[:,0],ray.idx[:,1]]

# ij = g.c2d(ray.t).T
# ray_map = np.zeros_like(bearing_grid)
# ray_map[ij[:,0],ij[:,1]] = range_grid[ij[:,0],ij[:,1]]

fig, axs = plt.subplots(2,2)
axs[0,0].imshow(bearing_grid)
axs[0,0].set_title('bearing - [-pi,+pi]')
axs[0,1].imshow(range_grid)
axs[0,1].set_title('range')
axs[1,0].imshow(raygrid)
axs[1,0].set_title(f'grid cells in ray')
# axs[1,1].imshow(ray_map)
# axs[1,1].set_title(f'grid cells after transform in ray')
plt.show()
