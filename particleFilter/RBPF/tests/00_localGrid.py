from particleFilter.RBPF.sensors import beamer
from particleFilter.geometry import pose2
import matplotlib.pyplot as plt
import numpy as np

sensor = beamer()

x_grid, y_grid, bearing_grid, range_grid = sensor.localGrid()

rays = sensor.localgrid2raymap()

ray = rays[0]
raygrid = np.zeros_like(bearing_grid)
raygrid[ray.idx[:,0],ray.idx[:,1]] = range_grid[ray.idx[:,0],ray.idx[:,1]]

sensor.pose = pose2(0,0,np.pi/4)
worldPoints = sensor.pose.transformFrom(ray.t)

fig, axs = plt.subplots(2,2)
axs[0,0].pcolorfast(bearing_grid)
axs[0,0].set_title('bearing - [-pi,+pi]')
axs[0,1].pcolorfast(range_grid)
axs[0,1].set_title('range')
axs[1,0].pcolorfast(raygrid)
axs[1,0].set_title(f'grid cells in ray')
axs[1,1].scatter(worldPoints[0,:],worldPoints[1,:])
axs[1,1].set_title(f'ray')
axs[1,1].set_aspect(1)
plt.show()
