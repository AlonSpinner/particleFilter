import numpy as np
from particleFilter.maps import o3d_meshes
from particleFilter.geometry import pose2

from particleFilter.RBPF.gridmaps import gridmap2
from particleFilter.RBPF.sensors import laser
from particleFilter.RBPF.models import inverse_measurement_model
from particleFilter.RBPF.gridRBPF import RBPF

import particleFilter.plotting as plotting
import matplotlib.pyplot as plt
import open3d as o3d

import os
import pickle

#enviorment same as test 05 in particleFilter just without floor and shifted
def createStructure():

    wallWidth = 0.2
    wallHeight = 2.0
    drop = -1.0
    wallColor = [0.5, 0.5, 0.3]
    firstwallColor = [0.2, 0.2, 0.3]
    xshift = 20
    products = []

    ###---- Room 1
    wall = o3d.geometry.TriangleMesh.create_box(width=wallWidth,height=3.0, depth=wallHeight)
    wall.paint_uniform_color(firstwallColor)
    T = pose2(0,0,0).T3d()
    T[2,3] = drop
    T[0,3] = T[0,3]+xshift
    wall.transform(T)
    products.append(wall)

    wall = o3d.geometry.TriangleMesh.create_box(width=wallWidth,height=3.0, depth=wallHeight)
    wall.paint_uniform_color(wallColor)
    T = pose2(0,0,np.pi/2).T3d()
    T[2,3] = drop
    T[0,3] = T[0,3]+xshift
    wall.transform(T)
    products.append(wall)

    wall = o3d.geometry.TriangleMesh.create_box(width=wallWidth,height=3.0, depth=wallHeight)
    wall.paint_uniform_color(wallColor)
    T = pose2(0,3-wallWidth,np.pi/2).T3d()
    T[2,3] = drop
    T[0,3] = T[0,3]+xshift
    wall.transform(T)
    products.append(wall)

    wall = o3d.geometry.TriangleMesh.create_box(width=wallWidth,height=1.5, depth=wallHeight)
    wall.paint_uniform_color(wallColor)
    T = pose2(-3,wallWidth,0).T3d()
    T[2,3] = drop
    T[0,3] = T[0,3]+xshift
    wall.transform(T)
    products.append(wall)

    ###--- Room 2
    wall = o3d.geometry.TriangleMesh.create_box(width=wallWidth,height=3.0, depth=wallHeight)
    wall.paint_uniform_color(wallColor)
    T = pose2(-3,3,0).T3d()
    T[2,3] = drop
    T[0,3] = T[0,3]+xshift
    wall.transform(T)
    products.append(wall)

    wall = o3d.geometry.TriangleMesh.create_box(width=wallWidth,height=4.0, depth=wallHeight)
    wall.paint_uniform_color(wallColor)
    T = pose2(-3,0,np.pi/2).T3d()
    T[2,3] = drop
    T[0,3] = T[0,3]+xshift
    wall.transform(T)
    products.append(wall)

    wall = o3d.geometry.TriangleMesh.create_box(width=wallWidth,height=4.0, depth=wallHeight)
    wall.paint_uniform_color(wallColor)
    T = pose2(-7-wallWidth,0,0).T3d()
    T[2,3] = drop
    T[0,3] = T[0,3]+xshift
    wall.transform(T)
    products.append(wall)

    wall = o3d.geometry.TriangleMesh.create_box(width=wallWidth,height=3.0, depth=wallHeight)
    wall.paint_uniform_color(wallColor)
    T = pose2(-7,4,np.pi/2).T3d()
    T[2,3] = drop
    T[0,3] = T[0,3]+xshift
    wall.transform(T)
    products.append(wall)

    wall = o3d.geometry.TriangleMesh.create_box(width=wallWidth,height=4.0, depth=wallHeight)
    wall.paint_uniform_color(wallColor)
    T = pose2(-10,4+wallWidth,0).T3d()
    T[2,3] = drop
    T[0,3] = T[0,3]+xshift
    wall.transform(T)
    products.append(wall)

    wall = o3d.geometry.TriangleMesh.create_box(width=wallWidth,height=7.0, depth=wallHeight)
    wall.paint_uniform_color(wallColor)
    T = pose2(-3,8+wallWidth,np.pi/2).T3d()
    T[2,3] = drop
    T[0,3] = T[0,3]+xshift
    wall.transform(T)
    products.append(wall)

    ###--- Room 3

    wall = o3d.geometry.TriangleMesh.create_box(width=wallWidth,height=3.0, depth=wallHeight)
    wall.paint_uniform_color(wallColor)
    T = pose2(0,8+wallWidth,np.pi/2).T3d()
    T[2,3] = drop
    T[0,3] = T[0,3]+xshift
    wall.transform(T)
    products.append(wall)

    wall = o3d.geometry.TriangleMesh.create_box(width=wallWidth,height=5.0+2*wallWidth, depth=wallHeight)
    wall.paint_uniform_color(wallColor)
    T = pose2(0,3,0).T3d()
    T[2,3] = drop
    T[0,3] = T[0,3]+xshift
    wall.transform(T)
    products.append(wall)

    return products
#for plotting in pyplot, we convert to 2D patches
def o3d_to_2patches(o3d_mesh):
    v = np.asarray(o3d_mesh.vertices)
    f = np.asarray(o3d_mesh.triangles)
    
    v = v[:,:2]
    triangles = v[f]

    good_triangles = []
    for tri in triangles:
        if tri.size == np.unique(tri, axis = 0).size:
            good_triangles.append(tri)

    color = np.asarray(np.average(o3d_mesh.vertex_colors,0))
    patches2d = {"triangles": good_triangles, "color": color}
    
    return patches2d
#wrapper class to include angles

#-------- create world map
products = createStructure()
patches2d = [o3d_to_2patches(p) for p in products]
rayCastingScene = o3d.t.geometry.RaycastingScene()
for p in products:
    p = o3d.t.geometry.TriangleMesh.from_legacy(p)
    rayCastingScene.add_triangles(p)
worldMap = o3d_meshes(patches2d,rayCastingScene)

#-------- create empty gridmap to be filled
gMap = gridmap2(12,10,0.1)
gMap.pose = pose2(9,-1,0)
#------- laser sensor on robot
sensor = laser(angles = np.radians(np.linspace(-180,180,50)), zmax = 2.0)
#-------- initalize robot
x = pose2(19.4,0.6,np.pi/2) #ground truth
Z_COV = np.array([0.01])
U_COV = np.zeros((3,3))
U_COV[0,0] = 0.01; U_COV[1,1] = 0.01; U_COV[2,2] = 0.01
U_COV = U_COV/10

#----------build odometry (circle around)
straight = [pose2(0.1,0,0)]
turnLeft = [pose2(0,0,np.pi/2/4)]
turnRight = [pose2(0,0,-np.pi/2/4)]
odom = straight*15 + turnLeft*4 + straight*10*5 + turnRight*4 + straight*10*5 + turnRight*4 + straight*40 + turnRight*4 + straight*20

#-------- initalize particle filter
n_particles = 5
initialParticles = []
for i in range(n_particles):
    noise = np.random.multivariate_normal(np.zeros(x.size),U_COV/2)
    initialParticles.append(x + pose2(noise[0],noise[1],noise[2]))
rbpf = RBPF(gMap,initialParticles,sensor)

#----- prep visuals
ax_grid = gMap.show()
_, ax_world = plotting.spawnWorld(xrange = (8,22), yrange = (0,10))
graphics_meas = ax_world.scatter([],[],s = 10, color = 'r')
graphics_particles = plotting.plot_pose2(ax_world, rbpf.get_poses(), scale = rbpf.weights)
worldMap.show(ax_world)
graphics_gt = plotting.plot_pose2(ax_world,[x],color = 'r')
plt.draw(); plt.pause(0.5)

#------ run simulation
with plt.ion():
    for i,u in enumerate(odom):
        x += u
        
        #compute noisey odometry
        w = np.random.multivariate_normal(np.zeros(u.size), U_COV)
        u_noise = u + pose2(w[0],w[1],w[2])

        #compute noisey map measurement
        z_perfect = worldMap.forward_measurement_model(x,sensor.angles)
        z_cov = np.kron(np.eye(int(z_perfect.size)),Z_COV) # amount of measurements might differ depending on x
        z_noise = np.random.multivariate_normal(z_perfect.squeeze(), z_cov).reshape(-1,1)

        rbpf.step(z_noise,z_cov,u_noise,U_COV)
 
        #add visuals
        graphics_gt.remove()
        graphics_particles.remove()
        graphics_particles = plotting.plot_pose2(ax_world, rbpf.get_poses(), scale = rbpf.weights)
        graphics_gt = plotting.plot_pose2(ax_world,[x],color = 'r')
        dx_meas = x.x + z_noise*np.cos(x.theta+sensor.angles).reshape(-1,1)
        dy_meas = x.y + z_noise*np.sin(x.theta+sensor.angles).reshape(-1,1)        
        graphics_meas.set_offsets(np.hstack((dx_meas,dy_meas)))

        plt.pause(0.01)
        if i%5==0:
            rbpf.particles[np.argmax(rbpf.weights)].map.show(ax_grid)

plt.show()



