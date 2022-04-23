import numpy as np
from particleFilter.maps import o3d_meshes
from particleFilter.geometry import pose2

from particleFilter.RBPF.gridmaps import gridmap2
from particleFilter.RBPF.sensors import laser2

import particleFilter.plotting as plotting
import matplotlib.pyplot as plt
import open3d as o3d

#enviorment same as test 05
def createStructure():

    wallWidth = 0.2
    wallHeight = 2.0
    drop = -1.0
    wallColor = [0.5, 0.5, 0.3]
    firstwallColor = [0.2, 0.2, 0.3]

    products = []

    ###---- Room 1
    wall = o3d.geometry.TriangleMesh.create_box(width=wallWidth,height=3.0, depth=wallHeight)
    wall.paint_uniform_color(firstwallColor)
    T = pose2(0,0,0).T3d()
    T[2,3] = drop
    wall.transform(T)
    products.append(wall)

    wall = o3d.geometry.TriangleMesh.create_box(width=wallWidth,height=3.0, depth=wallHeight)
    wall.paint_uniform_color(wallColor)
    T = pose2(0,0,np.pi/2).T3d()
    T[2,3] = drop
    wall.transform(T)
    products.append(wall)

    wall = o3d.geometry.TriangleMesh.create_box(width=wallWidth,height=3.0, depth=wallHeight)
    wall.paint_uniform_color(wallColor)
    T = pose2(0,3-wallWidth,np.pi/2).T3d()
    T[2,3] = drop
    wall.transform(T)
    products.append(wall)

    wall = o3d.geometry.TriangleMesh.create_box(width=wallWidth,height=1.5, depth=wallHeight)
    wall.paint_uniform_color(wallColor)
    T = pose2(-3,wallWidth,0).T3d()
    T[2,3] = drop
    wall.transform(T)
    products.append(wall)

    ###--- Room 2
    wall = o3d.geometry.TriangleMesh.create_box(width=wallWidth,height=3.0, depth=wallHeight)
    wall.paint_uniform_color(wallColor)
    T = pose2(-3,3,0).T3d()
    T[2,3] = drop
    wall.transform(T)
    products.append(wall)

    wall = o3d.geometry.TriangleMesh.create_box(width=wallWidth,height=4.0, depth=wallHeight)
    wall.paint_uniform_color(wallColor)
    T = pose2(-3,0,np.pi/2).T3d()
    T[2,3] = drop
    wall.transform(T)
    products.append(wall)

    wall = o3d.geometry.TriangleMesh.create_box(width=wallWidth,height=4.0, depth=wallHeight)
    wall.paint_uniform_color(wallColor)
    T = pose2(-7-wallWidth,0,0).T3d()
    T[2,3] = drop
    wall.transform(T)
    products.append(wall)

    wall = o3d.geometry.TriangleMesh.create_box(width=wallWidth,height=3.0, depth=wallHeight)
    wall.paint_uniform_color(wallColor)
    T = pose2(-7,4,np.pi/2).T3d()
    T[2,3] = drop
    wall.transform(T)
    products.append(wall)

    wall = o3d.geometry.TriangleMesh.create_box(width=wallWidth,height=4.0, depth=wallHeight)
    wall.paint_uniform_color(wallColor)
    T = pose2(-10,4+wallWidth,0).T3d()
    T[2,3] = drop
    wall.transform(T)
    products.append(wall)

    wall = o3d.geometry.TriangleMesh.create_box(width=wallWidth,height=7.0, depth=wallHeight)
    wall.paint_uniform_color(wallColor)
    T = pose2(-3,8+wallWidth,np.pi/2).T3d()
    T[2,3] = drop
    wall.transform(T)
    products.append(wall)

    ###--- Room 3

    wall = o3d.geometry.TriangleMesh.create_box(width=wallWidth,height=3.0, depth=wallHeight)
    wall.paint_uniform_color(wallColor)
    T = pose2(0,8+wallWidth,np.pi/2).T3d()
    T[2,3] = drop
    wall.transform(T)
    products.append(wall)

    wall = o3d.geometry.TriangleMesh.create_box(width=wallWidth,height=5.0+2*wallWidth, depth=wallHeight)
    wall.paint_uniform_color(wallColor)
    T = pose2(0,3,0).T3d()
    T[2,3] = drop
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
class robot():
    def __init__(self,x,y,theta,angles):
        self.x = x
        self.y = y
        self.theta = theta
        self.angles = angles

    def __add__(self,u):
        p = self.pose() + u
        return robot(p.x,p.y,p.theta,self.angles)

    def pose(self):
        return pose2(self.x,self.y,self.theta)

    def local(self):
        return self.pose().local()

#-------- create world map
products = createStructure()
patches2d = [o3d_to_2patches(p) for p in products]
rayCastingScene = o3d.t.geometry.RaycastingScene()
for p in products:
    p = o3d.t.geometry.TriangleMesh.from_legacy(p)
    rayCastingScene.add_triangles(p)
worldMap = o3d_meshes(patches2d,rayCastingScene)

#-------- create empty gridmap to be filled
gmap = gridmap2(1,1,40,40)

#-------- initalize robot
angles = np.radians(np.linspace(-180,180,50))
gt_x = robot(-0.4,0.6,np.pi/2,angles)
x0 = gt_x.pose()
Z_COV = np.array([0.01])

laser = laser2(angles)

#----------build odometry (circle around)
straight = [pose2(0.1,0,0)]
turnLeft = [pose2(0,0,np.pi/2/4)]
turnRight = [pose2(0,0,-np.pi/2/4)]
gt_odom = straight*15 + turnLeft*4 + straight*10*5 + turnRight*4 + straight*10*5 + turnRight*4 + straight*40 + turnRight*4 + straight*20

#----- prep visuals
_, ax = plotting.spawnWorld(xrange = (-12,1), yrange = (-1,9))
graphics_meas = ax.scatter([],[],s = 10, color = 'r')
worldMap.show(ax)
graphics_gt = plotting.plot_pose2(ax,[gt_x],color = 'r')
plt.draw(); plt.pause(0.5)

#------ run simulation
with plt.ion():
    for i,u in enumerate(gt_odom):
        gt_x += u
        
        #compute noisey map measurement
        z_real = worldMap.forward_measurement_model(gt_x)
        z_cov = np.kron(np.eye(int(z_real.size)),Z_COV) # amount of measurements might differ depending on gt_x0
        z_noise = np.random.multivariate_normal(z_real.squeeze(), z_cov).reshape(-1,1)

        for a,z in zip(laser.angles,z_real):
            dx = z * np.cos(a); dy = z * np.sin(a)
            lm = gt_x.pose().transformFrom(np.array([dx,dy]))
            gmap.updateHit(gmap.c2d(lm))
 
        #add visuals
        graphics_gt.remove()
        graphics_gt = plotting.plot_pose2(ax,[gt_x],color = 'r')
        dx_meas = gt_x.x + z_real*np.cos(gt_x.theta+angles).reshape(-1,1)
        dy_meas = gt_x.y + z_real*np.sin(gt_x.theta+angles).reshape(-1,1)
        
        graphics_meas.set_offsets(np.hstack((dx_meas,dy_meas)))

        plt.pause(0.01)

gmap.show()
plt.show()




