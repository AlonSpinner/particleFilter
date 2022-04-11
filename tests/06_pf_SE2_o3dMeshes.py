import numpy as np
from particleFilter.maps import o3d_meshes
from particleFilter.geometry import pose2
from particleFilter.filters import pf_vanila_SE2
import particleFilter.plotting as plotting
import matplotlib.pyplot as plt
import open3d as o3d

#--------build map
def createStructure():

    wallWidth = 0.2
    wallHeight = 2.0
    wallColor = [0.5, 0.5, 0.3]
    firstwallColor = [0.2, 0.2, 0.3]

    FloorColor = [0.8, 0.8, 0.8]

    products = []

    ###---- Room 1
    wall = o3d.geometry.TriangleMesh.create_box(width=wallWidth,height=3.0, depth=wallHeight)
    wall.paint_uniform_color(firstwallColor)
    wall.transform(pose2(0,0,0).T3d())
    products.append(wall)

    wall = o3d.geometry.TriangleMesh.create_box(width=wallWidth,height=3.0, depth=wallHeight)
    wall.paint_uniform_color(wallColor)
    wall.transform(pose2(0,0,np.pi/2).T3d())
    products.append(wall)

    wall = o3d.geometry.TriangleMesh.create_box(width=wallWidth,height=3.0, depth=wallHeight)
    wall.paint_uniform_color(wallColor)
    wall.transform(pose2(0,3-wallWidth,np.pi/2).T3d())
    products.append(wall)

    wall = o3d.geometry.TriangleMesh.create_box(width=wallWidth,height=1.5, depth=wallHeight)
    wall.paint_uniform_color(wallColor)
    wall.transform(pose2(-3,wallWidth,0).T3d())
    products.append(wall)

    ###--- Room 2
    wall = o3d.geometry.TriangleMesh.create_box(width=wallWidth,height=3.0, depth=wallHeight)
    wall.paint_uniform_color(wallColor)
    wall.transform(pose2(-3,3,0).T3d())
    products.append(wall)

    wall = o3d.geometry.TriangleMesh.create_box(width=wallWidth,height=4.0, depth=wallHeight)
    wall.paint_uniform_color(wallColor)
    wall.transform(pose2(-3,0,np.pi/2).T3d())
    products.append(wall)

    wall = o3d.geometry.TriangleMesh.create_box(width=wallWidth,height=4.0, depth=wallHeight)
    wall.paint_uniform_color(wallColor)
    wall.transform(pose2(-7-wallWidth,0,0).T3d())
    products.append(wall)

    wall = o3d.geometry.TriangleMesh.create_box(width=wallWidth,height=3.0, depth=wallHeight)
    wall.paint_uniform_color(wallColor)
    wall.transform(pose2(-7,4,np.pi/2).T3d())
    products.append(wall)

    wall = o3d.geometry.TriangleMesh.create_box(width=wallWidth,height=4.0, depth=wallHeight)
    wall.paint_uniform_color(wallColor)
    wall.transform(pose2(-10,4+wallWidth,0).T3d())
    products.append(wall)

    wall = o3d.geometry.TriangleMesh.create_box(width=wallWidth,height=7.0, depth=wallHeight)
    wall.paint_uniform_color(wallColor)
    wall.transform(pose2(-3,8+wallWidth,np.pi/2).T3d())
    products.append(wall)

    ###--- Room 3

    wall = o3d.geometry.TriangleMesh.create_box(width=wallWidth,height=3.0, depth=wallHeight)
    wall.paint_uniform_color(wallColor)
    wall.transform(pose2(0,8+wallWidth,np.pi/2).T3d())
    products.append(wall)

    wall = o3d.geometry.TriangleMesh.create_box(width=wallWidth,height=5.0+2*wallWidth, depth=wallHeight)
    wall.paint_uniform_color(wallColor)
    wall.transform(pose2(0,3,0).T3d())
    products.append(wall)

    #### Floor
    # floor = o3d.geometry.TriangleMesh.create_box(width=20.0,height=20.0, depth=0.1)
    # floor.paint_uniform_color(FloorColor)
    # T = pose2(-14,-5,0).T3d()
    # T[2,3] = -0.1
    # floor.transform(T)
    # products.append(floor)

    return products
def o3d_to_mymesh(o3d_mesh):
    v = np.asarray(o3d_mesh.vertices)
    f = np.asarray(o3d_mesh.triangles)
    
    v = v[:,:2]
    triangles = v[f]

    good_triangles = []
    for tri in triangles:
        if tri.size == np.unique(tri, axis = 0).size:
            good_triangles.append(tri)

    color = np.asarray(np.average(o3d_mesh.vertex_colors,0))
    mymesh = {"triangles": good_triangles, "color": color}
    
    return mymesh

    
products = createStructure()
meshes = [o3d_to_mymesh(p) for p in products]
rayCastingScene = o3d.t.geometry.RaycastingScene()
for p in products:
    p = o3d.t.geometry.TriangleMesh.from_legacy(p)
    rayCastingScene.add_triangles(p)
modelMap = o3d_meshes(meshes,rayCastingScene)

#-------- initalize robot
gt_x = pose2(-0.4,0.6,np.pi/2)
Z_COV = np.array([0.1])
U_COV = np.zeros((3,3))
U_COV[0,0] = 0.01; U_COV[1,1] = 0.01; U_COV[2,2] = 0.01

#----------build odometry (circle around)
straight = [pose2(0.1,0,0)]
turnLeft = [pose2(0,0,np.pi/2/4)]
turnRight = [pose2(0,0,-np.pi/2/4)]
gt_odom = straight*15 + turnLeft*4 + straight*10*5 + turnRight*4 + straight*10*5 + turnRight*4 + straight*10

#-------- initalize particle filter
n_particles = 100
initialParticles = []
for i in range(n_particles):
    x = np.random.uniform(-10,2.5)
    y = np.random.uniform(-2.5,10)
    theta = np.random.uniform(-np.pi,np.pi)
    initialParticles.append(pose2(x,y,theta))
pf = pf_vanila_SE2(modelMap,initialParticles)
pf.ETA_THRESHOLD = 10.0/n_particles # bigger - lower threshold
pf.SPREAD_THRESHOLD = 1.0 #bigger - higher threshold

#----- prep visuals
_, ax = plotting.spawnWorld(xrange = (-12,1), yrange = (-1,9))
modelMap.show(ax)
graphics_particles = plotting.plot_pose2(ax,pf.particles, scale = pf.weights)
graphics_gt = plotting.plot_pose2(ax,[gt_x],color = 'r')
mu, cov = pf.estimateGaussian()
graphics_cov = plotting.plot_cov_ellipse(ax,mu[:2],cov[:2,:2], nstd = 1)
plt.draw(); plt.pause(0.5)

#------ run simulation
with plt.ion():
    for u in gt_odom:
        gt_x += u

        #compute noisey odometry
        w = np.random.multivariate_normal(np.zeros(u.size), U_COV)
        u_noise = u + pose2(w[0],w[1],w[2])
        
        #compute noisey map measurement
        z = modelMap.forward_measurement_model(gt_x)
        z_cov = np.kron(np.eye(int(z.size)),Z_COV) # amount of measurements might differ depending on gt_x0
        z_noise = np.random.multivariate_normal(z.squeeze(), z_cov).reshape(-1,1)
        
        pf.step(z_noise,z_cov,u,U_COV)
        #pf.low_variance_sampler()
        
        mu, cov = pf.estimateGaussian()
 
        #add visuals
        graphics_particles.remove()
        graphics_gt.remove()
        graphics_cov.remove()
        graphics_particles = plotting.plot_pose2(ax,pf.particles, scale = pf.weights)
        graphics_gt = plotting.plot_pose2(ax,[gt_x],color = 'r')
        graphics_cov = plotting.plot_cov_ellipse(ax, mu[:2],cov[:2,:2], nstd = 1)

        #plotting.plot_pose2_weight_distribution(pf.particles, pf.weights, gt_x)
        plt.pause(0.1)




