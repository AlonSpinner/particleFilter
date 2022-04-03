import numpy as np
from particleFilter.maps import beacons2D_bearing
from particleFilter.geometry import pose2
from particleFilter.filters import pf_vanila_SE2
import particleFilter.plotting as plotting
import matplotlib.pyplot as plt

#--------build map
beacons = np.array([[-2,-2],
                    [2,1],
                    [0,0]]).reshape(-1,2,1)
modelMap = beacons2D_bearing(beacons)
modelMap.beaconsRange = 100 #basicaly inf

#-------- initalize robot
gt_x = pose2(1,0,0)
Z_STD = np.deg2rad(1)
U_COV = np.zeros((3,3))
U_COV[0,0] = 0.01; U_COV[1,1] = 0.01; U_COV[2,2] = 0.01

#----------build odometry (circle around)
straight = [pose2(0.1,0,0)]
turn = [pose2(0,0,np.pi/2/4)]
gt_odom = straight*10 + turn*4 + straight*20 + turn*4 + straight*40

#-------- initalize particle filter
n_particles = 100
initialParticles = []
for i in range(n_particles):
    x = np.random.uniform(-3,3)
    y = np.random.uniform(-3,3)
    theta = np.random.uniform(-np.pi,np.pi)
    initialParticles.append(pose2(x,y,theta))
pf = pf_vanila_SE2(modelMap,initialParticles)
pf.ETA_THRESHOLD = 10.0/n_particles # bigger - lower threshold
pf.SPREAD_THRESHOLD = 5.0 #bigger - higher threshold

#----- prep visuals
_, ax = plotting.spawnWorld(xrange = (-3,3), yrange = (-3,3))
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
        z_cov = np.kron(np.eye(z.size),Z_STD**2) # amount of measurements might differ depending on gt_x0
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




