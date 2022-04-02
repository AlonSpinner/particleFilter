import numpy as np
from particleFilter.maps import bearingBeacons_2D
from particleFilter.geometry import pose2
from particleFilter.filters import pf_vanila_SE2
import particleFilter.plotting as plotting
import matplotlib.pyplot as plt

#--------build map
beacons = np.array([[1,0],
                    [0,1],
                    [2,2]]).reshape(-1,2,1)
modelMap = bearingBeacons_2D(beacons)
modelMap.beaconsRange = 100 #basicaly inf

#-------- initalize robot
gt_x0 = pose2(1,0,0)
Z_STD = np.deg2rad(1)
U_COV = np.zeros((3,3))
U_COV[0,0] = 0.01; U_COV[1,1] = 0.01; U_COV[2,2] = 0.01

#----------build odometry (circle around)
dx = 0.309017
dy = 0.0489435
dtheta = 0.314159
gt_odom = [pose2(dx,dy,dtheta)] * 20

#-------- initalize particle filter
initialParticles = []
for i in range(100):
    x = np.random.uniform(-3,3)
    y = np.random.uniform(-3,3)
    theta = np.random.uniform(-np.pi,np.pi)
    initialParticles.append(pose2(x,y,theta))
pf = pf_vanila_SE2(modelMap,initialParticles)

#----- prep visuals
ax = modelMap.show(xrange = (-3,3), yrange = (-3,3))
graphics_particles = plotting.plot_pose2(ax,pf.particles)
graphocs_gt = plotting.plot_pose2(ax,[gt_x0],color = 'r')
plt.draw(); plt.pause(0.1)

#------ run simulation
with plt.ion():
    for u in gt_odom:
        gt_x0 += u

        #compute noisey odometry
        w = np.random.multivariate_normal(np.zeros(u.size), U_COV)
        u_noise = u + pose2(w[0],w[1],w[2])
        
        #compute noisey map measurement
        z = modelMap.forward_measurement_model(gt_x0)
        z_cov = np.kron(np.eye(z.size),Z_STD**2) # amount of measurements might differ depending on gt_x0
        z_noise = np.random.multivariate_normal(z.squeeze(), z_cov).reshape(-1,1)
        
        pf.step(z_noise,z_cov,u,U_COV)
        pf.low_variance_sampler()
        mu, cov = pf.bestEstimate()

        #add visuals
        graphics_particles.remove()
        graphocs_gt.remove()
        graphics_particles = plotting.plot_pose2(ax,pf.particles)
        graphocs_gt = plotting.plot_pose2(ax,[gt_x0],color = 'r')
        plt.pause(0.1)




