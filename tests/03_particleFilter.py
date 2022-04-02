import numpy as np
from sklearn.preprocessing import scale
from particleFilter.maps import bearingBeacons_2D
from particleFilter.geometry import pose2
from particleFilter.particleFilterLocalization_2D import ParticleFilter
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
pf = ParticleFilter(modelMap,initialParticles)

#----- prep visuals
def plotParticles(ax : plt.Axes ,particles : list[pose2]):
    locals = np.array([p.local() for p in particles])
    u = np.cos(locals[:,2])
    v = np.sin(locals[:,2])
    return ax.quiver(locals[:,0],locals[:,1],u,v)

def plotGT(ax : plt.Axes, x : pose2):
    return ax.quiver(x.x,x.y,np.cos(x.theta),np.sin(x.theta), color = 'r')

ax = modelMap.show(xrange = (-3,3), yrange = (-3,3))
graphics_particles = plotParticles(ax, pf.particles)
graphocs_gt = plotGT(ax,gt_x0)

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
        mu, cov = pf.bestEstimate()

        #add visuals
        graphics_particles.remove()
        graphocs_gt.remove()
        graphics_particles = plotParticles(ax, pf.particles)
        graphocs_gt = plotGT(ax,gt_x0)
        plt.pause(0.1)




