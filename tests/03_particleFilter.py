import numpy as np
from particleFilter.maps import beacons_2D
from particleFilter.geometry import pose2
from particleFilter.particleFilterLocalization_2D import ParticleFilter

#--------build map
beacons = np.array([[1,0],
                    [0,1],
                    [0,11]]).reshape(-1,2,1)
modelMap = beacons_2D(beacons)

#-------- initalize robot
gt_x0 = pose2(1,0,0)

#----------build odometry
dx = 0.309017 #taken from test 00: relative odometrey
dy = 0.0489435
dtheta = 0.314159
gt_odom = [pose2(dx,dy,dtheta)] * 20

#-------- initalize particle filter
initialParticles = []
for i in range(100):
    x = np.random.uniform(0,1)
    y = np.random.uniform(0,1)
    theta = np.random.uniform(-np.pi,np.pi)
    initialParticles.append(pose2(x,y,theta))
ParticleFilter(modelMap,initialParticles)

#------ run simulation
for o in gt_odom:
    gt_x0 += gt_odom[o]
    z = modelMap.forward_measurement_model(gt_x0)


print(modelMap.forward_measurement_model(x))




