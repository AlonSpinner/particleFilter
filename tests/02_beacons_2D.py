import numpy as np
from particleFilter.maps import beacons_2D
from particleFilter.geometry import pose2


beacons = np.array([[1,0],
                    [0,1],
                    [0,11]]).reshape(-1,2,1)
modelMap = beacons_2D(beacons)

x = pose2(0,0,0)

print(modelMap.forward_measurement_model(x))




