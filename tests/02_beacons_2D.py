import numpy as np
from particleFilter.maps import beacons2D_bearing
from particleFilter.geometry import pose2
import matplotlib.pyplot as plt

beacons = np.array([[1,0],
                    [0,1],
                    [0,11]]).reshape(-1,2,1)
modelMap = beacons2D_bearing(beacons)
modelMap.beaconsRange = 10.0
modelMap.show(xrange = (-10,10), yrange = (-5,15))
plt.show()

x = pose2(0,0,0)

print(modelMap.forward_measurement_model(x))








