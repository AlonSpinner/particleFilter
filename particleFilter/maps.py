import numpy as np
from particleFilter.geometry import pose2


class beacons_2D():
    def __init__(self,beacons : np.ndarray):
        self.beacons : np.ndarray = beacons #shape(m,2,1)
        self.beaconsRange : float = 10.0

    def forward_measurement_model(self, x : pose2):
        z = []
        for b in self.beacons:
            if x.range(b) < self.beaconsRange:
                z.append(x.bearing(b))
        return np.array(z)



