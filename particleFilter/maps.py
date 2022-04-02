import numpy as np
from particleFilter.geometry import pose2
import matplotlib.pyplot as plt

class Map:
    def __init__(self):
        # method to be over written
        return

    def forward_measurement_model():
        # method to be over written. 
        # returns nxm where n is the amount of measurements
        return

class beacons2D_bearing(Map):
    def __init__(self,beacons : np.ndarray):
        self.beacons : np.ndarray = beacons #shape(m,2,1)
        self.beaconsRange : float = 10.0

    def forward_measurement_model(self, x : pose2):
        z = []
        for b in self.beacons:
            if x.range(b) < self.beaconsRange:
                z.append(x.bearing(b))
        return np.array(z)

    def show(self,ax : plt.Axes = None, xrange = None, yrange = None, size = 50, color = 'b'):
        if ax == None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_xlabel('x'); ax.set_ylabel('y'); 
            ax.set_aspect('equal'); ax.grid()
        
        for b in self.beacons:
            ax.scatter(b[0],b[1], s = size, c = color)

        if xrange is not None: ax.set_xlim(xrange)
        if yrange is not None: ax.set_ylim(yrange)

        return ax

class beacons2D_range(Map):
    def __init__(self,beacons : np.ndarray):
        self.beacons : np.ndarray = beacons #shape(m,2,1)
        self.beaconsRange : float = 10.0

    def forward_measurement_model(self, x : pose2):
        z = []
        for b in self.beacons:
            if x.range(b) < self.beaconsRange:
                z.append(x.range(b))
        return np.array(z)

    def show(self,ax : plt.Axes = None, xrange = None, yrange = None, size = 50, color = 'b'):
        if ax == None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_xlabel('x'); ax.set_ylabel('y'); 
            ax.set_aspect('equal'); ax.grid()
        
        for b in self.beacons:
            ax.scatter(b[0],b[1], s = size, c = color)

        if xrange is not None: ax.set_xlim(xrange)
        if yrange is not None: ax.set_ylim(yrange)

        return ax




