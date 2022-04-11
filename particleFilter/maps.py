import numpy as np
from particleFilter.geometry import pose2
import matplotlib.pyplot as plt
import open3d as o3d
from matplotlib.patches import Polygon

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

class beacons2D_bearingRange(Map):
    def __init__(self,beacons : np.ndarray):
        self.beacons : np.ndarray = beacons #shape(m,2,1)
        self.beaconsRange : float = 10.0

    def forward_measurement_model(self, x : pose2):
        z = []
        for b in self.beacons:
            if x.range(b) < self.beaconsRange:
                z.append((x.bearing(b),x.range(b)))
        return np.array(z).reshape(-1,1)

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

class o3d_meshes(Map):
        #http://www.open3d.org/docs/release/tutorial/geometry/ray_casting.html
        def __init__(self,meshes,rayCastingScene):
            self.scene =  rayCastingScene #o3d.cuda.pybind.t.geometry.RaycastingScene
            self.meshes = meshes #{vertices,faces,color}

        def forward_measurement_model(self, x : pose2):
            angles = np.radians(np.linspace(-10,10,20))
            zmax = 100.0

            rays = o3d.core.Tensor([[x.x,x.y,2,0,0,x.theta+a] for a in angles],
                       dtype=o3d.core.Dtype.Float32)
            ans = self.scene.cast_rays(rays)
            z = ans['t_hit'].numpy()
            z[z == np.inf] = zmax
            return z.reshape(-1,1)

        def show(self,ax : plt.Axes = None, xrange = None, yrange = None):
            if ax == None:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.set_xlabel('x'); ax.set_ylabel('y'); 
                ax.set_aspect('equal'); ax.grid()

            for mesh in self.meshes:
                triangles = mesh["triangles"]
                for tri in triangles:
                    polygon = Polygon(tri, True)
                    ax.add_patch(polygon)

            if xrange is not None: ax.set_xlim(xrange)
            if yrange is not None: ax.set_ylim(yrange)

            return ax




