import numpy as np
import matplotlib.pyplot as plt
from particleFilter.geometry import pose2

class gridmap2:
    def __init__(self, scaleX, scaleY, width = None, height = None, p0 = None):
        
        #----------prior parameters
        if p0 is None:
            #construct with uniform map
            self.gridHits = np.zeros((width,height))
            self.gridMiss = np.zeros((width,height))
            self.width = width
            self.height = height
        else:
            #construct with pre-existing map
            self.grid = p0
            s = p0.shape
            self.width = s[0] 
            self.height = s[1] 

        self.scaleX = scaleX #(x,y)*scale -> i
        self.scaleY = scaleY

        #-------inverse_measurement_model parameters
    def show(self,ax : plt.Axes = None):
        if ax == None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_xlabel('x'); ax.set_ylabel('y'); 
            ax.set_aspect('equal'); ax.grid()
        
        # cell holds probability of being occupied, which we want to paint black. hence data = 1-self.grid
        # unlike imshow, pcolor flips the data matrix on the vertical axis. pcolorfast is way faster than regular pcolor
        # reminder: the convetion is that grid[0,0] <-> world_axis[0,0]
        grid = self.gridHits/(self.gridHits + self.gridMiss)
        ax.pcolorfast(1-np.flipud(grid),  vmin=0.0, vmax=1.0, cmap=plt.cm.gray)
        return ax

    # def update(self,cells,p):
    #     for c in freeCells:
    #         logOdds(c,xt,zt) - logOdds(c) + logOdds
        
    def c2d(self,loc): #continous2discrete
        x,y = loc #unpack
        i = np.round((self.height - y) * self.scaleY) - 20
        j = np.round(x * self.scaleX) + 20
        return (int(i),int(j))

    def d2c(self,loc): #discrete2continous
        i, j = loc #unpack
        x = (i + 0.5) / self.scaleX
        y = (self.height - j + 0.5) / self.scaleY
        return (x,y)

    def updateHit(self,c):
        self.gridHits[c[0],c[1]] += 1

    def updateMiss(self,c):
        self.gridMiss[c[0],c[1]] += 1

    #constructor from image
    def readFromImage(filename, scaleX = 1, scaleY = 1):
        # to make things as simple as possible, image axes ARE THE SAME as world axes.
        # in grid, 1 ~ 100% occupied, as such, we need to change image values
        im = rgb2gray(plt.imread(filename))
        processed = 1-im
        return  gridmap2(processed, scaleX, scaleY)

def rgb2gray(rgb):
    #https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
        
