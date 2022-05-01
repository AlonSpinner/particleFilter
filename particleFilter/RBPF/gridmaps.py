import numpy as np
import matplotlib.pyplot as plt
from particleFilter.geometry import pose2
import matplotlib.ticker as tkr     # has classes for tick-locating and -formatting
from functools import partial

class gridmap2:
    def __init__(self,  maxX, maxY, resX, resY):
        self.width = int(np.ceil(maxX / resX))
        self.height = int(np.ceil(maxY / resY))

        self.resX = resX #higher scale <-> lower resolution
        self.resY = resY

        #construct with uniform map
        self.gridHits = np.zeros((self.height,self.width))
        self.gridMiss = np.zeros((self.height,self.width))

    def show(self,ax : plt.Axes = None):
        if ax == None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_xlabel('x'); ax.set_ylabel('y'); 
            ax.set_aspect('equal'); ax.grid()
        
        # cell holds probability of being occupied, which we want to paint black. hence data = 1-self.grid
        # unlike imshow, pcolor flips the data matrix on the vertical axis. Which is something we want built in.
        # also: pcolorfast is way faster than regular pcolor
        grid = self.gridHits/(self.gridHits + self.gridMiss)
        ax.pcolorfast(1-grid,  vmin=0.0, vmax=1.0, cmap=plt.cm.gray)
        
        yfmt = tkr.FuncFormatter(partial(numfmt,self.resY))
        ax.yaxis.set_major_formatter(yfmt)
        xfmt = tkr.FuncFormatter(partial(numfmt,self.resX))
        ax.xaxis.set_major_formatter(xfmt)

        return ax

    def update(self,cells,update):
        for c,u in zip(cells,update):
            if u == 'free':
               self.updateMiss(c)
            elif u == 'occ':
                self.updateHit(c)
            
            # logOdds(c,xt,zt) - logOdds(c) + logOdds
        
    def c2d(self,loc): #continous2discrete
        # We simplify the whole grid situations to this setting:
        # → x
        #↓
        #y
        
        #* there are no negative values in the map
        #* this is only a storage world. when we want to present, we flip the y axis 

        x,y = loc #unpack
        i = np.int(y / self.resY)
        j = np.int(x / self.resX)
        return (i,j)

    def d2c(self,loc): #discrete2continous
        i, j = loc #unpack
        x = (j + 0.5) * self.resX
        y = (i + 0.5) * self.resY
        return (x,y)

    def updateHit(self,c):
        #product of c2d - (i,j)
        if 0 <= c[0] < self.height and 0 <= c[1] < self.width:
            self.gridHits[c[0],c[1]] += 1

    def updateMiss(self,c):
        if 0 <= c[0] < self.height and 0 <= c[1] < self.width:
            #product of c2d - (i,j)
            self.gridMiss[c[0],c[1]] += 1

    #constructor from image
    def readFromImage(filename, scaleX = 1, scaleY = 1):
        # in grid, 1 ~ 100% occupied, as such, we need to change image values
        # we flip the image as the grid storage and world axes are with negative y axes
        im = rgb2gray(plt.imread(filename))
        processed = 1-np.flipud(im)
        return  gridmap2(processed, scaleX, scaleY)

def rgb2gray(rgb):
    #https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
        
def numfmt(v, x, pos): # your custom formatter function: divide by 100.0
    s = '{}'.format(x * v)
    return s
