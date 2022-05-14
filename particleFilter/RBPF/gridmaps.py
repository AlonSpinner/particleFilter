import numpy as np
import matplotlib.pyplot as plt
from particleFilter.geometry import pose2
from particleFilter.RBPF.utils import p2logodds, logodds2p, numfmt
import matplotlib.ticker as tkr     # has classes for tick-locating and -formatting
from functools import partial

class gridmap2:
    def __init__(self,  maxX, maxY, res):
        self.width = int(np.ceil(maxX / res))
        self.height = int(np.ceil(maxY / res))

        self.res = res #higher scale <-> lower resolution

        #construct with uniform map
        self.gridOcc = np.zeros((self.height,self.width))
        self.gridFree = np.zeros((self.height,self.width))
        self.gridLogOdds = p2logodds(0.5*np.ones((self.height,self.width)))

        self.log_pm_zocc = p2logodds(0.8) #p(m=occupied|z=hit,x)
        self.log_pm_zfree = p2logodds(0.6) #p(m=free|z=hit,x)
        
        self.log_pm_zocc_neighbor = p2logodds(0.7) #p(m=occupied|z=hit,x,isneigbor)
        self.log_pm_zfree_neighbor = p2logodds(0.55) #p(m=free|z=hit,x,isneigbor)

        self.pose  = pose2(0,0,0) #grid map transform. grid map exists only in positive quadrant

    def update(self,cells_occ, cells_free, n = 0):
        # n - distance in cells for neighbors
        for c in cells_occ:
            if self.cell_in_gridmap(c):
                self.gridOcc[c[0],c[1]] += 1
                self.gridLogOdds[c[0],c[1]] += self.log_pm_zocc 
                
                neighbors= self.neighbors(c,n)
                for cn in neighbors:
                    self.gridLogOdds[cn[0],cn[1]] += self.log_pm_zocc_neighbor 
        for c in cells_free:
            if self.cell_in_gridmap(c):
                self.gridFree[c[0],c[1]] += 1
                self.gridLogOdds[c[0],c[1]] -= self.log_pm_zfree

                neighbors= self.neighbors(c,n)
                for cn in neighbors:
                    self.gridLogOdds[cn[0],cn[1]] -= self.log_pm_zfree_neighbor 
        return

    def worldToMap(self,xy): #continous2discrete ~ worldToMap
        #we assume order <x,y> is provided. if array it is of size 2xm
        
        # We simplify the whole grid situations to this setting:
        # → x
        #↓
        #y
        #* there are no negative values in the map
    
        yx = np.flipud(self.pose.transformTo(xy))
        ij = np.round(yx / self.res).astype(int)
        return ij

    def MapToWorld(self,ij): #discrete2continous ~ mapToWorld
        xy = self.pose.transformFrom((np.flipud(ij)+0.5) * self.res)
        return (xy)

    def neighbors(self,c,a = 1):
        if a == 0:
            return []
        bot = max(c[0]-a,0)
        top = min(c[0]+a,self.height-1)
        left = max(c[1]-a,0)
        right = min(c[1]+a,self.width-1)

        i = np.arange(bot,top+1)
        j = np.arange(left,right+1)
        iijj = np.meshgrid(i,j)
        
        return np.hstack((iijj[0].reshape(-1,1),iijj[1].reshape(-1,1))).tolist()

    def get_pGrid(self):
        return logodds2p(self.gridLogOdds)

    def cell_in_gridmap(self,c):
        return (0 <= c[0] < self.height and 0 <= c[1] < self.width)

    def show(self,ax : plt.Axes = None):
        if ax == None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_xlabel('x'); ax.set_ylabel('y'); 
            ax.set_aspect('equal'); ax.grid()
        
        # cell holds probability of being occupied, which we want to paint black. hence data = 1-self.grid
        # unlike imshow, pcolor flips the data matrix on the vertical axis. Which is something we want built in.
        # also: pcolorfast is way faster than regular pcolor
        
        # grid = self.gridOcc/(self.gridOcc + self.gridFree)
        grid = self.get_pGrid()
        ax.pcolorfast(1-grid,  vmin=0.0, vmax=1.0, cmap=plt.cm.gray)
        
        fmt = tkr.FuncFormatter(partial(numfmt,self.res))
        ax.yaxis.set_major_formatter(fmt)
        ax.xaxis.set_major_formatter(fmt)

        return ax