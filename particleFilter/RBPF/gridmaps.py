from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from particleFilter.geometry import pose2
import matplotlib.ticker as tkr     # has classes for tick-locating and -formatting
from functools import partial
from particleFilter.RBPF.sensors import laser2

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
        
    def c2d(self,xy): #continous2discrete ~ worldToMap
        #we assume order <x,y> is provided. if array it is of size 2xm
        
        # We simplify the whole grid situations to this setting:
        # → x
        #↓
        #y
        #* there are no negative values in the map
        
        yx = np.flipud(self.pose.transformTo(xy))
        ij = np.round(yx / self.res).astype(int)
        return ij

    def d2c(self,ij): #discrete2continous ~ mapToWorld
        xy = self.pose.transformFrom((np.flipud(ij)+0.5) * self.res)
        return (xy)

    def update(self,c_occ,c_free):
        for c in c_occ:
            if 0 <= c[0] < self.height and 0 <= c[1] < self.width:
                self.gridOcc[c[0],c[1]] += 1
                self.gridLogOdds[c[0],c[1]] += self.log_pm_zocc 
                
                neighbors= self.neighbors(c)
                for cn in neighbors:
                    self.gridLogOdds[cn[0],cn[1]] += self.log_pm_zocc_neighbor 

        for c in c_free:
            if 0 <= c[0] < self.height and 0 <= c[1] < self.width:
                #product of c2d - (i,j)
                self.gridFree[c[0],c[1]] += 1
                self.gridLogOdds[c[0],c[1]] -= self.log_pm_zfree

                neighbors= self.neighbors(c)
                for cn in neighbors:
                    self.gridLogOdds[cn[0],cn[1]] -= self.log_pm_zfree_neighbor


    def inverse_measurement_model(self, x: pose2, z: np.ndarray, laser : laser2):
        #based on "Algorithm inverse_range_sensor_model" from page 288 Probalistic Robotics
        c_occ = []
        c_free = []
        for i,zi in enumerate(z):
            idx_occ = np.argwhere((abs(laser.localraymap[i].z - zi) < laser.alpha/2) & (laser.localraymap[i].z < laser.zmax)).squeeze()
            if np.any(idx_occ):
                c_occ.append(self.c2d(x.transformFrom(laser.localraymap[i].t[:,idx_occ])))
    
            idx_free = np.argwhere(abs(laser.localraymap[i].z < zi)).squeeze()
            if np.any(idx_free):
                c_free.append(self.c2d(x.transformFrom(laser.localraymap[i].t[:,idx_free])))

        return np.hstack(c_occ), np.hstack(c_free)

    def neighbors(self,c,a = 1):
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

def p2logodds(p):
    return np.log(p / (1 - p))

def logodds2p(l):
    return 1 - 1 / (1 + np.exp(l))
        
def numfmt(v, x, pos): # your custom formatter function: divide by 100.0
    s = '{}'.format(x * v)
    return s
