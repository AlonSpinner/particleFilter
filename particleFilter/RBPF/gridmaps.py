import numpy as np
import matplotlib.pyplot as plt
from particleFilter.geometry import pose2
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
        self.gridLog = p2logodds(0.5*np.ones((self.height,self.width)))

        self.log_pm_zocc = p2logodds(0.8) #p(m=occupied|z=hit,x)
        self.log_p_zfree = p2logodds(0.6) #p(m=free|z=hit,x)
        
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
        grid = logodds2p(self.gridLog)
        ax.pcolorfast(1-grid,  vmin=0.0, vmax=1.0, cmap=plt.cm.gray)
        
        fmt = tkr.FuncFormatter(partial(numfmt,self.res))
        ax.yaxis.set_major_formatter(fmt)
        ax.xaxis.set_major_formatter(fmt)

        return ax
        
    def c2d(self,loc): #continous2discrete
        # We simplify the whole grid situations to this setting:
        # → x
        #↓
        #y
        
        #* there are no negative values in the map
        #* this is only a storage world. when we want to present, we flip the y axis 

        x,y = loc #unpack
        i = np.int(y / self.res)
        j = np.int(x / self.res)
        return (i,j)

    def d2c(self,loc): #discrete2continous
        i, j = loc #unpack
        x = (j + 0.5) * self.res
        y = (i + 0.5) * self.res
        return (x,y)

    def update(self,c_occ,c_free):
        for c in c_occ:
            if 0 <= c[0] < self.height and 0 <= c[1] < self.width:
                self.gridOcc[c[0],c[1]] += 1
                self.gridLog[c[0],c[1]] += self.log_pm_zocc 
        for c in c_free:
            if 0 <= c[0] < self.height and 0 <= c[1] < self.width:
                #product of c2d - (i,j)
                self.gridFree[c[0],c[1]] += 1
                self.gridLog[c[0],c[1]] -= self.log_p_zfree

    def inverse_measurement_model(self, x : pose2, a : np.ndarray, z : np.ndarray): #returns p(m|xt,zt)
    #inspired from:
    #from probablistic robotics chapter 9, table 9.2
    #Algorithm inverse_range_sensor_model
    
    #zt - array of range measurements correlating to laser's angles
        disc_x = self.c2d(x.t())
        c_occ = []
        c_free = []
        for ai,zi in zip(a,z):
            dp = (zi*[np.cos(ai),np.sin(ai)]).reshape(-1,1)
            lm = x.transformFrom(np.array(dp))
            disc_lm = self.c2d(lm)
            c_occ.append(disc_lm)
            c_free.extend(bresenham2(disc_x[0],disc_x[1],disc_lm[0],disc_lm[1]))
        return c_occ, c_free

        
def p2logodds(p):
    return np.log(p / (1 - p))

def logodds2p(l):
    return 1 - 1 / (1 + np.exp(l))
        
def numfmt(v, x, pos): # your custom formatter function: divide by 100.0
    s = '{}'.format(x * v)
    return s

def bresenham2(x1, y1, x2, y2):
    bres = []
    x = x1
    y = y1
    
    delta_x = np.abs(x2-x1)
    delta_y = np.abs(y2-y1)

    s_x = np.sign(x2 - x1)
    s_y = np.sign(y2 - y1)

    if delta_y > delta_x:
        delta_x, delta_y = delta_y, delta_x
        interchange = True
    else:
        interchange = False

    A = 2*delta_y
    B = 2*(delta_y-delta_x)
    E = 2* delta_y - delta_x

    bres.append((x,y))
    for i in range(1,delta_x):
        if E < 0:
            if interchange:
                y += s_y
            else:
                x += s_x
            E = E + A
        else:
            y += s_y
            x += s_x

        bres.append((x,y))

    return bres