from dataclasses import dataclass
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
        self.gridLogOdds = p2logodds(0.5*np.ones((self.height,self.width)))

        self.log_pm_zocc = p2logodds(0.8) #p(m=occupied|z=hit,x)
        self.log_pm_zfree = p2logodds(0.6) #p(m=free|z=hit,x)
        
        self.log_pm_zocc_neighbor = p2logodds(0.7) #p(m=occupied|z=hit,x,isneigbor)
        self.log_pm_zfree_neighbor = p2logodds(0.55) #p(m=free|z=hit,x,isneigbor)

        self.zmax = 5.0 #sensor max range
        self.beta = np.radians(10) #ray width in rad
        self.alpha = 0.1 #[m] - wallthickness parameter
        self.angles = np.linspace(-np.pi,+np.pi,180)
        self.raymap = self.localgrid2raymap()

        self.pose  = pose2(-self.zmax,-self.zmax,0) #grid map transform. In this world there exists only positives
        
    def c2d(self,xy): #continous2discrete
        #we assume order <x,y> is provided. if array it is of size 2xm
        
        # We simplify the whole grid situations to this setting:
        # → x
        #↓
        #y
        #* there are no negative values in the map
        
        yx = np.fliplr(self.pose.transformTo(xy))
        ij = np.round(yx / self.res).astype(int)
        return ij

    def d2c(self,loc): #discrete2continous
        i, j = loc #unpack
        x = (j + 0.5) * self.res
        y = (i + 0.5) * self.res
        return (x,y)

    def update(self,c_occ,c_free):
        for c in c_occ:
            if 0 <= c[0] < self.height and 0 <= c[1] < self.width:
                self.gridOcc[c[0],c[1]] += 1
                self.gridLogOdds[c[0],c[1]] += self.log_pm_zocc 
                
                neighbors= self.neighbors2(c)
                for cn in neighbors:
                    self.gridLogOdds[cn[0],cn[1]] += self.log_pm_zocc_neighbor 

        for c in c_free:
            if 0 <= c[0] < self.height and 0 <= c[1] < self.width:
                #product of c2d - (i,j)
                self.gridFree[c[0],c[1]] += 1
                self.gridLogOdds[c[0],c[1]] -= self.log_pm_zfree

                neighbors= self.neighbors2(c)
                for cn in neighbors:
                    self.gridLogOdds[cn[0],cn[1]] -= self.log_pm_zfree_neighbor

    def localGrid(self):
        #create local grid - should be called only once when initalizing
        #-------------
        #|         k |
        #|      ---->|
        #|           |
        #|___________|
        # <----n---->
        k = int(self.zmax/np.sqrt(2) / self.res) #sqrt(2) due to pythgoras. max range is a diagonal
        n = 2 * k + 1

        r = np.linspace(-self.zmax, +self.zmax, n)
        x_grid, y_grid = np.meshgrid(r,r)
        bearing_grid = np.arctan2(y_grid,x_grid)
        range_grid = np.sqrt(x_grid**2 + y_grid**2)
        return x_grid, y_grid, bearing_grid, range_grid

    def localgrid2raymap(self)->list:
        x_grid, y_grid, bearing_grid, range_grid = self.localGrid()
        raymap = []
        for a in self.angles:
            e = np.angle(np.exp(bearing_grid*1j)*np.exp(-a*1j)) #need to use exponential mapping to solve for cut-offs in bearing map
            idx = np.argwhere(abs(e) < self.beta)
            x = x_grid[idx[:, 0], idx[:, 1]]
            y = y_grid[idx[:, 0], idx[:, 1]]
            t = np.vstack((x,y))
            z = range_grid[idx[:, 0], idx[:, 1]]
            raymap.append(ray(a,t,z,idx))
        return raymap

    def inverse_measurement_model2(self, x: pose2, z: np.ndarray):
        #based on "Algorithm inverse_range_sensor_model" from page 288 Probalistic Robotics
        rm = self.raymap
        c_occ = []
        c_free = []
        for i,zi in enumerate(z):   
            # self.c2d(x.transformFrom(rm[i].t))
            occ = np.argwhere((abs(rm[i].z - zi) < self.alpha/2) & (rm[i].z < self.zmax))
            free = np.argwhere(abs(rm[i].z < zi))

            c_occ.extend(x.c2d(rm[i].t[occ]))
            c_free.extend(x.c2d(rm[i].t[free]))
        return c_occ, c_free

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

    def neighbors2(self,c,a = 1):
        bot = max(c[0]-a,0)
        top = min(c[0]+a,self.height-1)
        left = max(c[1]-a,0)
        right = min(c[1]+a,self.width-1)
        
        i = np.arange(bot,top+1)
        j = np.arange(left,right+1)
        iijj = np.meshgrid(i,j)
             
        return np.hstack((iijj[0].reshape(-1,1),iijj[1].reshape(-1,1))).tolist()

    def neighbors(self,c):
        #taken from https://github.com/Adrianndp/DjikstraVis/blob/master/brain.py
        x = c[0]; y = c[1]
        neighbors = []
        width = self.width - 1
        if (x + 1) < width:
            neighbors.append((x + 1, y))
        if (y + 1) < width:
            neighbors.append((x, y + 1))
        if (y + 1) < width and (x + 1) < width:
            neighbors.append((x + 1, y + 1))
        if (x - 1) > 0:
            neighbors.append((x -1, y))
        if (y - 1) > 0:
            neighbors.append((x, y - 1))
        if (x - 1) > 0 and (y - 1) > 0:
            neighbors.append((x - 1, y - 1))
        if (x + 1) < width and (y - 1) > 0:
            neighbors.append((x + 1, y - 1))
        if (x - 1) > 0 and (y + 1) < width:
            neighbors.append((x - 1, y + 1))
        return neighbors

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

def bresenham2(sx, sy, ex, ey):
    #from https://github.com/daizhirui/Bresenham2D/blob/main/bresenham2Dv1.py
    #notes that there exists a better scikit version
    """ Bresenham's ray tracing algorithm in 2D.
    :param sx: x of start point of ray
    :param sy: y of start point of ray
    :param ex: x of end point of ray
    :param ey: y of end point of ray
    :return: cells along the ray
    """
    dx = abs(ex - sx)
    dy = abs(ey - sy)    @dataclass(frozen = True)
    class ray:
        a : float #angle in radians
        x : np.ndarray
        y : np.ndarray
        z : np.ndarray
    steep = abs(dy) > abs(dx)
    if steep:
        dx, dy = dy, dx  # swap

    if dy == 0:
        q = np.zeros((dx + 1, 1), dtype=int)
    else:
        q = np.append(0, np.greater_equal(
            np.diff(
                np.mod(np.arange(  # If d exceed dx, decrease d by dx
                    np.floor(dx / 2), -dy * dx + np.floor(dx / 2) - 1, -dy,
                    dtype=int), dx
                )  # offset np.floor(dx / 2) to compare d with 0.5dx
            ), 0))

    if steep:
        if sy <= ey:
            y = np.arange(sy, ey + 1)
        else:
            y = np.arange(sy, ey - 1, -1)
        if sx <= ex:
            x = sx + np.cumsum(q)
        else:
            x = sx - np.cumsum(q)
    else:
        if sx <= ex:
            x = np.arange(sx, ex + 1)
        else:
            x = np.arange(sx, ex - 1, -1)
        if sy <= ey:
            y = sy + np.cumsum(q)
        else:
            y = sy - np.cumsum(q)

    bres = np.vstack((x,y)).T.tolist()
    return bres

@dataclass(frozen = True)
class ray:
    a : float #angle in radians
    t : np.ndarray # locations of middle of cells (2xm)
    z : np.ndarray # ranges to cells from (0,0)
    idx : np.ndarray