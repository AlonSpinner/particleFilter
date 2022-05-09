import numpy as np
from particleFilter.geometry import pose2
from dataclasses import dataclass

class laser2():
    def __init__(self, angles = np.linspace(-np.pi,np.pi,180), zmax = 5.0, beta = np.radians(10), alpha = 0.2, res = 0.1):

        self.pose  = pose2(0,0,0)

        self.zmax : float = zmax #sensor max range
        self.beta : float = beta #ray width parameter ~ camera pixel size
        self.alpha : float = alpha #[m] - wallthickness parameter
        self.angles : np.ndarray = angles
        self.res : float = res #resolution of queries 
        
        self.localraymap : list[ray] = self.localgrid2raymap()


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

    def localgrid2raymap(self): #list of rays
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
            
@dataclass(frozen = True)
class ray:
    a : float #angle in radians
    t : np.ndarray # locations of middle of cells (2xm)
    z : np.ndarray # ranges to cells from (0,0)
    idx : np.ndarray #indcies in local grid map. here for debugging purposes. not expensive as is constructed once