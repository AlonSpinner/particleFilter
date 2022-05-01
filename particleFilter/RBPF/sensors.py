import numpy as np
from particleFilter.geometry import pose2
from particleFilter.RBPF.gridmaps import gridmap2

class laser2:
    
    def __init__(self, angles):
        self.angles = angles
        self.alpha = 0.1 # object width in meters
        self.beta = np.radians(3) #beam anglar width in radians
        self.zmax = 10 #meters

    def inverse_measurement_model(self, x : pose2, z : np.ndarray, m : gridmap2): #returns p(m|xt,zt)
        #inspired from:
        #from probablistic robotics chapter 9, table 9.2
        #Algorithm inverse_range_sensor_model
        
        #zt - array of range measurements correlating to laser's angles
        
        cells = self.questionCells(x,m)
        update = []

        for c in cells:
            lm = np.array(m.d2c(c)).reshape(-1,1)
            r = x.range(lm)
            phi = x.bearing(lm)
            k = np.argmin(phi - self.angles) #find index of fitting ray

            if r > min(self.zmax,z[k]+self.alpha/2) or abs(self.angles[k]-phi) > self.beta/2:
                update.append('prior')
            elif z[k] < self.zmax and abs(r-self.zmax) < self.alpha/2:
                update.append('occ')
            else:
                update.append('free')

        return cells, update

    def questionCells(self, x : pose2, m : gridmap2):
        #used when applying forward_measurement_model OR inverse_measurement_model
        #we dont want to loop over the entire map when measuring locally.
        c = m.c2d(tuple(x.t()))
        aI = np.ceil((self.zmax + self.alpha/2) * m.scaleX)
        aJ = np.ceil((self.zmax + self.alpha/2) * m.scaleY)
            
        i_bracket = np.clip(np.array([c[0] - aI,c[0] + aI]),0,m.width-1).astype(int)
        j_bracket = np.clip(np.array([c[1] - aJ,c[1] + aJ]),0,m.height-1).astype(int)

        cells = []
        for i in np.arange(i_bracket[0],i_bracket[1]):
            for j in np.arange(j_bracket[0],j_bracket[1]):
                cells.append((i,j))
                
        return cells
