from particleFilter.geometry import pose2
from particleFilter.RBPF.gridmaps import gridmap2
from particleFilter.RBPF.sensors import laser
from particleFilter.RBPF.utils import bresenham2, logodds2p
import numpy as np

def inverse_measurement_model(sensor : laser, m : gridmap2, x : pose2, z : float):
    # first: we update for each ray individually. each ray is a sensor
    #
    # second: given ray measurement ai, zi  we need to understand which cells in the map were hit and which werent.
        #I've noticed two ways of doing this
        #a) Using iterators: <----------------------------------- WHAT WE DO CURRENTLY ("LASER")
            #create points from ai,zi in robot system.
            #transfer these points to map (requires discrtization) -> occupied cells
            #solve with iterators to understand free cells. easiest: bresenham (assume ray width is 0 along path)
        #b) the better way - interpolation between grid maps (gmapping and others):
            # creating a 'egoraymap' which is a grid in the robot's ego system of size [2*zmax,2*zmax]
            #    each cell in the raymap is given a range and a bearing value from (0,0) where the robot 'sits'
            #    as we know the angles of LRF we can precompute the cells 'belonging' to each angle
            #       thus: given zi range measurement, with 1 for loop we can decide 'occ' or 'free' for all cell belonging to rayi
            #    after deciding on cells in the egoraymap, we need to transfer the knowledge to the acutal map
            #       basic but wrong: just transform points from egoraymap to map.
            #           moving from disc -> cont -> rotating -> disc will create 'holes' in our estimation
            #       soltuion: precompute area in map, transform to egoraymap and interpolate. this might be expensive.

    #somehow inspired from:
    #from probablistic robotics chapter 9, table 9.2 page 288 - "inverse_range_sensor_model"

    #treet every zi as if it is an independent measurement
    disc_x = m.worldToMap(x.t())
    for ai,zi in zip(sensor.angles,z):
        dp = zi * [np.cos(ai), np.sin(ai)].reshape(-1,1)
        lm = x.transformFrom(np.array(dp))
        disc_lm = m.worldToMap(lm)
        c_occ = [disc_lm]
        c_free = bresenham2(disc_x[0],disc_x[1],disc_lm[0],disc_lm[1])
    return c_occ, c_free

# def measurementProbability(self,x: pose2, a : np.ndarray ,z : np.ndarray, z_cov : np.ndarray)->float:
#     #ajdusted from probablistic robotics: "beam_range_finder_model" - page 158
#     #sample ML measurement zhat from p(z|x,m)
    
#     #p(z|x,m) ~ N(x.range(m),COV); COV = z_cov * 1/entropy(p(m))
#     #entropy(p(m)) = -p(m)*log(p(m))
#     #transform laser measurements z from ego->world->map
    
#     #create zhat = forward_measurement_model(x,m)
#     #for each angle compute distribution p(z|x,m)

#     #compute p(z) from distribution of 

#     p = 1
#     for ai, zi in zip(a,z):
#         #compute probabilty mass function from cells between robot and zhit (make sure its normalized)
#         #sample probability of hit
#         cells_occ, cells_free = inverse_measurement_model(x,ai,zi)
#         p *= logodds2p(self.gridLogOdds[cells_occ]) / sum(logodds2p(self.gridLogOdds[cells_occ,cells_free]))
#     return p 