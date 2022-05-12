from dataclasses import dataclass
from particleFilter.geometry import pose2
from particleFilter.RBPF.gridmaps import gridmap2
from particleFilter.RBPF.sensors import laser
from particleFilter.RBPF.models import inverse_measurement_model, measurement_probability
from particleFilter.RBPF.utils import flatten_list
from copy import deepcopy
import numpy as np
import time

START_TIME = time.time()

@dataclass(frozen = False)
class particle2:
        pose : pose2
        map : gridmap2

class RBPF:
    def __init__(self, m : gridmap2 ,initial_poses : list[pose2], sensor : laser):
        
        self.N_PARTICLES : int = len(initial_poses) #amount of particles
        self.particles = [particle2(ip, m) for ip in initial_poses]
        self.weights : np.ndarray = np.ones(self.N_PARTICLES) * 1/self.N_PARTICLES
        self.sensor : laser = sensor
        
        self.ETA_THRESHOLD : float = 4.0/self.N_PARTICLES # bigger - lower threshold

        self.verbose = True

    def step(self, z, z_cov, u, u_cov):
        
        #update particles
        for i in range(self.N_PARTICLES):
            
            #create proposal distribution - move particle poses 
            noise = np.random.multivariate_normal(np.zeros(u.size),u_cov)
            self.particles[i].pose += u + pose2(noise[0],noise[1],noise[2])
            
            #update weights
            self.weights[i] *= measurement_probability(self.sensor,self.particles[i].map,self.particles[i].pose,z,z_cov)

            #update map
            c_occ, c_free = inverse_measurement_model(self.sensor,self.particles[i].map,self.particles[i].pose,z)
            self.particles[i].map.update(flatten_list(c_occ),flatten_list(c_free),1)

        #normalize
        sm = self.weights.sum()
        if sm == 0.0: #numerical errors can cause this if particles have diverged from solution
            if self.verbose: print(f'{time.time() - START_TIME}[s]: numerically caused weight reset')
            self.weights = np.ones(self.N_PARTICLES) * 1/self.N_PARTICLES
        else:
            self.weights = self.weights/sm
            
        #resample
        n_eff = self.weights.dot(self.weights)
        if n_eff < self.ETA_THRESHOLD:
            if self.verbose: print(f'{time.time() - START_TIME}[s]: resampling')
            self.low_variance_sampler()

    def low_variance_sampler(self):
        r = np.random.uniform()/self.N_PARTICLES
        idx = 0
        c = self.weights[idx]
        new_particles = []
        for i in range(self.N_PARTICLES):
            u = r + i*1/self.N_PARTICLES
            while u > c:
                idx += 1
                c += self.weights[idx]
            new_particles.append(deepcopy(self.particles[idx]))
        
        self.particles = new_particles
        self.weights = np.ones(self.N_PARTICLES) * 1/self.N_PARTICLES