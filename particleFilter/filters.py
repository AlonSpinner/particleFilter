import numpy as np
from particleFilter.geometry import pose2
from particleFilter.maps import Map
from particleFilter.gaussians import gauss_likelihood, gauss_fit

class pf_vanila_SE2:
    def __init__(self,m : Map ,initial_states : list[pose2]):
        self.N_PARTICLE : int = len(initial_states) #amount of particles
        self.STATE_SIZE : int = 3
        
        self.particles = initial_states
        self.m = m # map must have method forward_measurement_model(x)
        
        self.weights : np.ndarray = np.ones(self.N_PARTICLE) * 1/self.N_PARTICLE
        self.ETA_THRESHOLD : float = 2.0/self.N_PARTICLE #threshold for performing resampling
        self.EPS = 1e-10
        return

    def step(self,z,z_cov,
                    u,u_cov):
        
        #update particles
        for i in range(self.N_PARTICLE):
            
            #create proposal distribution
            noise = np.random.multivariate_normal(np.zeros((self.STATE_SIZE)),u_cov)
            noise = pose2(noise[0],noise[1],noise[2])
            self.particles[i] = self.particles[i] + (u + noise)
            
            #create target distribution
            zhat = self.m.forward_measurement_model(self.particles[i])
            self.weights[i] *= gauss_likelihood(z,zhat,z_cov)

        #normalize
        self.weights = self.weights/(self.weights.sum())# + self.EPS)

        #resample
        n_eff = self.weights.dot(self.weights)
        if n_eff < self.ETA_THRESHOLD:
            print('resampling')
            self.low_variance_sampler()

    def low_variance_sampler(self):
        r = np.random.uniform()/self.N_PARTICLE
        idx = 0
        c = self.weights[idx]
        new_particles = []
        for i in range(self.N_PARTICLE):
            u = r + i*1/self.N_PARTICLE
            while u > c:
                idx += 1
                c += self.weights[idx]
            new_particles.append(self.particles[idx])
        
        self.particles = new_particles
        self.weights = np.ones(self.N_PARTICLE) * 1/self.N_PARTICLE

    def bestEstimate(self):
        locals = np.array([p.local() for p in self.particles]).T
        mu, cov = gauss_fit(locals, self.weights)       
        return mu,cov