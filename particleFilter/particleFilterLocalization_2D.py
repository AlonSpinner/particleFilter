from typing import Callable
import numpy as np

class ParticleFilter:
    def __init__(self,m ,initial_states : list):
        self.N_PARTICLE : int = len(initial_states) #amount of particles
        self.STATE_SIZE : int = initial_states[0].size
        
        self.particles = initial_states #particle must have method createAlike(params)
        self.m : m # map must have method forward_measurement_model(x)
        
        self.weights : np.ndarray = np.ones((self.N_PARTICLE,1)) * 1/self.N_PARTICLE
        self.n_threshold : float = self.N_PARTICLE/2.0 #threshold for performing resampling
        return

    def step(self,z,z_cov,
                    u,u_cov):
        
        #update particles
        weights = np.zeros((self.N_PARTICLE,1)) #store weights in an array for resampling
        for i,p in enumerate(self.particles):
            
            #create proposal distribution
            noise = np.random.multivariate_normal(np.zeros((self.STATE_SIZE)),u_cov)
            noise = p.createAlike(noise[0],noise[1],noise[2]) #allow for all kinds of classes
            p = p + (u + noise)
            
            #create target distribution
            zhat = self.m.forward_measurement_model(p)
            self.weights[i] *= gauss_likelihood(z,zhat,z_cov)

        #normalize
        self.weights = self.weights/self.weights.sum()

        #resample
        #attempts to draw particle i with state x such that p(x) ~ w
        n_eff = self.weights.T @ weights
        if n_eff < self.n_threshold:
            self.low_variance_sampler()

    def low_variance_sampler(self):
        r = np.random.rand()/self.N_PARTICLE
        c = self.particles[0]
        idx = 0
        new_particles = []
        for i in range(self.N_PARTICLE):
            r += 1/self.N_PARTICLE
            while r > c:
                idx += 1
                c += self.weights[idx]
            new_particles.append(self.particles[idx])
        
        self.particles = new_particles
        self.weights = np.ones((self.N_PARTICLE,1)) * 1/self.N_PARTICLE


def gauss_likelihood(x : np.ndarray, mu : np.ndarray, cov : np.ndarray):
    k = x.size
    num =  np.exp(-0.5*(x-mu).T @ np.linalg.inv(cov) @ (x-mu))
    den = np.sqrt(2.0 * np.pi ** k * np.linalg.det(cov))
    p = num/den
    return p