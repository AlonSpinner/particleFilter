from typing import Callable
import numpy as np

class Map:
    def __init__(self,forward_measurement_model : Callable):
        self.forward_measurement_model : Callable

class ParticleFilter:
    def __init__(self,m: Map ,initial_states : list, noisey_motion_model : Callable):
        self.N_PARTICLE : int = len(initial_states) #amount of particles
        self.particles = initial_states
        self.weights = np.ones((self.N_PARTICLE,1)) * 1/self.N_PARTICLE
        self.noisey_motion_model : Callable = noisey_motion_model
        self.m : Map = m
        self.n_threshold : float = self.N_PARTICLE/2.0 #threshold for performing resampling
        return

    def step(self,z,z_cov,
                    u,u_cov):
        #update particles
        weights = np.zeros((self.N,1)) #store weights in an array for resampling
        for i,p in enumerate(self.particles):
            p = self.noisey_motion_model(p,u,u_cov)
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
    num =  np.exp(-0.5*(x-mu).T @ cov @ (x-mu))
    den = np.sqrt(2.0 * np.pi ** k * np.linalg.det(cov))
    p = num/den
    return p