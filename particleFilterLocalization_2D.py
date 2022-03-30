from typing import Callable
import numpy as np

@dataclass
class particle:
    x : np.ndarray
    weight : float
    noisey_motion_model : Callable

class Map:
    def __init__(self,forward_measurement_model):
        self.forward_measurement_model : Callable

class ParticleFilter:
    def __init__(self,m: Map ,initial_states : list):
        self.N_PARTICLE : int = len(initial_states) #amount of particles
        self.particles = [particle(x,1/self.N_PARTICLE) for x in initial_states]
        self.m : Map
        self.n_threshold : float = self.N_PARTICLE/2.0 #threshold for performing resampling
        return

    def step(self,z,z_cov,
                    u,u_cov):
        #update particles
        weights = np.zeros((self.N,1)) #store weights in an array for resampling
        for i,p in enumerate(self.particles):
            p.noisey_motion_model(u,u_cov)
            zhat = self.m.forward_measurement_model(p.x)
            p.weight *= gauss_likelihood(z,zhat,z_cov)
            weights[i] = p.weight

        #normalize
        sumWeights = weights.sum()
        for p in self.particles:
            p.weight /= sumWeights

        #resample
        #attempts to draw particle i with state x such that p(x) ~ w
        n_eff = weights @ weights.T
        if n_eff < self.n_threshold:
            self.low_variance_sampler()
            new_particles = []
            cumsumWeights = np.cumsum(weights)
            r = np.rand/self.N_PARTICLE
            for i,p in enumerate(self.particles):
                idx = 0
                r += 1/self.N_PARTICLE
                if cumsumWeights[idx] < r:
                    idx += 1
                new_particles.append(particle(x = self.particles[idx].x,
                        weight = 1.0 / self.N_PARTICLE))
            self.particles = new_particles

    def low_variance_sampler(self):
        r = np.rand/self.N_PARTICLE
        c = self.particles[0]
        idx = 0
        new_particles = []
        for i in range(self.N_PARTICLE):
            r += 1/self.N_PARTICLE
            while r > c:
                idx += 1
                c += self.particles[idx].weight
            new_particles.append(particle(x = self.particles[idx].x,
                        weight = 1.0 / self.N_PARTICLE))
        self.particles = new_particles


def gauss_likelihood(x : np.ndarray, mu : np.ndarray, cov : np.ndarray):
    k = x.size
    num =  np.exp(-0.5*(x-mu).T @ cov @ (x-mu))
    den = np.sqrt(2.0 * np.pi ** k * np.linalg.det(cov))
    p = num/den
    return p