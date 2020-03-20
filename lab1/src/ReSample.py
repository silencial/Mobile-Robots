#!/usr/bin/env python
'''
  Provides methods for re-sampling from a distribution represented by weighted samples
'''

import rospy
import numpy as np
from threading import Lock


class ReSampler:
    '''
    Initializes the resampler
        particles: The particles to sample from
        weights: The weights of each particle
        state_lock: Controls access to particles and weights
    '''
    def __init__(self, particles, weights, state_lock=None):
        self.particles = particles
        self.weights = weights

        # Indices for each of the M particles
        self.particle_indices = np.arange(self.particles.shape[0])

        # Bins for partitioning of weight values
        self.step_array = (1.0 / self.particles.shape[0]) * np.arange(self.particles.shape[0], dtype=np.float32)

        if state_lock is None:
            self.state_lock = Lock()
        else:
            self.state_lock = state_lock

    '''
    Student's IMPLEMENT
    Performs in-place, lower variance sampling of particles
        returns: nothing
    '''
    def resample_low_variance(self):
        self.state_lock.acquire()

        # YOUR CODE HERE
        M = self.particles.shape[0]
        M_inverse = 1.0 / M
        u = (np.random.rand() + np.arange(M)) * M_inverse
        c = np.cumsum(self.weights)
        particles = np.empty_like(self.particles)
        i = j = 0
        while j < M:
            if u[j] > c[i]:
                i += 1
            else:
                particles[j] = self.particles[i]
                j += 1
        self.particles[:] = particles[:]

        self.weights[:] = M_inverse

        self.state_lock.release()
