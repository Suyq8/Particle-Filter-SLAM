import numpy as np
from transform import lidar2world
from scipy.special import logsumexp
from numba import jit
from pr2_utils import mapCorrelation


class Particle_Filter:
    def __init__(self, n_particle=40, thres=0.2, noise=False):
        self.n_particle = n_particle
        self.n_thres = int(self.n_particle*thres)
        self.noise = noise
        self.particle = np.zeros((3, self.n_particle))  # [x,y,theta]
        self.alpha = np.ones(self.n_particle) / \
            self.n_particle  # particle weight

    def predict(self, speed_v, speed_w, tau, sigma_v=0.1, sigma_w=0.01):
        '''
        if self.noise:
            speed_v += np.random.normal(0, sigma, (1, self.n_particle))
            speed_w += np.random.normal(0, sigma, (1, self.n_particle))

        theta = self.particle[-1, :]
        self.particle += tau * np.vstack([speed_v*np.cos(theta),
                                         speed_v*np.sin(theta),
                                         np.ones((1, self.n_particle))*speed_w])

        '''
        self.particle = _predict(
            speed_v, speed_w, tau, self.noise, self.n_particle, self.particle, sigma_v, sigma_w)

    def update(self, lidar_data, Map):
        '''
        correlation = np.zeros_like(self.alpha)

        for i in range(self.n_particle):
            particle = self.particle[:, i]
            lidar_world = lidar2world(lidar_data, particle)

            c = Map.map_correlation(lidar_world)
            correlation[i] = c.max()
        '''
        grid_map, x_physical, y_physical, scan_x_range, scan_y_range = Map.map_correlation()
        correlation = loop(self.n_particle, self.particle, lidar_data,
                           grid_map, x_physical, y_physical, scan_x_range, scan_y_range)

        # update paticle weight
        log_weights = np.log(self.alpha) + correlation
        log_weights -= logsumexp(log_weights)
        self.alpha = np.exp(log_weights)

        n_eff = 1 / np.sum(self.alpha**2)

        if n_eff < self.n_thres:
            self.resampling()

        return self.particle[:, np.argmax(self.alpha)]

    def resampling(self):
        cmf = np.cumsum(self.alpha)
        j = 0
        u0 = np.random.rand()
        idx = []

        for u in ((u0+i)/self.n_particle for i in range(self.n_particle)):
            while u > cmf[j]:
                j += 1
            idx.append(j-1)

        self.particle = self.particle[:, idx]
        self.alpha = self.alpha[idx]
        self.alpha /= np.sum(self.alpha)


@jit(nopython=True, cache=True)
def _predict(speed_v, speed_w, tau, noise, n_particle, particle, sigma_v, sigma_w):
    speed_v = speed_v*np.ones((1, n_particle))
    speed_w = speed_w*np.ones((1, n_particle))
    if noise:
        speed_v = speed_v + np.random.normal(0, sigma_v, (1, n_particle))
        speed_w = speed_w + np.random.normal(0, sigma_w, (1, n_particle))

    theta = particle[2, :]
    particle += tau * np.vstack((speed_v*np.cos(theta),
                                 speed_v*np.sin(theta),
                                 speed_w))

    return particle


@jit(nopython=True, cache=True)
def loop(n_particle, particle, lidar_data, grid_map, x_physical, y_physical, scan_x_range, scan_y_range):
    correlation = np.ones(n_particle)
    for i in range(n_particle):
        particle_i = particle[:, i]
        lidar_world = lidar2world(lidar_data, particle_i)

        c = mapCorrelation(grid_map, x_physical, y_physical,
                           lidar_world, scan_x_range, scan_y_range)
        correlation[i] = c.max()

    return correlation
