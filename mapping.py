import numpy as np
from pr2_utils import mapCorrelation
import matplotlib.pyplot as plt
from skimage.draw import line
from transform import lidar2world, pixel2world
from numba import jit


class Map:
    def __init__(self, res=0.1, xmin=-50, xmax=50, ymin=-50, ymax=50):
        self.res = res
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

        self.sizex = int(np.ceil((self.xmax - self.xmin) / self.res + 1))
        self.sizey = int(np.ceil((self.ymax - self.ymin) / self.res + 1))
        self.grid_map = np.zeros((self.sizex, self.sizey))

        # x,y-positions of each pixel of the map
        self.x_physical = np.arange(self.xmin, self.xmax+self.res, self.res)
        self.y_physical = np.arange(self.ymin, self.ymax+self.res, self.res)

        t = 4
        self.scan_x_range = np.arange(-t*self.res, (t+1)*self.res, self.res)
        self.scan_y_range = self.scan_x_range.copy()

        self.texture_map = np.zeros((self.sizex, self.sizey, 3))

        # for alignment
        x = np.arange(1280)
        y = np.arange(560)
        self.u, self.v = np.meshgrid(x, y)
        self.u = self.u.reshape(-1)
        self.v = self.v.reshape(-1)
        self.ones = np.ones_like(self.u)

    def map_correlation(self):
        return self.grid_map > 0, self.x_physical, self.y_physical, self.scan_x_range, self.scan_y_range
        # mapCorrelation(self.grid_map > 0, self.x_physical, self.y_physical, lidar_data, self.scan_x_range, self.scan_y_range)

    def update_map(self, particle_max, lidar_data):
        lidar_world = lidar2world(lidar_data, particle_max)
        xis, yis = lidar_world[0, :], lidar_world[1, :]
        x_particle = np.ceil(
            (particle_max[0]-self.xmin)/self.res).astype(np.int16)-1
        y_particle = np.ceil(
            (particle_max[1]-self.ymin)/self.res).astype(np.int16)-1
        xis = np.ceil((xis-self.xmin)/self.res).astype(np.int16)-1
        yis = np.ceil((yis-self.ymin)/self.res).astype(np.int16)-1

        for i in range(len(xis)):
            x, y = line(x_particle, y_particle, xis[i], yis[i])
            x = x.astype(np.int16)
            y = y.astype(np.int16)
            x_free = x[:-1]
            y_free = y[:-1]

            indGood = (x_free > 1) & (y_free > 1) & (
                x_free < self.sizex) & (y_free < self.sizey)
            self.grid_map[x_free[indGood], y_free[indGood]] -= np.log(4)

            x_occupied = x[-1]
            y_occupied = y[-1]

            indGood = (x_occupied > 1) & (y_occupied > 1) & (
                x_occupied < self.sizex) & (y_occupied < self.sizey)
            if indGood:
                self.grid_map[x_occupied, y_occupied] += np.log(4)

    def plot_map(self, i, n, show=True):
        map_printed = np.zeros_like(self.grid_map)
        map_printed[self.grid_map > 0] = 0
        map_printed[self.grid_map < 0] = 1
        map_printed[self.grid_map == 0] = 0.5

        map_printed = np.flip(map_printed.T, axis=0)

        fig, ax = plt.subplots(figsize=(15, 13))
        ax.set_title(f"occupancy grid map (i={i}, n={n})")
        ax.imshow(map_printed, cmap="gray", vmin=0, vmax=1)
        fig.savefig(f"occupancy_grid_map_{n}_{i}.png")

        if not show:
            plt.close(fig)

    def update_texture_map(self, particle_max, disparity, img, alpha=0.2):
        h, w = disparity.shape
        disparity = disparity.reshape(w*h)
        img = img.reshape(w*h, 3)
        pixel = self.align(disparity)
        #b = np.ones((h, w), dtype=bool)
        # b[:220,:]=False

        indGood = (disparity > 0) & (img[:, 0] > 20) & (
            img[:, 1] > 20) & (img[:, 2] > 20)  # & b.reshape(w*h)
        pixel_world = pixel2world(pixel[:, indGood], particle_max)
        img = img[indGood, :]
        #p=pixel[:, indGood]

        indGood = (pixel_world[2, :] > -0.1) & (pixel_world[2, :] < 0.2)
        pixel_world = pixel_world[:, indGood]
        img = img[indGood, :]
        #p=p[:, indGood]

        xis = np.ceil((pixel_world[0, :]-self.xmin) /
                      self.res).astype(np.int16)-1
        yis = np.ceil((pixel_world[1, :]-self.ymin) /
                      self.res).astype(np.int16)-1

        indGood = (xis > 1) & (yis > 1) & (
            xis < self.sizex) & (yis < self.sizey)
        img = img[indGood, :]
        '''
        p=p[:, indGood]
        
        plt.scatter(p[0,:],p[1,:])
        plt.show()
        '''

        #self.texture_map[xis[indGood], yis[indGood]]=img
        self.texture_map = loop(
            self.texture_map, xis[indGood], yis[indGood], img, alpha)

    def align(self, disparity):
        return np.vstack((self.u, self.v, disparity, self.ones)).astype(np.float64)

    def plot_texture_map(self, i, n, show=True):
        map_printed = np.round(self.texture_map).astype(np.int32)
        map_printed = np.flip(np.transpose(map_printed, (1, 0, 2)), axis=0)

        fig, ax = plt.subplots(figsize=(15, 13))
        ax.set_title(f"texture map (i={i}, n={n})")
        ax.imshow(map_printed)
        fig.savefig(f"texture_map_{n}_{i}.png")

        if not show:
            plt.close(fig)

    def plot_texture_map_trimmed(self, i, n, show=True):
        map_printed = np.round(self.texture_map).astype(np.int32)
        indGood = (self.grid_map == 0)
        map_printed[indGood, :] = 0
        map_printed = np.flip(np.transpose(map_printed, (1, 0, 2)), axis=0)

        fig, ax = plt.subplots(figsize=(15, 13))
        ax.set_title(f"texture map trim (i={i}, n={n})")
        ax.imshow(map_printed)
        fig.savefig(f"texture_map_trim_{n}_{i}.png")

        if not show:
            plt.close(fig)


@jit(nopython=True, cache=True)
def loop(texture_map, xis, yis, img, alpha):
    for i in range(len(xis)):
        if np.sum(texture_map[xis[i], yis[i]]) == 0:
            texture_map[xis[i], yis[i]] = img[i]
        else:
            texture_map[xis[i], yis[i]] = (
                1-alpha)*texture_map[xis[i], yis[i]] + alpha*img[i]

    return texture_map
