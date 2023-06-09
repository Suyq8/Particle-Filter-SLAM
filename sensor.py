import numpy as np
from pr2_utils import read_data_from_csv
from numba import jit
import pandas as pd
import cv2


class Lidar():
    def __init__(self, path):
        self.timestamp, self.data = read_data_from_csv(path)
        self.angles = np.linspace(-5, 185, 286) / 180 * np.pi
        self.idx = 0

    def get_data(self):
        '''
        ranges = self.data[self.idx, :]

        indValid = (ranges < 80) & (ranges > 0.1)
        ranges = ranges[indValid]
        angles = self.angles[indValid]

        xs = ranges*np.cos(angles)
        ys = ranges*np.sin(angles)

        return np.stack((xs, ys, np.zeros_like(xs), np.ones_like(xs)))
        '''
        return _get_data(self.data, self.angles, self.idx)

    @property
    def time(self):
        return self.timestamp[self.idx]

    @property
    def length(self):
        return len(self.timestamp)

    def update_idx(self):
        if self.idx+1 < self.length:
            self.idx += 1


class Encoder():
    def __init__(self, path):
        self.timestamp, self.data = read_data_from_csv(path)
        self.angles = np.linspace(-5, 185, 286) / 180 * np.pi
        self.resolution = 4096
        self.avg_speed = (0.623479*np.diff(self.data[:, 0])+0.622806 *
                          np.diff(self.data[:, 1]))*np.pi/self.resolution\
            / np.diff(self.timestamp)/2*1e9  # in m/s
        self.idx = 0

    def get_data(self):
        return self.avg_speed[self.idx]

    @property
    def time(self):
        return self.timestamp[self.idx]

    @property
    def length(self):
        return len(self.avg_speed)

    def update_idx(self):
        if self.idx+1 < self.length:
            self.idx += 1


class FOG():
    def __init__(self, path):
        self.timestamp, self.data = read_data_from_csv(path)
        tau = np.diff(self.timestamp)/1e9
        tau = np.append(tau, tau[-1])
        self.angular_speed = self.data[:, -1]/tau  # in rad/s
        self.idx = 0

    def get_data(self):
        return self.angular_speed[self.idx]

    @property
    def time(self):
        return self.timestamp[self.idx]

    @property
    def length(self):
        return len(self.timestamp)

    @property
    def tau(self):
        if self.idx+1 < self.length:
            return (self.timestamp[self.idx+1]-self.timestamp[self.idx])/1e9
        else:
            return (self.timestamp[-1]-self.timestamp[-2])/1e9

    def update_idx(self):
        if self.idx+1 < self.length:
            self.idx += 1


class StereoCamera:
    def __init__(self, path_timestamp_l, path_timestamp_r):
        self.path_l = 'stereo_left/'
        self.path_r = 'stereo_right/'
        self.timestamp_l = pd.read_csv(
            path_timestamp_l, header=None).values.reshape(-1)
        self.timestamp_r = pd.read_csv(
            path_timestamp_r, header=None).values.reshape(-1)
        self.idx = 0

        self.w = 1280
        self.h = 560

        self.mtx_l = np.array([[8.1690378992770002e+02, 5.0510166700000003e-01, 6.0850726281690004e+02],
                               [0., 8.1156803828490001e+02,
                                2.6347599764440002e+02],
                               [0., 0., 1.]])

        self.dist_l = np.array([-5.6143027800000002e-02, 1.3952563200000001e-01,
                                -1.2155906999999999e-03, -9.7281389999999998e-04, -8.0878168799999997e-02])

        self.newcameramtx_l, self.roi_l = cv2.getOptimalNewCameraMatrix(
            self.mtx_l, self.dist_l, (self.w, self.h), 1, (self.w, self.h))

        self.mtx_r = np.array([[8.1378205539589999e+02, 3.4880336220000002e-01, 6.1386419539320002e+02],
                               [0., 8.0852165574269998e+02,
                                   2.4941049348650000e+02],
                               [0., 0., 1.]])

        self.dist_r = np.array([-5.4921981799999998e-02, 1.4243657430000001e-01,
                               7.5412299999999996e-05, -6.7560530000000001e-04, -8.5665408299999996e-02])

        self.newcameramtx_r, self.roi_r = cv2.getOptimalNewCameraMatrix(
            self.mtx_r, self.dist_r, (self.w, self.h), 1, (self.w, self.h))

    def compute_stereo(self, numDisparities=32, blockSize=9, undistort=False, a=10):
        path_l = f'{self.path_l}{self.timestamp_l[self.idx]}.png'
        path_r = f'{self.path_r}{self.timestamp_r[self.idx]}.png'

        image_l = cv2.imread(path_l, 0)
        image_r = cv2.imread(path_r, 0)

        image_l = cv2.cvtColor(image_l, cv2.COLOR_BAYER_BG2BGR)
        image_r = cv2.cvtColor(image_r, cv2.COLOR_BAYER_BG2BGR)

        if undistort:
            image_l = self.undistort_l(image_l)
            image_r = self.undistort_r(image_r)

        image_l_gray = cv2.cvtColor(image_l, cv2.COLOR_BGR2GRAY)
        image_r_gray = cv2.cvtColor(image_r, cv2.COLOR_BGR2GRAY)

        # You may need to fine-tune the variables `numDisparities` and `blockSize` based on the desired accuracy
        stereo = cv2.StereoBM_create(
            numDisparities=numDisparities, blockSize=blockSize)

        '''
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
        right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
        left_disp = left_matcher.compute(image_l_gray, image_r_gray)
        right_disp = right_matcher.compute(image_r_gray, image_l_gray)

        filtered_disp = wls_filter.filter(left_disp, image_l, disparity_map_right=right_disp)
        '''
        disp = stereo.compute(image_l_gray, image_r_gray).astype(np.float32)/16

        return disp, image_l

    @property
    def length(self):
        return len(self.timestamp_l)

    @property
    def time(self):
        return self.timestamp_l[self.idx]

    def update_idx(self):
        self.idx += 1

    def undistort_l(self, img):
        dst = cv2.undistort(img, self.mtx_l, self.dist_l,
                            None, self.newcameramtx_l)
        dst = dst[self.roi_l[1]:self.roi_l[1]+self.roi_l[3],
                  self.roi_l[0]:self.roi_l[0]+self.roi_l[2]]
        dst = cv2.resize(dst, (self.w, self.h))

        return dst

    def undistort_r(self, img):
        dst = cv2.undistort(img, self.mtx_r, self.dist_r,
                            None, self.newcameramtx_r)
        dst = dst[self.roi_r[1]:self.roi_r[1]+self.roi_r[3],
                  self.roi_r[0]:self.roi_r[0]+self.roi_r[2]]
        dst = cv2.resize(dst, (self.w, self.h))

        return dst

    def is_end(self):
        return self.idx == self.length


@jit(nopython=True, cache=True)
def _get_data(data, angles, idx):
    ranges = data[idx, :]

    indValid = (ranges < 60) & (ranges > 0.1)
    ranges = ranges[indValid]
    angles = angles[indValid]

    xs = ranges*np.cos(angles)
    ys = ranges*np.sin(angles)

    return np.stack((xs, ys, np.zeros_like(xs), np.ones_like(xs)))
