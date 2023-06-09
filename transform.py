import numpy as np
from numba import jit

vehicle_T_lidar = np.array([[0.00130201, 0.796097, 0.605167, 0.8349],
                            [0.999999, -0.000419027, -0.00160026, -0.0126869],
                            [-0.00102038, 0.605169, -0.796097, 1.76416],
                            [0., 0., 0., 1.]])

vechicle_T_stereo = np.array([[-0.00680499, -0.0153215, 0.99985, 1.64239],
                              [-0.999977, 0.000334627, -0.00680066, 0.247401],
                              [-0.000230383, -0.999883, -0.0153234, 1.58411],
                              [0., 0., 0., 1.]])

pixel_T_optical = np.array([[7.7537235550066748e+02, 0., 6.1947309112548828e+02, 0.],
                            [0., 7.7537235550066748e+02,
                                2.5718049049377441e+02, 0.],
                            [0., 0., 0., 3.6841758740842312e+02],
                            [0., 0., 1., 0.]])

optical_T_pixel = np.linalg.pinv(pixel_T_optical)


@jit(nopython=True, cache=True)
def lidar2world(lidar_data, particle):
    theta = particle[-1]

    world_T_vehicle = np.array([[np.cos(theta), -np.sin(theta), 0, particle[0]],
                                [np.sin(theta), np.cos(theta), 0, particle[1]],
                                [0., 0., 1., 0.],
                                [0., 0., 0., 1.]])

    return world_T_vehicle @ vehicle_T_lidar @ lidar_data


def pixel2world(pixel, particle):
    theta = particle[-1]

    world_T_vehicle = np.array([[np.cos(theta), -np.sin(theta), 0, particle[0]],
                                [np.sin(theta), np.cos(theta), 0, particle[1]],
                                [0., 0., 1., 0.],
                                [0., 0., 0., 1.]])

    optical_xyz = optical_T_pixel @ pixel
    optical_xyz /= optical_xyz[-1, :]

    return world_T_vehicle @ vechicle_T_stereo @ optical_xyz
