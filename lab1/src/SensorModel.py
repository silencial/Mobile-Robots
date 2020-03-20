#!/usr/bin/env python
'''
    Weights particles according to their agreement with the observed data
'''
import numpy as np
import rospy
import range_libc
import time
from threading import Lock
from nav_msgs.srv import GetMap
import rosbag
import matplotlib.pyplot as plt
import utils as Utils
from sensor_msgs.msg import LaserScan

THETA_DISCRETIZATION = 112  # Discretization of scanning angle
INV_SQUASH_FACTOR = 0.2  # Factor for helping the weight distribution to be less peaked

# Tune these Values!
Z_SHORT = 0.1  # Weight for short reading
Z_MAX = 0.05  # Weight for max reading
Z_RAND = 0.05  # Weight for random reading
SIGMA_HIT = 8.0  # Noise value for hit reading
Z_HIT = 0.80  # Weight for hit reading


class SensorModel:
    '''
    Initializes the sensor model
        scan_topic: The topic containing laser scans
        laser_ray_step: Step for downsampling laser scans
        exclude_max_range_rays: Whether to exclude rays that are beyond the max range
        max_range_meters: The max range of the laser
        map_msg: A nav_msgs/MapMetaData msg containing the map to use
        particles: The particles to be weighted
        weights: The weights of the particles
        state_lock: Used to control access to particles and weights
    '''
    def __init__(self, scan_topic, laser_ray_step, exclude_max_range_rays, max_range_meters, map_msg, particles, weights, car_length, state_lock=None):
        if state_lock is None:
            self.state_lock = Lock()
        else:
            self.state_lock = state_lock

        self.particles = particles
        self.weights = weights

        self.LASER_RAY_STEP = laser_ray_step  # Step for downsampling laser scans
        self.EXCLUDE_MAX_RANGE_RAYS = exclude_max_range_rays  # Whether to exclude rays that are beyond the max range
        self.MAX_RANGE_METERS = max_range_meters  # The max range of the laser
        self.CAR_LENGTH = car_length

        oMap = range_libc.PyOMap(map_msg)  # A version of the map that range_libc can understand
        max_range_px = int(self.MAX_RANGE_METERS / map_msg.info.resolution)  # The max range in pixels of the laser
        self.range_method = range_libc.PyCDDTCast(oMap, max_range_px, THETA_DISCRETIZATION)  # The range method that will be used for ray casting
        # self.range_method = range_libc.PyRayMarchingGPU(oMap, max_range_px) # The range method that will be used for ray casting
        self.range_method.set_sensor_model(self.precompute_sensor_model(max_range_px))  # Load the sensor model expressed as a table
        self.queries = None
        self.ranges = None
        self.laser_angles = None  # The angles of each ray
        self.downsampled_angles = None  # The angles of the downsampled rays
        self.do_resample = False  # Set so that outside code can know that it's time to resample

        # Subscribe to laser scans
        self.laser_sub = rospy.Subscriber(scan_topic, LaserScan, self.lidar_cb, queue_size=1)

        # Kidnapped
        self.confidence = 0.0

    '''
    Downsamples laser measurements and applies sensor model
        msg: A sensor_msgs/LaserScan
    '''
    def lidar_cb(self, msg):
        self.state_lock.acquire()

        # Down sample the laser rays
        if not self.EXCLUDE_MAX_RANGE_RAYS:
            # Initialize angle arrays
            if not isinstance(self.laser_angles, np.ndarray):
                self.laser_angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
                self.downsampled_angles = np.copy(self.laser_angles[0::self.LASER_RAY_STEP]).astype(np.float32)

                self.downsampled_ranges = np.array(msg.ranges[::self.LASER_RAY_STEP])  # Down sample
                self.downsampled_ranges[np.isnan(self.downsampled_ranges)] = self.MAX_RANGE_METERS  # Remove nans
                self.downsampled_ranges[self.downsampled_ranges[:] == 0] = self.MAX_RANGE_METERS  # Remove 0 values
        else:
            # Initialize angle array
            if not isinstance(self.laser_angles, np.ndarray):
                self.laser_angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
            ranges = np.array(msg.ranges)  # Get the measurements
            ranges[np.isnan(ranges)] = self.MAX_RANGE_METERS  # Remove nans
            valid_indices = np.logical_and(ranges > 0.01, ranges < self.MAX_RANGE_METERS)  # Find non-extreme measurements
            self.filtered_angles = np.copy(self.laser_angles[valid_indices]).astype(np.float32)  # Get angles corresponding to non-extreme measurements
            self.filtered_ranges = np.copy(ranges[valid_indices]).astype(np.float32)  # Get non-extreme measurements

            ray_count = int(self.laser_angles.shape[0] / self.LASER_RAY_STEP)  # Compute expected number of rays
            sample_indices = np.arange(0, self.filtered_angles.shape[0], float(self.filtered_angles.shape[0]) / ray_count).astype(np.int)  # Get downsample indices
            self.downsampled_angles = np.zeros(ray_count + 1, dtype=np.float32)  # Initialize down sample angles
            self.downsampled_ranges = np.zeros(ray_count + 1, dtype=np.float32)  # Initialize down sample measurements
            self.downsampled_angles[:sample_indices.shape[0]] = np.copy(self.filtered_angles[sample_indices])  # Populate downsample angles
            self.downsampled_ranges[:sample_indices.shape[0]] = np.copy(self.filtered_ranges[sample_indices])  # Populate downsample measurements

        # Compute the observation
        # obs is a a two element tuple
        # obs[0] is the downsampled ranges
        # obs[1] is the downsampled angles
        # Each element of obs must be a numpy array of type np.float32
        # Use self.LASER_RAY_STEP as the downsampling step
        # Keep efficiency in mind, including by caching certain things that won't change across future iterations of this callback
        obs = (np.copy(self.downsampled_ranges).astype(np.float32), self.downsampled_angles.astype(np.float32))

        self.apply_sensor_model(self.particles, obs, self.weights)
        self.confidence = np.mean(self.weights)
        self.weights /= np.sum(self.weights)

        self.last_laser = msg
        self.do_resample = True
        self.state_lock.release()

    '''
    STUDENTS IMPLEMENT
    Compute table enumerating the probability of observing a measurement
    given the expected measurement
    Element (r,d) of the table is the probability of observing measurement r
    when the expected measurement is d
        max_range_px: The maximum range in pixels
        returns: the table (which is a numpy array with dimensions [max_range_px+1, max_range_px+1])
    '''
    def precompute_sensor_model(self, max_range_px):

        table_width = int(max_range_px) + 1
        sensor_model_table = np.zeros((table_width, table_width))

        # YOUR CODE HERE
        # Don't forget to normalize the weights!

        SIGMA_HIT_SQUARE = SIGMA_HIT**2
        gau_norm = 1.0 / np.sqrt((2 * np.pi * SIGMA_HIT_SQUARE))

        def pdf(z, z_star):
            p_hit = gau_norm * np.exp(-0.5 * np.square(z - z_star) / SIGMA_HIT_SQUARE)
            p_short = (z < z_star) * 2.0 * (z_star-z) / np.square(z_star + 1e-8)
            p_max = (z == max_range_px)
            p_rand = 1.0 / max_range_px

            return Z_HIT*p_hit + Z_SHORT*p_short + Z_MAX*p_max + Z_RAND*p_rand

        sensor_model_table = np.fromfunction(pdf, (table_width, table_width), dtype=np.float32)
        sensor_model_table /= sensor_model_table.sum(axis=0)

        return sensor_model_table

    '''
    Updates the particle weights in-place based on the observed laser scan
        proposal_dist: The particles
        obs: The most recent observation
        weights: The weights of each particle
    '''
    def apply_sensor_model(self, proposal_dist, obs, weights):

        obs_ranges = obs[0]
        obs_angles = obs[1]
        num_rays = obs_angles.shape[0]

        # Only allocate buffers once to avoid slowness
        if not isinstance(self.queries, np.ndarray):
            self.queries = np.zeros((proposal_dist.shape[0], 3), dtype=np.float32)
            self.ranges = np.zeros(num_rays * proposal_dist.shape[0], dtype=np.float32)

        self.queries[:, :] = proposal_dist[:, :]

        self.queries[:, 0] += (self.CAR_LENGTH / 2) * np.cos(self.queries[:, 2])
        self.queries[:, 1] += (self.CAR_LENGTH / 2) * np.sin(self.queries[:, 2])

        # Raycasting to get expected measurements
        self.range_method.calc_range_repeat_angles(self.queries, obs_angles, self.ranges)

        # Evaluate the sensor model
        self.range_method.eval_sensor_model(obs_ranges, self.ranges, weights, num_rays, proposal_dist.shape[0])

        # Squash weights to prevent too much peakiness
        np.power(weights, INV_SQUASH_FACTOR, weights)
