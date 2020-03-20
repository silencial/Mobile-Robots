import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from Dubins import path_length
from Dubins import dubins_path_planning
from MapEnvironment import MapEnvironment


class DubinsMapEnvironment(MapEnvironment):
    def __init__(self, map_data, curvature=5):
        super(DubinsMapEnvironment, self).__init__(map_data)
        self.curvature = curvature

    def compute_distances(self, start_config, end_configs):
        """
        Compute distance from start_config and end_configs using Dubins path
        @param start_config: tuple of start config
        @param end_configs: list of tuples of end confings
        @return numpy array of distances
        """

        # TODO
        distances = np.empty(len(end_configs))
        for i, end_config in enumerate(end_configs):
            distances[i] = path_length(start_config, end_config, self.curvature)
        return distances

    def compute_heuristic(self, config, goal):
        """
        Use the Dubins path length from config to goal as the heuristic distance.
        """

        # TODO
        heuristic = path_length(config, goal, self.curvature)
        return heuristic

    def generate_path(self, config1, config2):
        """
        Generate a dubins path from config1 to config2
        The generated path is not guaranteed to be collision free
        Use dubins_path_planning to get a path
        return: (numpy array of [x, y, yaw], curve length)
        """

        # TODO
        x, y, yaw, path_length = dubins_path_planning(config1, config2, self.curvature)
        path = np.array([x, y, yaw]).transpose()
        return path, path_length
