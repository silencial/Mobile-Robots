import numpy as np
from matplotlib import pyplot as plt


class MapEnvironment(object):
    def __init__(self, map_data, stepsize=0.5):
        """
        @param map_data: 2D numpy array of map
        @param stepsize: size of a step to generate waypoints
        """
        # Obtain the boundary limits.
        # Check if file exists.
        self.map = map_data
        self.xlimit = [0, self.map.shape[0]]
        self.ylimit = [0, self.map.shape[1]]
        self.limit = np.array([self.xlimit, self.ylimit])
        self.maxdist = np.float('inf')
        self.stepsize = stepsize

        # Display the map.
        plt.imshow(self.map, interpolation='nearest', origin='lower')
        plt.savefig('map.png')
        print("Saved map as map.png")

    def state_validity_checker(self, configs):
        """
        @param configs: 2D array of [num_configs, dimension].
        Each row contains a configuration
        @return numpy list of boolean values. True for valid states.
        """
        if len(configs.shape) == 1:
            configs = configs.reshape(1, -1)

        # TODO
        # 1. Check for state bounds within xlimit and ylimit
        validity = (configs[:, 0] >= self.xlimit[0]) & (configs[:, 0] < self.xlimit[1]) & (configs[:, 1] >= self.ylimit[0]) & (
            configs[:, 1] < self.ylimit[1])
        # 2. Check collision
        configs_int = np.round(configs).astype(int)
        x = np.clip(configs_int[:, 0], self.xlimit[0], self.xlimit[1] - 1)
        y = np.clip(configs_int[:, 1], self.ylimit[0], self.ylimit[1] - 1)
        validity = validity & np.logical_not(self.map[x, y])

        return validity

    def edge_validity_checker(self, config1, config2):
        """
        Checks whether the path between config 1 and config 2 is valid
        """
        path, length = self.generate_path(config1, config2)
        if length == 0:
            return False, 0

        valid = self.state_validity_checker(path)

        if not np.all(valid):
            return False, self.maxdist
        return True, length

    def compute_heuristic(self, config, goal):
        """
        Returns a heuristic distance between config and goal
        @return a float value
        """
        # TODO
        alpha = 20.0
        heuristic = np.linalg.norm(np.array(config) - np.array(goal))

        return heuristic * alpha

    def compute_distances(self, start_config, end_configs):
        """
        Compute distance from start_config and end_configs in L2 metric.
        @param start_config: tuple of start config
        @param end_configs: list of tuples of end confings
        @return 1D  numpy array of distances
        """

        distances = np.linalg.norm(np.array(end_configs) - np.array(start_config), axis=1)

        return distances

    def generate_path(self, config1, config2):
        config1 = np.array(config1)
        config2 = np.array(config2)
        dist = np.linalg.norm(config2 - config1)
        if dist == 0:
            return config1, dist
        direction = (config2-config1) / dist
        steps = dist // self.stepsize + 1

        waypoints = np.array([np.linspace(config1[i], config2[i], steps) for i in range(2)]).transpose()

        return waypoints, dist

    def get_path_on_graph(self, G, path_nodes):
        plan = []
        for node in path_nodes:
            plan += [G.nodes[node]["config"]]
        plan = np.array(plan)

        path = []
        xs, ys, yaws = [], [], []
        for i in range(np.shape(plan)[0] - 1):
            path += [self.generate_path(plan[i], plan[i + 1])[0]]

        return np.concatenate(path, axis=0)

    def shortcut(self, G, waypoints, num_trials=100):
        """
        Short cut waypoints if collision free
        @param waypoints list of node indices in the graph
        """
        print("Originally {} waypoints".format(len(waypoints)))
        for _ in range(num_trials):

            if len(waypoints) == 2:
                break
            # TODO
            # 1. Choose two configurations
            # 2. Check for collision
            # 3. Connect them if collision free
            i = np.random.randint(0, len(waypoints) - 2)
            j = np.random.randint(i+2, len(waypoints))
            n1, n2 = waypoints[i], waypoints[j]
            if self.edge_validity_checker(G.nodes[n1]['config'], G.nodes[n2]['config'])[0]:
                del waypoints[i+1:j]

        print("Path shortcut to {} waypoints".format(len(waypoints)))
        return waypoints

    def visualize_plan(self, G, path_nodes, saveto=""):
        '''
        Visualize the final path
        @param plan Sequence of states defining the plan.
        '''
        plan = []
        for node in path_nodes:
            plan += [G.nodes[node]["config"]]
        plan = np.array(plan)

        plt.clf()
        plt.imshow(self.map, interpolation='none', cmap='gray', origin='lower')

        # Comment this to hide all edges. This can take long.
        # edges = G.edges()
        # for edge in edges:
        #     config1 = G.nodes[edge[0]]["config"]
        #     config2 = G.nodes[edge[1]]["config"]
        #     x = [config1[0], config2[0]]
        #     y = [config1[1], config2[1]]
        #     plt.plot(y, x, 'grey')

        path = self.get_path_on_graph(G, path_nodes)
        plt.plot(path[:, 1], path[:, 0], 'y', linewidth=1)

        for vertex in G.nodes:
            config = G.nodes[vertex]["config"]
            plt.scatter(config[1], config[0], s=10, c='r')

        plt.tight_layout()

        if saveto != "":
            plt.savefig(saveto)
            return
        plt.show()

    def visualize_graph(self, G, saveto=""):
        plt.clf()
        plt.imshow(self.map, interpolation='nearest', origin='lower')
        edges = G.edges()
        for edge in edges:
            config1 = G.nodes[edge[0]]["config"]
            config2 = G.nodes[edge[1]]["config"]
            path = self.generate_path(config1, config2)[0]
            plt.plot(path[:, 1], path[:, 0], 'w')

        num_nodes = G.number_of_nodes()
        for i, vertex in enumerate(G.nodes):
            config = G.nodes[vertex]["config"]

            if i == num_nodes - 2:
                # Color the start node with blue
                plt.scatter(config[1], config[0], s=30, c='b')
            elif i == num_nodes - 1:
                # Color the goal node with green
                plt.scatter(config[1], config[0], s=30, c='g')
            else:
                plt.scatter(config[1], config[0], s=30, c='r')

        plt.tight_layout()

        if saveto != "":
            plt.savefig(saveto)
            return
        plt.ylabel('x')
        plt.xlabel('y')
        plt.show()
