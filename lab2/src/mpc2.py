import numpy as np
import rospy
import utils
from matplotlib import cm
import matplotlib.colors as colors

from controller import BaseController
from nav_msgs.srv import GetMap
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from sensor_msgs.msg import LaserScan


class ModelPredictiveController(BaseController):
    def __init__(self):
        super(ModelPredictiveController, self).__init__()

        self.reset_params()
        self.reset_state()

        self.laser_sub = rospy.Subscriber(rospy.get_param('~scan_topic', default='/scan'), LaserScan, self.lidar_cb, queue_size=1)
        self.rp_costs = rospy.Publisher(rospy.get_param("~costs_viz_topic", default="/controller/path/costs"), MarkerArray, queue_size=50)

        self.init_laser()

    def init_laser(self):
        self.LASER_RAY_STEP = 18  # Step for downsampling laser scans
        self.MAX_RANGE_METERS = 11.0  # The max range of the laser
        self.CAR_LENGTH = 0.33

        self.laser_angles = None  # The angles of each ray
        self.downsampled_angles = None  # The angles of the downsampled rays
        self.downsampled_ranges = None
        self.obs = None

    def lidar_cb(self, msg):
        # Initialize angle array
        if not isinstance(self.laser_angles, np.ndarray):
            self.laser_angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        ranges = np.array(msg.ranges)  # Get the measurements
        ranges[np.isnan(ranges)] = self.MAX_RANGE_METERS  # Remove nans
        valid_indices = np.logical_and(ranges > 0.01, ranges < self.MAX_RANGE_METERS)  # Find non-extreme measurements
        self.filtered_angles = np.copy(self.laser_angles[valid_indices]).astype(np.float32)  # Get angles corresponding to non-extreme measurements
        self.filtered_ranges = np.copy(ranges[valid_indices]).astype(np.float32)  # Get non-extreme measurements

        ray_count = int(self.laser_angles.shape[0] / self.LASER_RAY_STEP)  # Compute expected number of rays
        sample_indices = np.arange(0, self.filtered_angles.shape[0],
                                   float(self.filtered_angles.shape[0]) / ray_count).astype(np.int)  # Get downsample indices
        # self.downsampled_angles = np.zeros(ray_count + 1, dtype=np.float32)  # Initialize down sample angles
        # self.downsampled_ranges = np.zeros(ray_count + 1, dtype=np.float32)  # Initialize down sample measurements
        # self.downsampled_angles[:sample_indices.shape[0]] = np.copy(self.filtered_angles[sample_indices])  # Populate downsample angles
        # self.downsampled_ranges[:sample_indices.shape[0]] = np.copy(self.filtered_ranges[sample_indices])  # Populate downsample measurements
        self.downsampled_angles = np.copy(self.filtered_angles[sample_indices])
        self.downsampled_ranges = np.copy(self.filtered_ranges[sample_indices])

        self.obs = (np.copy(self.downsampled_ranges).astype(np.float32), self.downsampled_angles.astype(np.float32))

    def get_reference_index(self, pose):
        '''
        get_reference_index finds the index i in the controller's path
            to compute the next control control against
        input:
            pose - current pose of the car, represented as [x, y, heading]
        output:
            i - referencence index
        '''
        with self.path_lock:
            # TODO: INSERT CODE HERE
            # Determine a strategy for selecting a reference point
            # in the path. One option is to find the nearest reference
            # point and chose the next point some distance away along
            # the path. You are welcome to use the same method for MPC
            # as you used for PID or Pure Pursuit.
            #
            # Note: this method must be computationally efficient
            # as it is running directly in the tight control loop.
            dis = np.linalg.norm(self.path[:, :2] - pose[:2], axis=1)
            ind = np.argmin(dis)
            pose_min = self.path[ind, :2]
            for i in range(ind, self.path.shape[0]):
                if np.linalg.norm(self.path[i, :2] - pose_min) > self.waypoint_lookahead:
                    return i
            return i

    def get_control(self, pose, index):
        '''
        get_control - computes the control action given an index into the
            reference trajectory, and the current pose of the car.
            Note: the output velocity is given in the reference point.
        input:
            pose - the vehicle's current pose [x, y, heading]
            index - an integer corresponding to the reference index into the
                reference path to control against
        output:
            control - [velocity, steering angle]
        '''
        assert len(pose) == 3

        rollouts = np.zeros((self.K, self.T, 3))

        # TODO: INSERT CODE HERE
        # With MPC, you should roll out K control trajectories using
        # the kinematic car model, then score each one with your cost
        # function, and finally select the one with the lowest cost.

        # For each K trial, the first position is at the current position (pose)

        # For control trajectories, get the speed from reference point.

        # Then step forward the K control trajectory T timesteps using
        # self.apply_kinematics
        pose_ref = self.get_reference_pose(index)
        self.trajs[:, :, 0] = pose_ref[3]
        rollouts[:, 0, :] = pose
        for i in range(1, self.T):
            rollouts[:, i, :] = np.array(self.apply_kinematics(rollouts[:, i - 1, :], self.trajs[:, i - 1, :])).T

        # Apply the cost function to the rolled out poses.
        costs = self.apply_cost(rollouts, index)

        self.visualize_cost(rollouts, costs)

        # Finally, find the control trajectory with the minimum cost.
        min_control = np.argmin(costs)

        # Return the controls which yielded the min cost.
        return self.trajs[min_control][0]

    def reset_state(self):
        '''
        Utility function for resetting internal states.
        '''
        with self.path_lock:
            self.trajs = self.get_control_trajectories()
            assert self.trajs.shape == (self.K, self.T, 2)
            self.scaled = np.zeros((self.K * self.T, 3))
            self.bbox_map = np.zeros((self.K * self.T, 2, 4))
            self.perm = np.zeros(self.K * self.T).astype(np.int)
            self.map = self.get_map()
            self.perm_reg = self.load_permissible_region(self.map)
            self.map_x = self.map.info.origin.position.x
            self.map_y = self.map.info.origin.position.y
            self.map_angle = utils.rosquaternion_to_angle(self.map.info.origin.orientation)
            self.map_c = np.cos(self.map_angle)
            self.map_s = np.sin(self.map_angle)

    def reset_params(self):
        '''
        Utility function for updating parameters which depend on the ros parameter
            server. Setting parameters, such as gains, can be useful for interative
            testing.
        '''
        with self.path_lock:
            self.wheelbase = float(rospy.get_param("trajgen/wheelbase", 0.33))
            self.min_delta = float(rospy.get_param("trajgen/min_delta", -0.34))
            self.max_delta = float(rospy.get_param("trajgen/max_delta", 0.34))

            self.K = int(rospy.get_param("mpc/K", 62))
            self.T = int(rospy.get_param("mpc/T", 8))

            self.speed = float(rospy.get_param("mpc/speed", 1.0))
            self.finish_threshold = float(rospy.get_param("mpc/finish_threshold", 1.0))
            self.exceed_threshold = float(rospy.get_param("mpc/exceed_threshold", 4.0))
            # Average distance from the current reference pose to lookahed.
            self.waypoint_lookahead = float(rospy.get_param("mpc/waypoint_lookahead", 1.0))
            self.collision_w = float(rospy.get_param("mpc/collision_w", 1e5))
            self.error_w = float(rospy.get_param("mpc/error_w", 1.0))
            self.obstacle_w = float(rospy.get_param("mpc/obstacle_w", 2.0))

            self.car_length = float(rospy.get_param("mpc/car_length", 0.33))
            self.car_width = float(rospy.get_param("mpc/car_width", 0.15))

    def get_control_trajectories(self):
        '''
        get_control_trajectories computes K control trajectories to be
            rolled out on each update step. You should only execute this
            function when initializing the state of the controller.

            various methods can be used for generating a sufficient set
            of control trajectories, but we suggest stepping over the
            control space (of steering angles) to create the initial
            set of control trajectories.
        output:
            ctrls - a (K x T x 2) vector which lists K control trajectories
                of length T
        '''
        # TODO: INSERT CODE HERE
        # Create a trajectory library of K trajectories for T timesteps to
        # roll out when computing the best control.
        ctrls = np.zeros((self.K, self.T, 2))
        # step_size = (self.max_delta - self.min_delta) / (self.K - 1)
        deltas = np.linspace(self.min_delta, self.max_delta, self.K)
        ctrls[:, :, 1] = deltas.reshape(-1, 1)
        return ctrls

    def apply_kinematics(self, cur_x, control):
        '''
        apply_kinematics 'steps' forward the pose of the car using
            the kinematic car model for a given set of K controls.
        input:
            cur_x   (K x 3) - current K "poses" of the car
            control (K x 2) - current controls to step forward
        output:
            (x_dot, y_dot, theta_dot) - where each *_dot is a list
                of k deltas computed by the kinematic car model.
        '''
        # TODO: INSERT CODE HERE
        # Use the kinematic car model discussed in class to step
        # forward the pose of the car. We will step all K poses
        # simultaneously, so we recommend using Numpy to compute
        # this operation.
        #
        # ulimately, return a triplet with x_dot, y_dot_, theta_dot
        # where each is a length K numpy vector.
        dt = 0.1

        v, delta = control[:, 0], control[:, 1]

        theta_dot = cur_x[:, 2] + v / self.car_length * np.tan(delta) * dt
        x_dot = cur_x[:, 0] + self.car_length / np.tan(delta) * (np.sin(theta_dot) - np.sin(cur_x[:, 2]))
        y_dot = cur_x[:, 1] + self.car_length / np.tan(delta) * (-np.cos(theta_dot) + np.cos(cur_x[:, 2]))

        # Limit rotation to be between -pi and pi
        theta_dot[theta_dot < -np.pi] += 2 * np.pi
        theta_dot[theta_dot > np.pi] -= 2 * np.pi

        return (x_dot, y_dot, theta_dot)

    def apply_cost(self, poses, index):
        '''
        rollouts (K,T,3) - poses of each rollout
        index    (int)   - reference index in path
        '''
        all_poses = poses.copy()
        all_poses.resize(self.K * self.T, 3)
        # collisions = self.check_collisions_in_map(all_poses)
        # collisions.resize(self.K, self.T)
        # collision_cost = collisions.sum(axis=1) * self.collision_w
        error_cost = np.linalg.norm(poses[:, self.T - 1, :2] - self.path[index, :2], axis=1) * self.error_w
        obstacles_cost = self.check_obstacles(poses) * self.obstacle_w
        # return collision_cost + error_cost + obstacles_cost
        return error_cost + obstacles_cost

    def check_obstacles(self, poses):
        # Calculate obstacles position in world coordinate by layser info
        ranges, angles = self.obs
        pose = poses[0, 0, :]
        theta = pose[2]
        layser_start = self.car_length / 2 * np.array([np.cos(theta), np.sin(theta)]) + pose[:2]
        obstacles = layser_start + (ranges * np.array([np.cos(angles + theta), np.sin(angles + theta)])).T
        # 1. Calculate the distance to obstacles for every trajectory, at each timestep (K * T * num_obstacles)
        # 2. Calculate the min distance (K * T)
        # 3. Calculate the average distance w.r.t. time (K)
        # 4. Use inverse as cost
        cost = 0.0
        pose_bc = np.tile(poses[:, :, :2], (obstacles.shape[0], 1, 1, 1)).transpose((1, 2, 0, 3))
        cost = np.linalg.norm(pose_bc - obstacles, axis=3).min(axis=2).sum(axis=1)
        cost /= poses.shape[1]

        return 1 / (cost + 1e-4)

    def check_collisions_in_map(self, poses):
        '''
        check_collisions_in_map is a collision checker that determines whether a set of K * T poses
            are in collision with occupied pixels on the map.
        input:
            poses (K * T x 3) - poses to check for collisions on
        output:
            collisions - a (K * T x 1) float vector where 1.0 signifies collision and 0.0 signifies
                no collision for the input pose with corresponding index.
        '''

        self.world2map(poses, out=self.scaled)

        L = self.car_length
        W = self.car_width

        # Specify specs of bounding box
        bbox = np.array([[L / 2.0, W / 2.0], [L / 2.0, -W / 2.0], [-L / 2.0, W / 2.0], [-L / 2.0, -W / 2.0]]) / (self.map.info.resolution)

        x = np.tile(bbox[:, 0], (len(poses), 1))
        y = np.tile(bbox[:, 1], (len(poses), 1))

        xs = self.scaled[:, 0]
        ys = self.scaled[:, 1]
        thetas = self.scaled[:, 2]

        c = np.resize(np.cos(thetas), (len(thetas), 1))
        s = np.resize(np.sin(thetas), (len(thetas), 1))

        self.bbox_map[:, 0] = (x*c - y*s) + np.tile(np.resize(xs, (len(xs), 1)), 4)
        self.bbox_map[:, 1] = (x*s + y*c) + np.tile(np.resize(ys, (len(ys), 1)), 4)

        bbox_idx = self.bbox_map.astype(np.int)

        self.perm[:] = 0
        self.perm = np.logical_or(self.perm, self.perm_reg[bbox_idx[:, 1, 0], bbox_idx[:, 0, 0]])
        self.perm = np.logical_or(self.perm, self.perm_reg[bbox_idx[:, 1, 1], bbox_idx[:, 0, 1]])
        self.perm = np.logical_or(self.perm, self.perm_reg[bbox_idx[:, 1, 2], bbox_idx[:, 0, 2]])
        self.perm = np.logical_or(self.perm, self.perm_reg[bbox_idx[:, 1, 3], bbox_idx[:, 0, 3]])

        return self.perm.astype(np.float)

    def get_map(self):
        '''
        get_map is a utility function which fetches a map from the map_server
        output:
            map_msg - a GetMap message returned by the mapserver
        '''
        srv_name = rospy.get_param("static_map", default="/static_map")
        rospy.logdebug("Waiting for map service")
        rospy.wait_for_service(srv_name)
        rospy.logdebug("Map service started")

        map_msg = rospy.ServiceProxy(srv_name, GetMap)().map
        return map_msg

    def load_permissible_region(self, map):
        '''
        load_permissible_region uses map data to compute a 'permissible_region'
            matrix given the map information. In this matrix, 0 is permissible,
            1 is not.
        input:
            map - GetMap message
        output:
            pr - permissible region matrix
        '''
        map_data = np.array(map.data)
        array_255 = map_data.reshape((map.info.height, map.info.width))
        pr = np.zeros_like(array_255, dtype=bool)

        # Numpy array of dimension (map_msg.info.height, map_msg.info.width),
        # With values 0: not permissible, 1: permissible
        pr[array_255 == 0] = 1
        pr = np.logical_not(pr)  # 0 is permissible, 1 is not

        return pr

    def world2map(self, poses, out):
        '''
        world2map is a utility function which converts poses from global
            'world' coordinates (ROS world frame coordinates) to 'map'
            coordinates, that is pixel frame.
        input:
            poses - set of X input poses
            out - output buffer to load converted poses into
        '''
        out[:] = poses
        # translation
        out[:, 0] -= self.map_x
        out[:, 1] -= self.map_y

        # scale
        out[:, :2] *= (1.0 / float(self.map.info.resolution))

        # we need to store the x coordinates since they will be overwritten
        temp = np.copy(out[:, 0])
        out[:, 0] = self.map_c * out[:, 0] - self.map_s * out[:, 1]
        out[:, 1] = self.map_s * temp + self.map_c * out[:, 1]
        out[:, 2] += self.map_angle

    def visualize_cost(self, rollouts, costs):
        markers = MarkerArray()
        norm = colors.Normalize(clip=True)
        norm.autoscale(costs)
        color_map = cm.get_cmap()
        for i, (rollout, cost) in enumerate(zip(rollouts, costs)):
            marker = Marker()
            marker.header.stamp = rospy.Time.now()
            marker.header.frame_id = 'map'
            marker.ns = 'costs'
            marker.id = i
            marker.type = Marker.LINE_STRIP
            # marker.action = Marker.ADD

            marker.pose.position.x = 0
            marker.pose.position.y = 0
            marker.pose.position.z = 0

            marker.pose.orientation.x = 0
            marker.pose.orientation.y = 0
            marker.pose.orientation.z = 0
            marker.pose.orientation.w = 1.0

            marker.scale.x = 0.02

            color = color_map(norm(cost))
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = color[3]
            for pose in rollout:
                p = Point()
                p.x, p.y = pose[0], pose[1]
                marker.points.append(p)

            markers.markers.append(marker)

        self.rp_costs.publish(markers)
