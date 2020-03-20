#!/usr/bin/env python

from nav_msgs.srv import GetPlan
from nav_msgs.srv import GetMap

from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped, PointStamped
from nav_msgs.msg import Odometry

from lab2.msg import XYHV, XYHVPath
from lab2.srv import FollowPath

import numpy as np
from scipy import signal
import rospy
import os
import time
from matplotlib import pyplot as plt
import networkx as nx

from DubinsMapEnvironment import DubinsMapEnvironment
from MapEnvironment import MapEnvironment
from DubinsSampler import DubinsSampler
import graph_maker
import util
import lazy_astar


class ROSPlanner:
    def __init__(self,
                 heuristic_func,
                 weight_func,
                 num_vertices,
                 connection_radius,
                 graph_file='ros_graph.pkl',
                 do_shortcut=False,
                 num_goals=1,
                 curvature=0.04):
        """
        @param heuristic_func: Heuristic function to be used in lazy_astar
        @param weight_func: Weight function to be used in lazy_astar
        @param num_vertices: Number of vertices in the graph
        @param connection_radius: Radius for connecting vertices
        @param graph_file: File to load. If provided and the file does not exist,
        then the graph is constructed and saved in this filename
        @param do_shortcut: If True, shortcut the path
        @param num_goals: If > 1, takes multiple goals
        """

        rospy.init_node('planner', anonymous=True)

        # load map
        self.map = self.get_map()
        self.map_x = self.map.info.origin.position.x
        self.map_y = self.map.info.origin.position.y
        self.map_angle = util.rosquaternion_to_angle(self.map.info.origin.orientation)
        self.map_c = np.cos(self.map_angle)
        self.map_s = np.sin(self.map_angle)
        self.map_data = self.load_permissible_region(self.map)

        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.get_goal)
        rospy.Subscriber(rospy.get_param("~pose_topic", "/sim_car_pose/pose"), PoseStamped, self.get_current_pose)

        self.multi_goals = num_goals > 1
        self.num_goals = num_goals
        if self.multi_goals:
            print("Accept multiple goals")
            self.goals = []
            self.route_sent = False

        self.do_shortcut = do_shortcut

        # Setup planning env
        self.planning_env = DubinsMapEnvironment(self.map_data.transpose(), curvature=curvature)
        if os.path.exists(graph_file):
            print("Opening {}".format(graph_file))
            self.G = graph_maker.load_graph(graph_file)
        else:
            print("Generating graph, wait for completion")
            self.G = graph_maker.make_graph(self.planning_env,
                                            sampler=DubinsSampler(self.planning_env),
                                            num_vertices=num_vertices,
                                            connection_radius=connection_radius,
                                            saveto=graph_file,
                                            lazy=True,
                                            directed=False)
            print("visualize graph")
            self.planning_env.visualize_graph(self.G, saveto="graph.png")

        self.num_vertices = num_vertices
        self.connection_radius = connection_radius

        self.heuristic_func = lambda n1, n2: heuristic_func(n1, n2, self.planning_env, self.G)
        self.weight_func = lambda n1, n2: weight_func(n1, n2, self.planning_env, self.G)

        print("Ready to take goals")
        rospy.spin()

    def plan_to_goal(self, start, goal):
        """
        Plan a path from start to goal
        Return a path
        """
        # TODO
        # Plan with lazy_astar
        self.G, start_id = graph_maker.add_node(self.G, start, env=self.planning_env, connection_radius=self.connection_radius, start_from_config=True)
        self.G, goal_id = graph_maker.add_node(self.G, goal, env=self.planning_env, connection_radius=self.connection_radius, start_from_config=False)
        path_nodes = lazy_astar.astar_path(self.G, source=start_id, target=goal_id, weight=self.weight_func, heuristic=self.heuristic_func)
        if self.do_shortcut:
            self.planning_env.shortcut(self.G, path_nodes)
        path = self.planning_env.get_path_on_graph(self.G, path_nodes)
        path = self.map2world(path)
        return path

    def plan_multi_goals(self, start):
        """
        Plan a route from start to self.goals
        Return a path connecting them.
        """
        # TODO
        # Plan with lazy_astar
        path = []
        for goal in self.goals:
            path.append(self.plan_to_goal(start, goal))
            start = goal
        return np.concatenate(path, axis=0)

    def get_goal(self, msg):
        print("Got a new goal\n{}".format(msg.pose))
        goal_pose = util.rospose_to_posetup(msg.pose)
        self.goal = self.world2map(np.array([goal_pose]))[0]

        start_pose = util.rospose_to_posetup(self.current_pose)
        self.start = self.world2map(np.array([start_pose]))[0]

        world_points = None

        if self.multi_goals:
            if self.route_sent:
                self.route_sent = False
                self.goals = []
            self.goals += [self.goal.copy()]
            if len(self.goals) == self.num_goals:
                print("Got the final goal for mutli goal. Planning for multiple goals")
                world_points = self.plan_multi_goals(self.start)
            else:
                print("Got {}/{} goals.".format(len(self.goals), self.num_goals))
        else:
            world_points = self.plan_to_goal(self.start, self.goal)
        if world_points is not None and len(world_points) > 0:
            success = self.send_path(world_points)
            print("Sent path")

    def get_current_pose(self, msg):
        self.current_pose = msg.pose

    def send_path(self, waypoints):
        h = Header()
        h.stamp = rospy.Time.now()
        desired_speed = 2.0

        speeds = np.zeros(len(waypoints))
        speeds[:] = desired_speed
        speeds[-1] = 0.0
        path = XYHVPath(h, [XYHV(*[waypoint[0], waypoint[1], waypoint[2], speed])
                        for waypoint, speed in zip(waypoints, speeds)])

        # print("path")
        # print(path)
        print("Sending path...")
        controller = rospy.ServiceProxy("/controller/follow_path", FollowPath())
        success = controller(path)
        print("Controller started.", success)
        return success

    def get_map(self):
        '''
        get_map is a utility function which fetches a map from the map_server
        output:
            map_msg - a GetMap message returned by the mapserver
        '''
        srv_name = "/static_map"  # self.params.get_str("static_map", default="/static_map")
        #self.logger.debug("Waiting for map service")
        print("Waiting for map service")
        rospy.wait_for_service(srv_name)
        print("Map service started")
        #self.logger.debug("Map service started")

        map_msg = rospy.ServiceProxy(srv_name, GetMap)().map
        return map_msg

    def load_permissible_region(self, map):
        # set up caching
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
        pr = np.zeros_like(array_255, dtype=np.uint8)

        # Numpy array of dimension (map_msg.info.height, map_msg.info.width),
        # With values 0: not permissible, 1: permissible
        pr[array_255 == 0] = 1
        pr = np.logical_not(pr)  # 0 is permissible, 1 is not

        return pr

    def world2map(self, poses):
        '''
        world2map is a utility function which converts poses from global
            'world' coordinates (ROS world frame coordinates) to 'map'
            coordinates, that is pixel frame.
        input:
            poses - set of X input poses
            out - output buffer to load converted poses into
        '''
        out = poses.copy()
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
        return out

    def map2world(self, poses):
        scale = self.map.info.resolution
        angle = -self.map_angle

        # Rotation
        c, s = np.cos(angle), np.sin(angle)

        # Store the x coordinates since they will be overwritten
        temp = np.copy(poses[:, 0])
        poses[:, 0] = c * poses[:, 0] - s * poses[:, 1]
        poses[:, 1] = s*temp + c * poses[:, 1]

        # Scale
        poses[:, :2] *= float(scale)

        # Translate
        poses[:, 0] += self.map_x
        poses[:, 1] += self.map_y
        poses[:, 2] += angle

        return poses

if __name__ == '__main__':

    heuristic = lambda n1, n2, env, G: env.compute_heuristic(G.nodes[n1]['config'], G.nodes[n2]['config'])
    weight = lambda n1, n2, env, G: env.edge_validity_checker(G.nodes[n1]['config'], G.nodes[n2]['config'])

    # ROSPlanner(heuristic, weight, num_vertices=300, connection_radius=500)
    ROSPlanner(heuristic, weight, num_vertices=250, connection_radius=200, do_shortcut=True, num_goals=2)
