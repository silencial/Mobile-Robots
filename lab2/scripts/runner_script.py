#!/usr/bin/env python
from lab2.msg import XYHV, XYHVPath
from lab2.srv import FollowPath
from std_msgs.msg import Header

import numpy as np
import pickle
from scipy import signal
import rospy


def saw():
    """
    Generates a sawtooth path
    """
    # Change alpha to get a different frequency
    t = np.linspace(0, 20, 100)
    alpha = 0.5
    saw = signal.sawtooth(alpha * np.pi * t)
    configs = [[x, y, 0] for (x, y) in zip(t, saw)]
    return configs


def circle():
    """
    Generates a circular path.
    """
    # Change radius to modify the size of the circle
    waypoint_sep = 0.1
    radius = 2.5
    center = [0, radius]
    num_points = int((2 * radius * np.pi) / waypoint_sep)
    thetas = np.linspace(-1 * np.pi / 2, 2 * np.pi - (np.pi / 2), num_points)
    poses = [[radius * np.cos(theta) + center[0], radius * np.sin(theta) + center[1], theta + (np.pi / 2)] for theta in thetas]
    return poses


def left_turn():
    """
    Generates a path that goes straight and turns left 90 degrees.
    """
    # Change turn_radius and straight_len to modify the path
    waypoint_sep = 0.1
    turn_radius = 1.5
    straight_len = 10.0
    turn_center = [straight_len, turn_radius]
    straight_xs = np.linspace(0, straight_len, int(straight_len / waypoint_sep))
    straight_poses = [[x, 0, 0] for x in straight_xs]
    num_turn_points = int((turn_radius * np.pi * 0.5) / waypoint_sep)
    thetas = np.linspace(-1 * np.pi / 2, 0, num_turn_points)
    turn_poses = [[turn_radius * np.cos(theta) + turn_center[0], turn_radius * np.sin(theta) + turn_center[1], theta + (np.pi / 2)] for theta in thetas]
    poses = straight_poses + turn_poses
    return poses


def right_turn():
    """
    Generates a path that goes straight and turns right 90 degrees.
    """
    # Change turn_radius and straight_len to modify the path
    waypoint_sep = 0.1
    turn_radius = 1.5
    straight_len = 10.0
    turn_center = [straight_len, -turn_radius]
    straight_xs = np.linspace(0, straight_len, int(straight_len / waypoint_sep))
    straight_poses = [[x, 0, 0] for x in straight_xs]
    num_turn_points = int((turn_radius * np.pi * 0.5) / waypoint_sep)
    thetas = np.linspace(1 * np.pi / 2, 0, num_turn_points)
    turn_poses = [[turn_radius * np.cos(theta) + turn_center[0], turn_radius * np.sin(theta) + turn_center[1], theta - (np.pi / 2)] for theta in thetas]
    poses = straight_poses + turn_poses
    return poses


def cse022_path():
    with open('cse022_path.pickle', 'r') as f:
        p = pickle.load(f)
    return p


plans = {'circle': circle, 'left turn': left_turn, 'right turn': right_turn, 'saw': saw, 'cse022 real path': cse022_path}
plan_names = ['circle', 'left turn', 'right turn', 'saw', 'cse022 real path']


def generate_plan():
    print "Which plan would you like to generate? "
    for i, name in enumerate(plan_names):
        print "{} ({})".format(name, i)
    index = int(raw_input("num: "))
    if index >= len(plan_names):
        print "Wrong number. Exiting."
        exit()
    return plans[plan_names[index]]()


if __name__ == '__main__':
    rospy.init_node("controller_runner")
    configs = generate_plan()

    if type(configs) == XYHVPath:
        path = configs
    else:
        h = Header()
        h.stamp = rospy.Time.now()
        desired_speed = 2.0
        ramp_percent = 0.1
        ramp_up = np.linspace(0.0, desired_speed, int(ramp_percent * len(configs)))
        ramp_down = np.linspace(desired_speed, 0.3, int(ramp_percent * len(configs)))
        speeds = np.zeros(len(configs))
        speeds[:] = desired_speed
        speeds[0:len(ramp_up)] = ramp_up
        speeds[-len(ramp_down):] = ramp_down
        path = XYHVPath(h, [XYHV(*[config[0], config[1], config[2], speed]) for config, speed in zip(configs, speeds)])
    print "Sending path..."
    controller = rospy.ServiceProxy("/controller/follow_path", FollowPath())
    success = controller(path)
    print "Controller started."
