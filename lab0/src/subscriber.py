#!/usr/bin/env python

import rospy
import utils

from std_msgs.msg import String, Float64
from std_srvs.srv import Empty
from geometry_msgs.msg import PoseWithCovarianceStamped
from ackermann_msgs.msg import AckermannDriveStamped


class Subscriber:
    def __init__(self):
        rospy.init_node('subscriber', anonymous=True, disable_signals=True)

        self.p_initpose = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=1)
        self.p_nav0 = rospy.Publisher('/vesc/high_level/ackermann_cmd_mux/input/nav_0', AckermannDriveStamped, queue_size=1)

        self.nav0 = AckermannDriveStamped()
        self.nav0.header.frame_id = 'map'

        rospy.Subscriber('/lab0/initpose', String, self._cb_init)
        rospy.Subscriber('/lab0/velocity', Float64, self._cb_velocity)
        rospy.Subscriber('/lab0/heading', Float64, self._cb_heading)

        rospy.Service('/subscriber/reset', Empty, self.reset)

    def _cb_init(self, data):
        rospy.loginfo('Receiving init pose: ' + data.data)
        data = data.data.split(',')
        init_pose = PoseWithCovarianceStamped()
        init_pose.header.stamp = rospy.Time.now()
        init_pose.header.frame_id = 'map'
        init_pose.pose.pose.position.x = float(data[0])
        init_pose.pose.pose.position.y = float(data[1])
        init_pose.pose.pose.orientation = utils.angle_to_quaternion(float(data[2]))
        self.p_initpose.publish(init_pose)

    def _cb_velocity(self, data):
        rospy.loginfo('Receiving velocity: ' + str(data.data))
        self.nav0.drive.speed = data.data

    def _cb_heading(self, data):
        rospy.loginfo('Receiving heading angle: ' + str(data.data))
        self.nav0.drive.steering_angle = data.data

    def pub(self):
        self.nav0.header.stamp = rospy.Time.now()
        self.p_nav0.publish(self.nav0)

    def reset(self, data):
        rospy.loginfo('Reset velocity and heading angle')
        self.nav0.drive.speed = 0
        self.nav0.drive.steering_angle = 0
        return []


if __name__ == '__main__':
    s = Subscriber()
    r = rospy.Rate(20)
    while not rospy.is_shutdown():
        s.pub()
        r.sleep()
