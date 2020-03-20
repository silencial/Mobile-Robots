#!/usr/bin/env python

import rospy
import rosbag
import utils

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from ackermann_msgs.msg import AckermannDriveStamped

heading = 0.0
circle = 0


def cb_heading(data):
    global heading
    heading = utils.quaternion_to_angle(data.pose.orientation)


rospy.init_node('fig8', anonymous=True, disable_signals=True)

p_initpose = rospy.Publisher("/initialpose", PoseWithCovarianceStamped, queue_size=1)
p_nav0 = rospy.Publisher('/vesc/high_level/ackermann_cmd_mux/input/nav_0', AckermannDriveStamped, queue_size=1)
rospy.Subscriber('/sim_car_pose/pose', PoseStamped, cb_heading)

# Publish initial position
init_pose = PoseWithCovarianceStamped()
init_pose.header.stamp = rospy.Time.now()
init_pose.header.frame_id = 'map'
init_pose.pose.pose.position.x = 0.0
init_pose.pose.pose.position.y = 0.0
init_pose.pose.pose.orientation = utils.angle_to_quaternion(0.0)
rospy.sleep(1)  # publisher needs some time before it can publish
rospy.loginfo('Publish initialpose')
p_initpose.publish(init_pose)
rospy.sleep(1)

# Publish drive info
cmd = AckermannDriveStamped()
cmd.header.frame_id = 'map'
cmd.drive.steering_angle = 0.1
cmd.drive.speed = 2
r = rospy.Rate(20)
while not rospy.is_shutdown():
    if heading < -0.5 and circle == 0:
        circle = 1
    elif heading > -0.1 and circle == 1:
        cmd.drive.steering_angle = -cmd.drive.steering_angle
        circle = 2
    elif heading > 0.5 and circle == 2:
        circle = 3
    elif heading < 0.1 and circle == 3:
        break
    p_nav0.publish(cmd)
    r.sleep()
