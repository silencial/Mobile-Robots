#!/usr/bin/env python

import rospy
import rosbag
import utils

from geometry_msgs.msg import PoseWithCovarianceStamped
from ackermann_msgs.msg import AckermannDriveStamped


rospy.init_node('bag_follower', anonymous=True, disable_signals=True)

p_initpose = rospy.Publisher("/initialpose", PoseWithCovarianceStamped, queue_size=1)
p_nav0 = rospy.Publisher('/vesc/high_level/ackermann_cmd_mux/input/nav_0', AckermannDriveStamped, queue_size=1)

bag_file = rospy.get_param('~bag_file')
rospy.loginfo('Bag file: ' + str(bag_file))

reverse = rospy.get_param('~reverse')
rospy.loginfo('Reverse: ' + str(reverse))

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

# Publish drive info
r = rospy.Rate(20)
bag = rosbag.Bag(bag_file)
for topic, msg, t in bag.read_messages(topics=['/vesc/low_level/ackermann_cmd_mux/input/teleop']):
    if reverse:
        msg.drive.speed = -msg.drive.speed
    p_nav0.publish(msg)
    r.sleep()
bag.close()
