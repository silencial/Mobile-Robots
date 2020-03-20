#!/usr/bin/env python
import rospy
from std_msgs.msg import Float64

rospy.init_node("mynode")
pub = rospy.Publisher("command", Float64)
pub.publish(45.0)
rospy.spin()
