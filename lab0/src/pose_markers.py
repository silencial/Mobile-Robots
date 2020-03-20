#!/usr/bin/env python

import rospy

from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker


class PoseMarker:
    def __init__(self):
        rospy.init_node('pose_markers', anonymous=True, disable_signals=True)

        self.marker = Marker()
        self.marker.header.frame_id = 'map'
        self.marker.ns = "my_namespace"
        self.marker.id = 0
        self.marker.type = Marker.ARROW
        self.marker.action = Marker.ADD
        self.marker.id = 0
        self.marker.scale.x = 1
        self.marker.scale.y = 0.05
        self.marker.scale.z = 0.05
        self.marker.color.a = 1.0
        self.marker.color.r = 0.0
        self.marker.color.g = 0.0
        self.marker.color.b = 1.0

        self.p_markers = rospy.Publisher('/pose_markers/markers', Marker, queue_size=1)
        rospy.Subscriber('/sim_car_pose/pose', PoseStamped, self._cb_pose)

    def _cb_pose(self, data):
        self.marker.pose = data.pose

    def pub(self):
        self.marker.header.stamp = rospy.Time.now()
        self.marker.id += 1

        self.p_markers.publish(self.marker)


if __name__ == '__main__':
    marker = PoseMarker()
    r = rospy.Rate(2)
    while not rospy.is_shutdown():
        marker.pub()
        r.sleep()
