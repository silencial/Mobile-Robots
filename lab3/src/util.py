#!/usr/bin/env python

#Patrick Lancaster
#DO NOT EDIT

import rospy
import numpy as np

from std_msgs.msg import Header
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Pose, PoseStamped, PoseArray, Quaternion, PolygonStamped, Polygon, Point32, PoseWithCovarianceStamped, PointStamped
import tf.transformations
import tf
import matplotlib.pyplot as plt


def angle_to_quaternion(angle):
    '''
    Convert yaw angle in radians into a quaternion message
    angle: The yaw angle
    Returns: An equivalent geometry_msgs/Quaternion message
    '''
    return Quaternion(*tf.transformations.quaternion_from_euler(0, 0, angle))


def quaternion_to_angle(q):
    '''
    Convert a quaternion message into a yaw angle in radians.
      q: A geometry_msgs/Quaternion message
      Returns: The equivalent yaw angle
    '''
    x, y, z, w = q.x, q.y, q.z, q.w
    roll, pitch, yaw = tf.transformations.euler_from_quaternion((x, y, z, w))
    return yaw


def rotation_matrix(theta):
    '''
    Constructs a rotation matrix from a given angle in radians
      theta: The angle in radians
      Returns: The equivalent 2x2 numpy rotation matrix
    '''
    c, s = np.cos(theta), np.sin(theta)
    return np.matrix([[c, -s], [s, c]])


def particle_to_pose(particle):
    '''
    Converts a particle to a pose message
      particle: The particle to convert - [x,y,theta]
      Returns: An equivalent geometry_msgs/Pose
    '''
    pose = Pose()
    pose.position.x = particle[0]
    pose.position.y = particle[1]
    pose.orientation = angle_to_quaternion(particle[2])
    return pose


def particles_to_poses(particles):
    '''
    Converts a list of particles to a list of pose messages
      particles: A list of particles, where each element is itself a list of the form [x,y,theta]
      Returns: A list of equivalent geometry_msgs/Pose messages
    '''
    return map(particle_to_pose, particles)


def make_header(frame_id, stamp=None):
    '''
    Creates a header with the given frame_id and stamp. Default value of stamp is
    None, which results in a stamp denoting the time at which this function was called
      frame_id: The desired coordinate frame
      stamp: The desired stamp
      Returns: The resulting header
    '''
    if stamp == None:
        stamp = rospy.Time.now()
    header = Header()
    header.stamp = stamp
    header.frame_id = frame_id
    return header


def point(npt):
    '''
    Converts a list with coordinates into a point message
    npt: A list of length two containing x and y coordinates
    Returns: A geometry_msgs/Point32 message
    '''
    pt = Point32()
    pt.x = npt[0]
    pt.y = npt[1]
    return pt


def points(arr):
    '''
    Converts a list of coordinates into a list of equivalent point messages
    arr: A list of coordinates, where each element is itself a two dimensional list
    Returns: A list of geometry_msgs/Point32 messages
    '''
    return map(point, arr)


def get_map(map_topic):
    ''' Get the map from the map server
  In:
    map_topic: The service topic that will provide the map
  Out:
    map_img: A numpy array with dimensions (map_info.height, map_info.width).
            A zero at a particular location indicates that the location is impermissible
            A one at a particular location indicates that the location is permissible
    map_info: Info about the map, see
              http://docs.ros.org/kinetic/api/nav_msgs/html/msg/MapMetaData.html
              for more info
    '''
    rospy.wait_for_service(map_topic)
    map_msg = rospy.ServiceProxy(map_topic, GetMap)().map
    array_255 = np.array(map_msg.data).reshape((map_msg.info.height, map_msg.info.width))
    map_img = np.zeros_like(array_255, dtype=bool)
    map_img[array_255 == 0] = 1

    return map_img, map_msg.info


def map_to_world(poses, map_info):
    '''
  Convert an array of pixel locations in the map to poses in the world. Does computations
  in-place
    poses: Pixel poses in the map. Should be a nx3 numpy array
    map_info: Info about the map (returned by get_map)
    '''
    scale = map_info.resolution
    angle = quaternion_to_angle(map_info.origin.orientation)

    # Rotation
    c, s = np.cos(angle), np.sin(angle)

    # Store the x coordinates since they will be overwritten
    temp = np.copy(poses[:, 0])
    poses[:, 0] = c * poses[:, 0] - s * poses[:, 1]
    poses[:, 1] = s*temp + c * poses[:, 1]

    # Scale
    poses[:, :2] *= float(scale)

    # Translate
    poses[:, 0] += map_info.origin.position.x
    poses[:, 1] += map_info.origin.position.y
    poses[:, 2] += angle

    return poses


def world_to_map(poses, map_info):
    '''
  Convert array of poses in the world to pixel locations in the map image
    pose: The poses in the world to be converted. Should be a nx3 numpy array
    map_info: Info about the map (returned by get_map)
    '''
    scale = map_info.resolution
    angle = -quaternion_to_angle(map_info.origin.orientation)

    # Translation
    poses[:, 0] -= map_info.origin.position.x
    poses[:, 1] -= map_info.origin.position.y

    # Scale
    poses[:, :2] *= (1.0 / float(scale))

    # Rotation
    c, s = np.cos(angle), np.sin(angle)

    # Store the x coordinates since they will be overwritten
    temp = np.copy(poses[:, 0])
    poses[:, 0] = c * poses[:, 0] - s * poses[:, 1]
    poses[:, 1] = s*temp + c * poses[:, 1]
    poses[:, 2] += angle


def angle_to_rosquaternion(angle):
    """Convert an angle in radians into a quaternion _message_."""
    return Quaternion(*tf.transformations.quaternion_from_euler(0, 0, angle))


def rosquaternion_to_angle(q):
    """Convert a quaternion _message_ into an angle in radians.
    The angle represents the yaw.
    This is not just the z component of the quaternion."""
    x, y, z, w = q.x, q.y, q.z, q.w
    _, _, yaw = tf.transformations.euler_from_quaternion((x, y, z, w))
    return yaw


def rospose_to_posetup(posemsg):
    x = posemsg.position.x
    y = posemsg.position.y
    th = rosquaternion_to_angle(posemsg.orientation)
    return x, y, th
