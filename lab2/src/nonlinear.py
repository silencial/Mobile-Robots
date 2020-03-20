import numpy as np
import rospy

from controller import BaseController


# Uses Proportional-Differential Control from
# https://www.a1k0n.net/2018/11/13/fast-line-following.html
class NonLinearController(BaseController):
    def __init__(self):
        super(NonLinearController, self).__init__()
        self.reset_params()
        self.reset_state()

    def get_reference_index(self, pose):
        with self.path_lock:
            dis = np.linalg.norm(self.path[:, :2] - pose[:2], axis=1)
            ind = np.argmin(dis)
            pose_min = self.path[ind, :2]
            for i in range(ind, self.path.shape[0]):
                if np.linalg.norm(self.path[i, :2] - pose_min) > self.waypoint_lookahead:
                    return i
            return i

    def get_control(self, pose, index):
        e_ct = self.get_error(pose, index)[1]
        pose_ref = self.get_reference_pose(index)
        v = pose_ref[3]
        e_theta = pose[2] - pose_ref[2]
        steering_angle = np.arctan(-self.k1 * e_ct * self.car_length * np.sin(e_theta) / e_theta - self.k2 * self.car_length * e_ct / v)
        return [v, steering_angle]

    def reset_state(self):
        with self.path_lock:
            pass

    def reset_params(self):
        with self.path_lock:
            self.k1 = float(rospy.get_param("/nonlinear/k1", 0.6))  # 0.6
            self.k2 = float(rospy.get_param("/nonlinear/k2", 0.5))  # 0.5
            print(self.k1, self.k2)
            self.finish_threshold = float(rospy.get_param("/nonlinear/finish_threshold", 0.2))
            self.exceed_threshold = float(rospy.get_param("/nonlinear/exceed_threshold", 4.0))
            # Average distance from the current reference pose to lookahed.
            self.waypoint_lookahead = float(rospy.get_param("/nonlinear/waypoint_lookahead", 0.6))  # 0.6
            self.car_length = float(rospy.get_param("/nonlinear/car_length", 0.33))
