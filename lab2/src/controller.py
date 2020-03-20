import numpy as np
import threading


class BaseController(object):
    def __init__(self):
        self.path_lock = threading.RLock()
        self.path = np.array([])
        self._ready = False
        self.car_length = 0.33

    def ready(self):
        '''
        ready returns whether controller is ready to begin tracking trajectory.
        output:
            ready_val - whether controller is ready to begin tracking trajectory.
        '''
        return self._ready

    def set_path(self, path):
        '''
        set_path - sets trajectory of controller, implicitly resets internal state
        input:
            path - reference path to track
        output:
            none
        '''
        with self.path_lock:
            self.path = np.array([np.array([path[i].x, path[i].y, path[i].h, path[i].v]) for i in range(len(path))])
            self.reset_state()
            self._ready = True
            self.waypoint_diff = np.average(np.linalg.norm(np.diff(self.path[:, :2], axis=0), axis=1))

    def path_complete(self, pose, error):
        '''
        path_complete computes whether the vehicle has completed the path
            based on whether the reference index refers to the final point
            in the path and whether e_x is below the finish_threshold
            or e_y exceeds an 'exceed threshold'.
        input:
            pose - current pose of the vehicle [x, y, heading]
            error - error vector [e_x, e_y]
        output:
            is_path_complete - boolean stating whether the vehicle has
                reached the end of the path
        '''
        err_l2 = np.linalg.norm(error)
        return (self.get_reference_index(pose) == (len(self.path) - 1) and err_l2 < self.finish_threshold) or (err_l2 > self.exceed_threshold)

    def get_reference_pose(self, index):
        '''
        Utility function returns the reference pose from the reference path given a
            reference index.
        '''
        with self.path_lock:
            assert len(self.path) > index
            return self.path[index]

    def get_error(self, pose, index):
        '''
        Computes the error vector for a given pose and reference index.
        input:
            pose - pose of the car [x, y, heading]
            index - integer corresponding to the reference index into the
                reference path
        output:
            e_p - error vector [e_x, e_y]
        '''
        # TODO: INSERT CODE HERE
        # Use the method described in the handout to
        # compute the error vector. Be careful about the order
        # in which you subtract the car's pose from the
        # reference pose.
        pose_ref = self.get_reference_pose(index)
        theta = pose_ref[2]
        cos, sin = np.cos(theta), np.sin(theta)
        rot = np.array([[cos, sin], [-sin, cos]])
        return rot.dot(pose[:2] - pose_ref[:2])
