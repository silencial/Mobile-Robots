#!/usr/bin/env python

import rospy
import threading
import signal

from std_msgs.msg import String, Float64
from std_srvs.srv import Empty


class Publisher:
    def __init__(self):
        rospy.init_node('publisher', anonymous=True, disable_signals=True)

        self.p_init = rospy.Publisher(rospy.get_param('~t_init'), String, queue_size=1)
        self.p_velocity = rospy.Publisher(rospy.get_param('~t_velocity'), Float64, queue_size=1)
        self.p_heading = rospy.Publisher(rospy.get_param('~t_heading'), Float64, queue_size=1)

        rospy.Service(rospy.get_param('~s_reset'), Empty, self.reset)

        self.action_cond = threading.Condition()

        self._load_plan()
        signal.signal(signal.SIGINT, self.sigint)

    def sigint(self, signum, frame):
        rospy.signal_shutdown("shutting down")
        self.action_cond.set()
        exit(1)

    def _load_plan(self):
        self.action_cond.acquire()
        plan_file = rospy.get_param("~plan_file")
        self.actions = []
        with open(plan_file) as f:
            self.actions.append(self.action_init_pose(f.readline()))
            for l in f:
                s = l.split(',')
                if len(s) == 1:  # this is a time
                    self.actions.append(self.action_sleep(int(s[0])))
                elif s[0] == 'v':
                    self.actions.append(self.action_velocity(s[1]))
                elif s[0] == 'd':
                    self.actions.append(self.action_heading(s[1]))
                else:
                    rospy.logerr("Unknown command: " + l)
        self.action_cond.notify()
        self.action_cond.release()

    def action_sleep(self, t):
        def sleep():
            rospy.loginfo("Sleeping for " + str(t) + " ms")
            d = rospy.Duration(nsecs=t * 10e5)
            print d
            rospy.sleep(d)

        return sleep

    def action_init_pose(self, init):
        def init_pose():
            rospy.loginfo("Sending init pose: " + init)
            self.p_init.publish(String(init))

        return init_pose

    def action_velocity(self, v):
        def pub_vel():
            rospy.loginfo("Sending velocity: " + str(v))
            self.p_velocity.publish(Float64(float(v)))

        return pub_vel

    def action_heading(self, h):
        def pub_heading():
            rospy.loginfo("Sending heading: " + str(h))
            self.p_heading.publish(Float64(float(h)))

        return pub_heading

    def reset(self, empty):
        self._load_plan()
        return []

    def start(self):
        while not rospy.is_shutdown():
            with self.action_cond:
                while len(self.actions) == 0:
                    self.action_cond.wait()
                    if rospy.is_shutdown():
                        return
                action = self.actions.pop(0)
                action()

    def stop(self):
        with self.action_cond:
            self.action_cond.notify()


if __name__ == '__main__':
    p = Publisher()

    def sigint(signum, frame):
        rospy.signal_shutdown("sigint")
        p.stop()
        exit(1)

    signal.signal(signal.SIGINT, sigint)

    # Let node topics warmup
    rospy.sleep(1.0)
    t = threading.Thread(target=p.start)
    t.start()

    while not rospy.is_shutdown():
        signal.pause()

    t.join()
