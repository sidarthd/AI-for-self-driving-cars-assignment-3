#!/usr/bin/env python
import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from ackermann_msgs.msg import AckermannDriveStamped

K_p = 0.07
K_i = -0.0
K_d = 0.03
SPEED = 0.4
STEERING_FACTOR = 0.03
PUB_TOPIC = '/vesc/high_level/ackermann_cmd_mux/input/nav_0'
PUB_RATE = 20

class PID:
    def __init__(self):
        self.sum_errors = 0
        self.previous_error = None
        self.previous_time = rospy.Time.now()
        self.pub = rospy.Publisher(PUB_TOPIC, AckermannDriveStamped, queue_size=2)
        self.r = rospy.Rate(PUB_RATE)

    def calc_control(self, error):
        # print('Error: ', error)
        self.sum_errors += error
        current_time = rospy.Time.now()
        de_dt = 0
        if self.previous_error is not None:
            dt = (current_time - self.previous_time).to_sec()
            de_dt = float(error - self.previous_error) / float(dt)
        self.previous_time = current_time
        control = K_p*error + K_i*self.sum_errors + K_d*de_dt
        self.previous_error = error
        # print('Control: ', control)
        return control

    def drive(self, control):
        msg = AckermannDriveStamped()
        msg.drive.speed = SPEED
        msg.drive.steering_angle = STEERING_FACTOR * control
        print('Steering angle: ',msg.drive.steering_angle)
        self.pub.publish(msg)
        self.r.sleep()