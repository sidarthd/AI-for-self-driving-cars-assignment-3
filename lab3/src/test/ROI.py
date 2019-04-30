#!/usr/bin/env python
import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from PID import PID


"""
 Publish a filtered image from the given image data.
 Currently, it filters out all except light blue colors.
 sub_topic: where the original image is from
 pub_topic: where to publish the new image to
"""
class ROI:
    def __init__(self, sub_topic, pub_topic):
        print("Subscribing to", sub_topic)
        print("Publishing to", pub_topic)

        self.pub = rospy.Publisher(pub_topic, Image, queue_size=10)
        self.sub = rospy.Subscriber(sub_topic, Image, self.apply_filter_cb)
        self.bridge = CvBridge()

        self.error = 0.0
        self.PID = PID()
        self.tape_seen = False
        self.tape_at_bottom = False


    """
    A callback to remove all except some range of colors.
    Assuming that msg is sensor_msgs/Image
    """
    def apply_filter_cb(self, msg):
        cv_image = None
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
        except CvBridgeError as e:
            print(e)

        cv_image = cv2.GaussianBlur(cv_image, (3,3), 0)

        hsv  = cv2.cvtColor(cv_image, cv2.COLOR_RGB2HSV)

        # To see the light blue tape
        # was 70-150 hue
        # lower = np.array([70, 100, 100]) #100
        # upper = np.array([150, 255, 255])
        # HUE is between 0 and 180
        # S, V are from 0 to 255
        lower = np.array([70, 40, 150])
        upper = np.array([150, 255, 255])
        mask_b = cv2.inRange(hsv, lower, upper)

        # To see the red tape
        # lower_l = np.array([0, 100, 100])
        # lower_h = np.array([20, 255, 255])
        # higher_l = np.array([170, 100, 100])
        # higher_r = np.array([179, 255, 255])
        # mask_rl = cv2.inRange(hsv, lower_l, lower_h)
        # mask_r = cv2.inRange(hsv, higher_l, higher_r)
        # mask_r = cv2.addWeighted(mask_rl, 1.0, mask_rh, 1.0)

        mask = mask_b

        res = cv2.bitwise_and(cv_image, cv_image, mask = mask)
        #print("shape is ", res.shape)
        #print("Publishing:")
        try:
            # res[:,:,0] = 0
            # res[:,:,1] = 0
            # print("shape is ", res.shape)
            self.drawROI(res)

            self.pub.publish(self.bridge.cv2_to_imgmsg(res, 'rgb8'))
        except CvBridgeError as e:
            print(e)
        # print("DONE!")

    def getError(self):
        return self.error


    """
        Draw out thecv2.inRange(hsv, lower, upper) ROI, and get the center
    """
    def drawROI(self, img):
        height = None
        width = None
        channels = 1

        edges = None


        if len(img.shape) == 2:
            height, width = img.shape
            #edges = img.copy()
        else:
            height, width, channels  = img.shape

            #edges = self.getEdges(img)

        # Used to set the lines of the ROI
        bottomLineHeight = height - 10
        topLineHeight = 0 #height - height/6

        # Mask img for ROI
        mask = np.zeros((height, width), dtype = 'uint8')
        mask[(topLineHeight):(bottomLineHeight) ,:] = 1
        ROIimg = cv2.bitwise_and(img ,img , mask = mask)
        set_pixels = ROIimg.sum()

        gray = cv2.cvtColor(ROIimg, cv2.COLOR_RGB2GRAY)
        thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        M = cv2.moments(cnts)
        #print("M is ", M)
        # If the tape is not within the ROI, then don't calculate moment
        # print("Set:", set_pixels)
        if M["m00"] == 0 or set_pixels < 1000000:
            # print("Not on top of tape")
            self.error = 0.0
            self.tape_seen = False
            self.tape_at_bottom = False
        else:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # draw the contour and center of the shape on the image
            cv2.circle(img, (cX, cY), 7, (255, 255, 255), -1)
            self.error = float(width/2.0 - float(cX))
            # print('Cx: ', float(cX))
            # print('Center:', width/2.0)
            self.tape_seen = True
            self.tape_at_bottom = cY > (height * 4 / 5)

        # negative error, turn right... hopefully
        # control = self.PID.calc_control(self.error)
        # self.PID.drive(control)

        cv2.line(img, (0,topLineHeight), (width,topLineHeight), color = (0, 255, 0))
        cv2.line(img, (0,bottomLineHeight), (width, bottomLineHeight), color = (0, 255, 0))


if __name__ == '__main__':
    sub_topic = None # The image topic to be subscribed to
    pub_topic = None # The topic to publish filtered images to
    fast_convolve = False # Whether or not the nested for loop or Scipy's convolve method should be used for applying the filter

    rospy.init_node('apply_filter', anonymous=True)

    # Populate params with values passed by launch file
    sub_topic = rospy.get_param('~sub_topic')
    pub_topic = rospy.get_param('~pub_topic')

    # Create a ROI object and pass it the loaded parameters
    roi = ROI(sub_topic, pub_topic)

    rospy.spin()