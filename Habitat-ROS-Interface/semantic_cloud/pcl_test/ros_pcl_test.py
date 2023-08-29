#!/usr/bin/env python
from __future__ import print_function
from __future__ import division
import rospy
import sys
import cv2
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge, CvBridgeError
from color_pcl_generator import ColorPclGenerator
import message_filters
import numpy as np
from scipy import misc
from skimage import io
import time
import matplotlib.pyplot as plt

'''
Test generating color point cloud from image stream
'''
class testPclRos:

    def __init__(self):
        fx = 615.6707153320312
        fy = 615.962158203125
        cx = 328.0010681152344
        cy = 241.31031799316406
        self.intrinsic = np.matrix([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype = np.float32)
        self.pcl_pub = rospy.Publisher("pcl_test",PointCloud2, queue_size = 1, latch = True)
        self.bridge = CvBridge()
        self.color_sub = message_filters.Subscriber("/camera/color/image_raw",Image, queue_size = 1, buff_size = 24*640*500 )
        self.depth_sub = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, queue_size = 1, buff_size = 32*640*500 ) # increase buffer size to avoid delay (despite queue_size = 1)
        self.ats = message_filters.ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], queue_size = 1, slop = 0.3)
        self.ats.registerCallback(self.callback)
        self.cloud_gen = ColorPclGenerator(self.intrinsic, 640,480, "/camera_color_optical_frame" , 0)

    def callback(self, color_img_ros, depth_img_ros):
        #timer = time.time()
        try:
            color_img = self.bridge.imgmsg_to_cv2(color_img_ros, "bgr8")
            depth_img = self.bridge.imgmsg_to_cv2(depth_img_ros, "32FC1")
        except CvBridgeError as e:
            print(e)
        #io.imsave("color_image.png", color_img)
        #io.imsave("depth_image.tiff", depth_img)
        #print("depth image", depth_img)
        #print("image size", depth_img.shape)
        #print("Prepare image took", time.time() - timer)
        #cv2.imshow('depth', depth_img)
        #cv2.imshow('color',color_img)
        #cv2.waitKey(3)
        # Register depth to generate point cloud
        cloud_ros = self.cloud_gen.generate_cloud_color(color_img, depth_img, depth_img_ros.header.stamp)
        #timer = time.time()
        self.pcl_pub.publish(cloud_ros)
        #print("Publish cloud took", time.time() - timer)
        cv2.waitKey()
def main(args):
    rospy.init_node('test_pcl', anonymous=True)
    tpr = testPclRos()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == '__main__':
    main(sys.argv)
