#!/home/ros/anaconda3/envs/semantic/bin/python
#!coding:utf-8
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import message_filters
import cv2 as cv
image_count = 0
bridge = CvBridge()
def color_map_callback(color_img_ros):
    
    color_img = bridge.imgmsg_to_cv2(color_img_ros, "bgr8")
    
    global image_count
    image_count += 1
    path = '/home/ros/kjx/semantic_ws/data/'
    color_img_name = 'color_img_' + str(image_count) + '.png'
   
    cv.imwrite(path + color_img_name, color_img)
   
    


rospy.init_node('sscnav_map_node', anonymous=True)

color_sub = rospy.Subscriber('rgbd_camera/color/image', Image,color_map_callback,queue_size=10)


rospy.spin()


