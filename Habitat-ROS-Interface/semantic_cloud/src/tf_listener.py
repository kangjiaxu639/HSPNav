#!/home/ros/anaconda3/envs/python27/bin/python
#!coding:utf-8
import tf
import rospy
from nav_msgs.msg import Odometry
rospy.init_node('sscnav_tf_node', anonymous=True)
rate = rospy.Rate(1.0)
listener = tf.TransformListener()
tf_pub = rospy.Publisher("/sscnav_tf", Odometry, queue_size = 1)
tf_msg = Odometry()
while not rospy.is_shutdown():
    try:
        now = rospy.Time.now()
        listener.waitForTransform('/map', '/rgbd_camera', now, rospy.Duration(0.2))
        (trans, rot) = listener.lookupTransform('/map', '/rgbd_camera', now)
        # print(trans[0], trans[1], trans[2], rot[0], rot[1], rot[2], rot[3])
        tf_msg.header.stamp = now
        tf_msg.pose.pose.position.x = trans[0]
        tf_msg.pose.pose.position.y = trans[1]
        tf_msg.pose.pose.position.z = trans[2]
        
        tf_msg.pose.pose.orientation.w = rot[3]
        tf_msg.pose.pose.orientation.x = rot[0]
        tf_msg.pose.pose.orientation.y = rot[1]
        tf_msg.pose.pose.orientation.z = rot[2]
        
        tf_pub.publish(tf_msg)
    except (tf.Exception, tf.LookupException, tf.ConnectivityException):
        print("tf echo error")
        continue
    rate.sleep()
