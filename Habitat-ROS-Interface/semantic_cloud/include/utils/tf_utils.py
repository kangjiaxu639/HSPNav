#!/home/ros/anaconda3/envs/python27/bin/python
#!coding:utf-8
import math
import numpy as np
import quaternion
def euler_to_quaternion(roll, pitch, yaw):
    w=math.cos(pitch/2)*math.cos(yaw/2)*math.cos(roll/2) + math.sin(pitch/2)*math.sin(yaw/2)*math.sin(roll/2)
    x=math.sin(pitch/2)*math.cos(yaw/2)*math.cos(roll/2) - math.cos(pitch/2)*math.sin(yaw/2)*math.sin(roll/2)
    y=math.cos(pitch/2)*math.sin(yaw/2)*math.cos(roll/2) + math.sin(pitch/2)*math.cos(yaw/2)*math.sin(roll/2)
    z=math.cos(pitch/2)*math.cos(yaw/2)*math.sin(roll/2) - math.sin(pitch/2)*math.sin(yaw/2)*math.cos(roll/2)
    return (w, x, -y, z)

def quaternion_to_euler(orientation):
    x, y, z, w = orientation.x, orientation.y, orientation.z, orientation.w
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.sin(2 * (w * y - x * z))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))
    return roll, pitch, yaw

# print(euler_to_quaternion(-1.571, -0.000, 0.720 ))
# qu = quaternion.from_float_array([0.527652550955, 0, 0,  0.849460290697])
# print(quaternion_to_euler(qu))