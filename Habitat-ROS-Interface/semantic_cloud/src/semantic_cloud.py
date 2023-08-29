#!/home/ros/anaconda3/envs/semantic/bin/python
#!coding:utf-8
"""
Take in an image (rgb or rgb-d)
Use CNN to do semantic segmantation
Out put a cloud point with semantic color registered
\author Xuan Zhang
\date May - July 2018
"""

from __future__ import division
from __future__ import print_function

import sys
sys.path.append('/home/ros/kjx/semantic_ws/src/semantic_cloud/include/')
import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import String 
from cv_bridge import CvBridge, CvBridgeError

import numpy as np

from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from color_pcl_generator import PointType, ColorPclGenerator
import message_filters
import time

from skimage.transform import resize
import cv2

import torch
from ptsemseg.models import get_model
from ptsemseg.utils import convert_state_dict

from utils.dataloader_seg import *
import os
from sklearn.cluster import KMeans
from utils.mapper import Mapper
from utils.utils import d3_41_colors_rgb
from utils.tf_utils import euler_to_quaternion, quaternion_to_euler
import quaternion
name2id = {
        'chair':2,
        'door': 3,
        'table': 4,
        'sofa': 9,
        'bed': 10,
        'sink': 14,
        'toilet': 17,
        'bathtub': 24,
        'shower': 22,
        'counter': 25
        }
def compress_semmap(semmap):
    device = semmap.device
    c_map = torch.zeros((semmap.shape[1], semmap.shape[2]))
    for i in range(0, semmap.shape[0]):
        c_map[semmap[i] > 0.] = i
    return c_map.int().to(device)

def color_map(N=256, normalized=False):
    """
    Return Color Map in PASCAL VOC format (rgb)
    对每一个标签索引分配相应的rgb分量，从而使得原始的标签分割图片转化为上色后的彩色分割图片
    根据序号索引，每一个类别分配一个颜色，制作调色板  按照BGR图片的格式传递图片
    \param N (int) number of classes
    \param normalized (bool) whether colors are normalized (float 0-1)
    \return (Nx3 numpy array) a color map
    """
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)
    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3
        cmap[i] = np.array([r, g, b])
    cmap = cmap/255.0 if normalized else cmap
    return cmap

def decode_segmap(temp, n_classes, cmap):
    """
    Given an image of class predictions, produce an bgr8 image with class colors
    根据种类索引为语义分割后图片着色
    \param temp (2d numpy int array) input image with semantic classes (as integer)
    \param n_classes (int) number of classes 物体种类
    \cmap (Nx3 numpy array) input color map
    \return (numpy array bgr8) the decoded image with class colors
    """
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, n_classes):
        # l = int(Forty2Thirteen.get(str(l)))
        r[temp == l] = cmap[l,2]  # temp是标签，标签中为l的部分说明这里有l类的物体，分别进行rgb着色
        g[temp == l] = cmap[l,1]
        b[temp == l] = cmap[l,0]
    bgr = np.zeros((temp.shape[0], temp.shape[1], 3))
    bgr[:, :, 0] = b
    bgr[:, :, 1] = g
    bgr[:, :, 2] = r
    return bgr.astype(np.uint8)

class SemanticCloud:
    """
    Class for ros node to take in a color image (bgr) and do semantic segmantation on it to produce an image with semantic class colors (chair, desk etc.)
    Then produce point cloud based on depth information
    CNN: PSPNet (https://arxiv.org/abs/1612.01105) (with resnet50) pretrained on ADE20K, fine tuned on SUNRGBD or not
    """
    def __init__(self, gen_pcl = True):
        """
        Constructor
        \param gen_pcl (bool) whether generate point cloud, if set to true the node will subscribe to depth image
        """
        # Success threshold
        self.success_threshold = 1.0
        self.seg_threshold = 2000
        self.last_translation = None
        # Get point type
        point_type = rospy.get_param('/semantic_pcl/point_type')
        if point_type == 0:
            self.point_type = PointType.COLOR
            print('Generate color point cloud.')
        elif point_type == 1:
            self.point_type = PointType.SEMANTICS_MAX
            print('Generate semantic point cloud [max fusion].')
        elif point_type == 2:
            self.point_type = PointType.SEMANTICS_BAYESIAN
            print('Generate semantic point cloud [bayesian fusion].')
        else:
            print("Invalid point type.")
            return
        # Get image size
        self.img_width, self.img_height = rospy.get_param('/camera/width'), rospy.get_param('/camera/height')
        # Set up CNN is use semantics
        if self.point_type is not PointType.COLOR:
            print('Setting up CNN model...')
            # Set device
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # Get dataset
            dataset = rospy.get_param('/semantic_pcl/dataset')
            # Setup model
            model_name ='ACNet'
            model_path = rospy.get_param('/semantic_pcl/model_path')
            if dataset == 'sunrgbd': # If use version fine tuned on sunrgbd dataset
                self.n_classes = 38 # Semantic class number
                self.model = get_model(model_name, self.n_classes, version = 'sunrgbd_res50')
                state = torch.load(model_path,map_location='cuda:0')
                self.model.load_state_dict(state)
                self.cnn_input_size = (321, 321)
                self.mean = np.array([104.00699, 116.66877, 122.67892]) # Mean value of dataset
            elif dataset == 'ade20k':
                self.n_classes = 150 # Semantic class number
                self.model = get_model(model_name, self.n_classes, version = 'ade20k')
                state = torch.load(model_path)
                self.model.load_state_dict(convert_state_dict(state['model_state'])) # Remove 'module' from dictionary keys
                self.cnn_input_size = (473, 473)
                self.mean = np.array([104.00699, 116.66877, 122.67892]) # Mean value of dataset
            elif dataset == 'MP3D':
                self.n_classes = 41
                self.model =  get_model(model_name, self.n_classes, version = 'mp3d')
                state = torch.load(model_path,map_location='cuda:0')
                self.model.load_state_dict(state)
                self.cnn_input_size = (640, 480)
                self.mean = np.array([104.00699, 116.66877, 122.67892]) # Mean value of dataset
            self.model = self.model.to(self.device)
            self.model.eval()
            # self.cmap = color_map(N = self.n_classes, normalized = False) # Color map for semantic classes
            self.mapper = Mapper(self.device, 512, 512, 12., 12., self.n_classes - 1, 2.0, [17, 40])
            self.cmap = np.array(
            [
                [31, 119, 180],
                [174, 199, 232],
                [255, 127, 14], # chair
                [255, 187, 120],
                [44, 160, 44],
                [152, 223, 138],
                [214, 39, 40],
                [255, 152, 150],
                [148, 103, 189],
                [197, 176, 213],# sofa  [97, 100, 255]
                [140, 86, 75],
                [196, 156, 148],
                [227, 119, 194],
                [247, 182, 210],
                [127, 127, 127],
                [199, 199, 199],
                [188, 189, 34],
                [219, 219, 141],
                [23, 190, 207],
                [158, 218, 229],
                [57, 59, 121],
                [82, 84, 163],
                [107, 110, 207],
                [156, 158, 222],
                [99, 121, 57],
                [140, 162, 82],
                [181, 207, 107],
                [206, 219, 156],
                [140, 109, 49],
                [189, 158, 57],
                [231, 186, 82],
                [231, 203, 148],
                [132, 60, 57],
                [173, 73, 74],
                [214, 97, 107],
                [231, 150, 156],
                [123, 65, 115],
                [165, 81, 148],
                [206, 109, 189],
                [222, 158, 214], # misc
                [255, 255, 255] # 白色背景
            ],
            dtype=np.uint8,
        )
        self.ros_count = 0
        # Declare array containers
        if self.point_type is PointType.SEMANTICS_BAYESIAN:
            self.semantic_colors = np.zeros((3, self.img_height, self.img_width, 3), dtype = np.uint8) # Numpy array to store 3 decoded semantic images with highest confidences
            self.confidences = np.zeros((3, self.img_height, self.img_width), dtype = np.float32) # Numpy array to store top 3 class confidences
        
        # Set up ROS
        print('Setting up ROS...')
        self.bridge = CvBridge() # CvBridge to transform ROS Image message to OpenCV image
        # Semantic image publisher
        self.sem_img_pub = rospy.Publisher("/semantic_pcl/semantic_image", Image, queue_size = 50)
        # Set up ros image subscriber
        # Set buff_size to average msg size to avoid accumulating delay
        if gen_pcl:
            # Point cloud frame id
            frame_id = rospy.get_param('/semantic_pcl/frame_id')
            # Camera intrinsic matrix
            fx = rospy.get_param('/camera/fx')
            fy = rospy.get_param('/camera/fy')
            cx = rospy.get_param('/camera/cx')
            cy = rospy.get_param('/camera/cy')

            intrinsic = np.matrix([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype = np.float32)
            self.pcl_pub = rospy.Publisher("/semantic_pcl/semantic_pcl", PointCloud2, queue_size = 100)
            self.color_sub = message_filters.Subscriber(rospy.get_param('/semantic_pcl/color_image_topic'), Image, queue_size = 50, buff_size = 30*480*640)
            self.depth_sub = message_filters.Subscriber(rospy.get_param('/semantic_pcl/depth_image_topic'), Image, queue_size = 50, buff_size = 40*480*640 ) # increase buffer size to avoid delay (despite queue_size = 1)
            self.state_sub = message_filters.Subscriber('/state_estimation',  Odometry, queue_size = 50)
            self.ts = message_filters.ApproximateTimeSynchronizer([self.color_sub, self.depth_sub, self.state_sub], queue_size = 1, slop = 0.3) # Take in one color image and one depth image with a limite time gap between message time stamps
            self.ts.registerCallback(self.color_depth_callback)
            self.cloud_generator = ColorPclGenerator(intrinsic, self.img_width,self.img_height, frame_id , self.point_type)
            
        else:
            self.image_sub = rospy.Subscriber(rospy.get_param('/semantic_pcl/color_image_topic'), Image, self.color_callback, queue_size = 50, buff_size = 30*480*640)
        self.stop_checker_sub = rospy.Subscriber("stop_check", String, self.stop_checker, queue_size = 1)
        self.stop_pub = rospy.Publisher("/nav_stop", String, queue_size = 1)
        self.sscnav_map_pub = rospy.Publisher("/sscnav_global_map", Image, queue_size = 1)

        print('Ready.')
        self.last_translation = None
    def color_callback(self, color_img_ros):
        """
        Callback function for color image, de semantic segmantation and show the decoded image. For test purpose
        \param color_img_ros (sensor_msgs.Image) input ros color image message
        """
        print('callback')
        try:
            color_img = self.bridge.imgmsg_to_cv2(color_img_ros, "bgr8") # Convert ros msg to numpy array
        except CvBridgeError as e:
            print(e)
        # Do semantic segmantation
        class_probs = self.predict(color_img)
        confidence, label = class_probs.max(1)
        confidence, label = confidence.squeeze(0).numpy(), label.squeeze(0).numpy()
        label = resize(label, (self.img_height, self.img_width), order = 0, mode = 'reflect', anti_aliasing=False, preserve_range = True) # order = 0, nearest neighbour
        label = label.astype(np.int)
        # Add semantic class colors
        decoded = decode_segmap(label, self.n_classes, self.cmap)        # Show input image and decoded image
        confidence = resize(confidence, (self.img_height, self.img_width),  mode = 'reflect', anti_aliasing=True, preserve_range = True)
        cv2.imshow('Camera image', color_img)
        cv2.imshow('confidence', confidence)
        cv2.imshow('Semantic segmantation', decoded)
        cv2.waitKey(3)

    def color_depth_callback(self, color_img_ros, depth_img_ros, state_ros):
        """
        Callback function to produce point cloud registered with semantic class color based on input color image and depth image
        在RGB-D图像的基础上，生成点云，并标注语义分割
        \param color_img_ros (sensor_msgs.Image) the input color image (bgr8)
        \param depth_img_ros (sensor_msgs.Image) the input depth image (registered to the color image frame) (float32) values are in meters
        """
        # Convert ros Image message to numpy array
        try:
            color_img = self.bridge.imgmsg_to_cv2(color_img_ros, "bgr8")
            depth_img = self.bridge.imgmsg_to_cv2(depth_img_ros, "32FC1")
            self.translation = np.array([state_ros.pose.pose.position.x, state_ros.pose.pose.position.y, state_ros.pose.pose.position.z])
            self.quaternion = np.array([ state_ros.pose.pose.orientation.x, state_ros.pose.pose.orientation.y, state_ros.pose.pose.orientation.z, state_ros.pose.pose.orientation.w], dtype=np.float64)
            self.quaternion = quaternion.from_float_array(self.quaternion)
            # print(self.quaternion)
        except CvBridgeError as e:
            print(e)
        # Resize depth
        if depth_img.shape[0] is not self.img_height or depth_img.shape[1] is not self.img_width:
            depth_img = resize(depth_img, (self.img_height, self.img_width), order = 0, mode = 'reflect', anti_aliasing=False, preserve_range = True) # order = 0, nearest neighbour
            depth_img = depth_img.astype(np.float32)
        self.depth_img = depth_img
        self.color_img = color_img
        self.depth_img[np.isnan(self.depth_img)] = 0.0
        
        if self.point_type is PointType.COLOR:
            cloud_ros = self.cloud_generator.generate_cloud_color(color_img, depth_img, color_img_ros.header.stamp)
        else:
            # Do semantic segmantation
            if self.point_type is PointType.SEMANTICS_MAX:
                semantic_color, pred_confidence = self.predict_max(color_img)
                # 点云信息
                cloud_ros = self.cloud_generator.generate_cloud_semantic_max(color_img, depth_img, semantic_color, pred_confidence, color_img_ros.header.stamp)

            elif self.point_type is PointType.SEMANTICS_BAYESIAN:
                self.predict_bayesian(color_img, depth_img) # change self.semantic_color 着色 self.confidence 置信度
                # Produce point cloud with rgb colors, semantic colors and confidences
                cloud_ros = self.cloud_generator.generate_cloud_semantic_bayesian(color_img, depth_img, self.semantic_colors, self.confidences, color_img_ros.header.stamp)
            
            # Publish semantic image
            if self.sem_img_pub.get_num_connections() > 0:
                if self.point_type is PointType.SEMANTICS_MAX:
                    semantic_color_msg = self.bridge.cv2_to_imgmsg(semantic_color, encoding="bgr8")
                else:
                    semantic_color_msg = self.bridge.cv2_to_imgmsg(self.semantic_colors[0], encoding="bgr8")
                self.sem_img_pub.publish(semantic_color_msg)
        # Publish point cloud
        self.pcl_pub.publish(cloud_ros)

        self.observations = {
            'rgb': self.color_img,
            'depth': np.expand_dims(self.depth_img, axis=2),
            'semantics': self.raw_semantics
        }
        self.height = 1.5
        self.offset = 0.3
        self.floor_threshold = 0.1
        if self.ros_count == 0:
            # roof_thre = self.translation[1] + self.height / 2. + self.offset
            # floor_thre = self.translation[1] - self.height / 2.  - self.floor_threshold
            roof_thre =  1.8
            floor_thre = -0.3
            
            self.mapper.reset(None, roof_thre, floor_thre)
            self.ros_count += 1
        self.mapper.append(self.quaternion, self.translation, self.observations, self.raw_semantics)
        # 发送语义地图
        sscnav_global_map = compress_semmap(self.mapper.get_map_global())
        sscnav_global_map = self.cmap[sscnav_global_map]
        msg = self.bridge.cv2_to_imgmsg(sscnav_global_map, encoding="bgr8")
        self.sscnav_map_pub.publish(msg)
        

        
    def predict_max(self, img, depth):

        """
        Do semantic prediction for max fusion
        \param img (numpy array rgb8)
        """
        class_probs = self.predict(img, depth) #N,C,W,H(1,150,473,473)
        self.raw_semantics = torch.argmax(class_probs, dim=0).numpy() # [1,128, 128]
        # Take best prediction and confidence
        pred_confidence, pred_label = class_probs.max(1)
        #pred_confidence(1*473*473)每个值对应pred_label每个类别的置信度
        #pred_label(1*473*473)每个值对应每个像素点所属的类别标签
        pred_confidence = pred_confidence.squeeze(0).cpu().numpy()
        pred_label = pred_label.squeeze(0).cpu().numpy()
        pred_label = resize(pred_label, (self.img_height, self.img_width), order = 0, mode = 'reflect', anti_aliasing=False, preserve_range = True) # order = 0, nearest neighbour
        pred_label = pred_label.astype(np.int)
        
        semantic_color = decode_segmap(pred_label, self.n_classes, self.cmap)
        pred_confidence = resize(pred_confidence, (self.img_height, self.img_width),  mode = 'reflect', anti_aliasing=True, preserve_range = True)
        return (semantic_color, pred_confidence)

    def predict_bayesian(self, img, depth):
        """
        Do semantic prediction for bayesian fusion 语义分割贝叶斯融合
        \param img (numpy array rgb8)
        """
        class_probs = self.predict(img,depth)
        self.raw_semantics = torch.argmax(class_probs, dim=0).numpy() # [1,128, 128] 获取语义地图
        # Take 3 best predictions and their confidences (probabilities)
        
        pred_confidences, pred_labels  = torch.topk(input = class_probs, k = 3, dim = 1, largest = True, sorted = True)
        
        pred_labels = pred_labels.squeeze(0).cpu().numpy() # [3, 480, 640]
        pred_confidences = pred_confidences.squeeze(0).cpu().numpy()
        # Resize predicted labels and confidences to original image size
        for i in range(pred_labels.shape[0]):
            pred_labels_resized = resize(pred_labels[i], (self.img_height, self.img_width), order = 0, mode = 'reflect', anti_aliasing=False, preserve_range = True) # order = 0, nearest neighbour
            pred_labels_resized = pred_labels_resized.astype(np.int) # 分割后的语义标签
            # print("pred_label",np.max(pred_labels_resized),np.min(pred_labels_resized))
            # Add semantic class colors
            self.semantic_colors[i] = decode_segmap(pred_labels_resized, self.n_classes, self.cmap)
            if i == 0:
                self.raw_semantics = pred_labels_resized
        
        for i in range(pred_confidences.shape[0]):
            self.confidences[i] = resize(pred_confidences[i], (self.img_height, self.img_width),  mode = 'reflect', anti_aliasing=True, preserve_range = True)

    def predict(self, img, depth):
        """
        Do semantic segmantation
        \param img: (numpy array bgr8) The input cv image
        """
        img = img.copy() # Make a copy of image because the method will modify the image
        # orig_size = (img.shape[0], img.shape[1]) # Original image size
        # Prepare image: first resize to CNN input size then extract the mean value of SUNRGBD dataset. No normalization
        img = resize(img, self.cnn_input_size, mode = 'reflect', anti_aliasing=True, preserve_range = True) # Give float64
        img = img.astype(np.float32)
        
        # Convert WHC -> HWC
        img = img.transpose(1, 0, 2)
        transform =  transforms.Compose([
             scaleNorm(),
             ToTensor(),
             Normalize()
        ])
        depth[np.isnan(depth)] = np.min(depth[~np.isnan(depth)])
        sample = {
            'image' : img,
            'depth' : depth,
            'label': depth
        }
        sample = transform(sample)
        img = sample['image']
        depth = sample['depth']
        
        img = img.unsqueeze(0)
        depth = depth.unsqueeze(0)
       
        with torch.no_grad():
            img = img.to(self.device)
            depth = depth.to(self.device)
        
            outputs = self.model(img, depth).detach().cpu().float() #获取语义分割图像
            # Apply softmax to obtain normalized probabilities
            outputs = torch.nn.functional.softmax(outputs, 1) #N,C,W,H(1,40,480,640)
            return outputs

    def  stop_checker(self, goal):
        target = name2id[str(goal.data)]
        
        legal = None
        if legal is None:
            legal = (self.raw_semantics != target)
        else:
            legal = legal & (self.raw_semantics != target)

        legal = legal | (self.depth_img == 0.)
        check = (~legal) & (self.depth_img <= self.success_threshold)
        

        if np.sum(check.astype('int')) > self.seg_threshold:
            print("The target %s is found successfully!" %(target))
            cv2.imwrite('/home/ros/kjx/semantic_ws/src/semantic_cloud/src/success_current_sem.png', self.semantic_colors[0])
            cv2.imwrite('/home/ros/kjx/semantic_ws/src/semantic_cloud/src/success_current_color.png', self.color_img)
            
            img = cv2.hconcat([self.semantic_colors[0],self.color_img])
           
            cv2.imshow("The target %s is found successfully!" %(target), img)
            cv2.waitKey(0)
            msg = String()
            msg.data = "nav_stop"
            self.stop_pub.publish(msg)
            os.kill(os.getpid()+1, 9)
            os.kill(os.getpid(), 9)
            return True
        else:
            msg = String()
            msg.data = "no_nav_stop"
            self.stop_pub.publish(msg)
        
        return False
    
def main(args):
    rospy.init_node('semantic_cloud', anonymous=True)
    seg_cnn = SemanticCloud(gen_pcl = True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == '__main__':
    main(sys.argv)
