from models import QNet, ResNet, Bottleneck,\
DeconvBottleneck, ACNet, Q_discrete

from utils.dataloader_seg import *


import habitat
from habitat.tasks.nav.nav import NavigationEpisode
from habitat.tasks.nav.object_nav_task import (
            ObjectGoal,
                ObjectGoalNavEpisode,
                    ObjectViewLocation,
                    )

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision

import pickle
import _pickle as cPickle

import random
import numpy as np
#import pandas as pd
from quaternion import as_rotation_matrix, from_rotation_matrix
from utils.utils import generate_pc, color2local3d, repeat4, pc2local, pc2local_gpu, d3_41_colors_rgb
from collections import namedtuple, OrderedDict
import copy
import os
from utils.mapper import Mapper
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
import message_filters
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped, Point
import cv2
import rospy
from octomap_generator.srv import waypoint, waypointRequest, waypointResponse
#import skfmm
name2id = {
        'chair': 2,
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
layer_infos = [
                  [64, 7, 2, 3],
                  [3, 2, 1],
                  [12, 64, 3, 2, 1, 1],
                  [16, 128, 3, 1, 1, 1],
                  [24, 256, 3, 1, 1, 1],
                  [12, 512, 3, 1, 1, 1],
                  [12, 512, 3, 1, 1, 0, 1],
                  [24, 256, 3, 1, 1, 0, 1],
                  [16, 128, 3, 1, 1, 0, 1],
                  [12, 64, 3, 2, 1, 1, 1],
                  [4, 64, 3, 2, 1, 1, 1]
    ]
# 数据格式变化
transform = transforms.Compose([
    scaleNorm(),
    ToTensor(),
    Normalize()
    ])

Transition = namedtuple('Transition',
        ('state', 'action', 'next_state', 'reward'))

class HPNavSimAgent:
    def __init__(
            self, 
            device, 
            config_paths,
            flip,
            pano,
            user_semantics, 
            seg_pretrained,
            cmplt, 
            cmplt_pretrained,
            conf,
            conf_pretrained,
            targets, 
            aggregate, 
            memory_size, 
            num_channel,
            success_threshold, 
            collision_threshold, 
            ignore, 
            training, 
            Q_pretrained,
            offset,
            floor_threshold, 
            lr, 
            momentum,
            weight_decay,
            gamma, 
            batch_size, 
            buffer_size,
            height,
            area_x, 
            area_z, 
            h, 
            w,
            h_new,
            w_new,
            max_step,
            navigable_base,
            success_reward,
            step_penalty,
            approach_reward,
            collision_penalty,
            save_dir,
            scene_types,
            max_dist,
            double_dqn,
            TAU,
            preconf,
            seg_threshold,
            current_position,
            min_dist=0.,
            shortest=False,
            new_eval=False,
            fake_conf=False,
            discrete=False,
            att=False,
            rc=False,
            unconf=False,
            full_map=False,
            num_local=25,
            adj=11
            ):
        self.adj = adj
        self.num_local = num_local
        self.new_eval = new_eval
        assert self.new_eval, "Only support challenge setting!"
        self.shortest = shortest # 非最短路径

        self.min_dist = min_dist
        self.TAU = TAU
        self.double_dqn = double_dqn
        self.scene_types = scene_types.split("|")

        self.max_dist = max_dist
        self.seg_threshold = seg_threshold
        self.success_reward = success_reward
        self.step_penalty = step_penalty
        self.approach_reward = approach_reward
        self.collision_penalty = collision_penalty
        
        self.rc = rc
        self.unconf = unconf
        if self.unconf:
            assert fake_conf, "currently only one-hot completion when fake confidence map is provided"
        self.batch_size = batch_size
        self.device = device
        self.save_dir = save_dir

        # create environment
        # disable habitat's metrics
        config = habitat.get_config(config_paths=config_paths)
        config.defrost()
        config.TASK.SUCCESS_DISTANCE = -float("inf")
        config.ENVIRONMENT.MAX_EPISODE_STEPS = float("inf")
        config.ENVIRONMENT.MAX_EPISODE_SECONDS = float("inf")

        config.freeze()
        self.env = habitat.Env(config=config)

        self.num_channel = num_channel
        self.ignore = ignore.split("|")
        self.ignore = [int(ig) for ig in self.ignore]
        self.offset = offset
        self.floor_threshold = floor_threshold
        self.success_threshold = success_threshold
        self.collision_threshold = collision_threshold


        self.user_semantics = user_semantics
        self.max_step = max_step
        if self.new_eval:
            self.max_step = 500
        self.navigable_base = navigable_base.split("|")
        self.navigable_base = [int(base) for base in self.navigable_base]
        if user_semantics: # 语义分割模型
            self.seg_model = ACNet(num_class = num_channel - 1)
            self.seg_model.load_state_dict(torch.load(seg_pretrained))
            self.seg_model = torch.nn.DataParallel(self.seg_model).to(device)
            self.seg_model.eval()
        self.cmplt = cmplt
        self.fake_conf = fake_conf
        self.conf = conf 
        if cmplt: # 部分观察补全网络
            self.cmplt_model = ResNet(Bottleneck, DeconvBottleneck,
                    layer_infos, num_channel).to(device)
            self.cmplt_model.load_state_dict(torch.load(cmplt_pretrained))
            self.cmplt_model = torch.nn.DataParallel(self.cmplt_model)
            self.cmplt_model.eval()
            if conf and not self.fake_conf:
                # 置信度模型
                self.conf_model = ResNet(Bottleneck, DeconvBottleneck,
                        layer_infos, num_channel, inp=1).to(device)

                self.conf_model.load_state_dict(torch.load(conf_pretrained))
                self.conf_model = torch.nn.DataParallel(self.conf_model)
                self.conf_model.eval()
        self.pano = pano
        self.discrete = discrete
        self.att = att
        if discrete:
            self.Q\
            = Q_discrete(self.num_channel, (not att and conf) or
                    self.fake_conf, preconf=preconf)
        else:
            self.Q\
            = QNet(self.num_channel, (not att and conf) or self.fake_conf, 
                    rc=rc, preconf=preconf)

        if Q_pretrained != "":
            state_dict = torch.load(Q_pretrained)
            own_state = self.Q.state_dict()
            try:
                for name, param in state_dict.items():
                    if name not in own_state:
                        continue
                    own_state[name].copy_(param)

            except:
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] # remove `module.`
                    new_state_dict[name] = v

                for name, param in state_dict.items():
                    if name not in own_state:
                        continue

                    own_state[name].copy_(param)
        self.Q = torch.nn.DataParallel(self.Q).to(device)
        self.training = training
        if training:
            if discrete:
                self.Q_t\
                = Q_discrete(self.num_channel, (not att and conf) or self.fake_conf,
                         preconf=preconf)
            else:
                self.Q_t\
                = QNet(self.num_channel, (not att and conf) or self.fake_conf,
                        rc=rc, preconf=preconf)
            self.Q_t.load_state_dict(self.Q.module.state_dict())
            self.Q_t.eval()

            self.optimizer = optim.SGD(self.Q.parameters(), lr=lr,
                    momentum=momentum, weight_decay=weight_decay) 
            self.gamma = gamma
            self.Q_t = torch.nn.DataParallel(self.Q_t).to(device)
        else:
            self.Q.eval()



        self.targets = targets.split("|")
        self.aggregate = aggregate
        self.memory_size = memory_size



        self.height = height
        self.area_x = area_x
        self.area_z = area_z
        self.h = h
        self.w = w
        self.h_new = h_new
        self.w_new = w_new
        
        self.d2x = np.zeros((480, 640, 1))
        for i in range(480):
            for j in range(640):
                self.d2x[i, j, 0] =np.sqrt(3) / 2. + (240. - i) /640.

        self.mapper = Mapper(self.device, 1024, 1024, 48., 48.,
                    self.num_channel - 1, 2.0, self.ignore)
        # every episode
        self.memory = None
        self.target = None
        self.target_map = None
        self.target_objects = []
        self.target_radiuss = []
        self.target_positions = []
        self.best_path_length = float("inf")
        self.path_length = 0.
        self.eps_len = 0.
        self.reward = 0.
        self.action = None
        self.raw_semantics = None
        self.current_obs = None
        self.cmplted_obs = None
        self.conf_obs = None
        self.old_state = None
        self.state = None
        self.q_map = None

        self.obstacle = None
        self.action_list = []

        self.done = False
        self.success = False

        self.image = None
        self.depth = None
        
        self.navigable = None
        self.flip = flip
        self.episode = None
        
        self.full_map = full_map
        self.bridge = CvBridge() # CvBridge to transform ROS Image message to OpenCV image
        self.step_flag = True # 判断这个时候的信息是否被使用
        self.observation = {}
    
    def embedding(self, target): # 嵌入
        embed = torch.zeros(1, self.num_channel, self.h, self.w)
        embed[:, target, ...] = 1
        return embed

    def get_trinsics(self):
        # 返回航向和位置
        quaternion = self.rotation
        translation = self.position
        return quaternion, translation
    def get_observations(self):
        return self.observation

    def euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(position_b - position_a, ord=2)
    
    def action_picker(self, eps_threshold):
        # distribution?
        # deterministic?
        sample = random.random()
        if sample > eps_threshold:
            
            tmp = torch.argmax(self.q_map.view(-1))
            a_w = tmp % self.w_new 
            a_h = (tmp - a_w) / self.w_new
            self.action = (a_h, a_w)

        else:
            self.action = (random.randint(0, self.h_new - 1), \
                    random.randint(0, self.w_new - 1))
    
    def planner(self, eps_threshold):
        """面向像素点方向前进0.25m"""
        self.action_picker(eps_threshold)
        action_list = []
        source = (self.h_new / 2 - 0.5, self.w_new / 2 - 0.5) # 源坐标, 每个角度是相对于cmap来说的
        
        
        # rotate to face target
        angle = None
        if self.action[1] > source[1]:
            if self.action[0] < source[0]:
                angle = np.arctan((source[0] - self.action[0]) / (self.action[1] - source[1]))
                angle = np.pi - angle
            else:
                angle = np.arctan((self.action[0] - source[0])\
                        / (self.action[1] - source[1]))
                angle = np.pi + angle

        elif self.action[0] < source[0]:
            angle = np.arctan((source[0] - self.action[0]) / (source[1] - self.action[1]))
            # angle = -angle
            angle = angle

        else:
            angle = np.arctan((self.action[0] - source[0]) / (source[1] - self.action[1]))
            angle = -angle
        action_list.append(angle)
        dis = ((source[1] - self.action[1]) ** 2 + (source[0] - self.action[0]) ** 2) ** 0.5
        action_list.append(dis)
        action_list.append("MOVE_FORWARD")

        return action_list # 返回位置坐标结点，并计算选择角度

    def stop_checker(self):
        
        if not self.user_semantics: # 如果不使用语义地图使用成功检测器
            return self.success_checker()
        else:
            depth = self.get_observations()['depth']
            targets = [self.target]
            
            observations = self.get_observations()
            rgb = observations['rgb']
            dep = observations['depth']
            sample = {'image': rgb[..., [2, 1, 0]],
                    'depth': dep[..., 0],
                    'label': dep[..., 0]}
            sample = transform(sample)       
            rgb = sample['image'].unsqueeze(0)
            dep = sample['depth'].unsqueeze(0)
            with torch.no_grad():
                raw_semantics = self.seg_model(rgb.to(self.device),
                        dep.to(self.device)).detach()[0]
                raw_semantics = torch.argmax(raw_semantics,
                        dim=0).cpu().numpy()

            depth = depth * self.d2x
            legal = None
            for target in targets:
                if legal is None:
                    legal = (raw_semantics != target)
                else:
                    legal = legal & (raw_semantics != target)

            legal = legal | (depth[..., 0] == 0.)
            check = (~legal) & (depth[..., 0] <= self.success_threshold)

            if np.sum(check.astype('int')) > self.seg_threshold:
                return True

            # self.old_position = 4 # old position
            return False
        
    def step(self, randomness):
        # 在这里智能体执行每一步，即确定一个长期导航目标 
        if self.eps_len == self.max_step:
            print("The eps_len >= max_step! The episode files!")
            return
        self.eps_len += 1 # 步长加一
        self.step_flag = False # 确定一个新的导航点智能体在到达新到导航点之前不接受新的信息
        # think about which point to go towards 
        with torch.no_grad():
            self.q_map\
            = self.Q(self.state.to(self.device)).detach().cpu()[0][0]


        action_list = self.planner(randomness) # 找到偏航角
        print("action:", self.action)
        # 计算waypoint的坐标
        dx = 0.5 * np.cos(action_list[0])# 分辨率
        dy = 0.5 * np.sin(action_list[0])
        print(dy,dx)
        self.way_ponit_pos = [dy, dx]



    def view(self):
        """Q-Net Input update state. e.g. Python view() 重定义网络形状"""

        self.state = torch.cat((self.current_obs, self.target_map),
                dim=1)
        with torch.no_grad():
            if self.cmplt:
                self.cmplted_obs\
                    = self.cmplt_model(self.current_obs.to(self.device)).detach().cpu() 
                if self.unconf:
                    max_idx = torch.argmax(self.cmplted_obs, 1, keepdim=True)
                    normed_cmplt\
                    = torch.FloatTensor(self.cmplted_obs.shape)
                    normed_cmplt.zero_()
                    normed_cmplt.scatter_(1, max_idx, 1)
                else:
                    normed_cmplt = F.softmax(self.cmplted_obs, dim=1)
                normed_cmplt = torch.rot90(normed_cmplt, -1, dims=[2,3])
                normed_cmplt = torch.flip(normed_cmplt, [3])
                self.state = torch.cat((normed_cmplt, self.target_map), dim=1)
                if self.fake_conf:
                    self.conf_obs\
                            = (torch.argmax(self.current_obs,
                                    dim=1)!=self.num_channel-1).unsqueeze(1).float()

                    self.state = torch.cat((self.state, self.conf_obs), dim=1)

                elif self.conf:
                    self.conf_obs\
                            = self.conf_model(torch.cat((self.current_obs.to(self.device),
                                normed_cmplt.to(self.device)), dim=1)).detach().cpu()

                    if self.rc:
                        self.seen = (torch.argmax(self.current_obs, dim=1) !=
                                (self.num_channel - 1)).float()

                        self.seen = self.seen.unsqueeze(0)

                  #  self.conf_obs[..., torch.argmax(self.current_obs,dim=1) !=
                   #         self.num_channel - 1] = 1.

                    if not self.att:
                        self.state = torch.cat((self.state, self.conf_obs), dim=1)
                    else:
                        normed_cmplt = normed_cmplt*(1+self.conf_obs) / 2.
                        self.state = torch.cat((normed_cmplt, self.target_map),
                                dim=1)
                    if self.rc:
                        self.state = torch.cat((self.state, self.seen), dim=1) # [1,83,128,128]
        # resize state for Q Net
        assert self.h == self.h_new, "no resizing for now"
    def rgb_to_semantic(self, obs):
        obs = obs.transpose(2,1,0)
        c, h, w = np.shape(obs)
        semantic = np.zeros((self.num_channel, h, w))
        for i in range(h):
            for j in range(w):
                for k in range(self.num_channel):
                    color = d3_41_colors_rgb[k]
                    color_distance = np.abs((color[0]-obs[0, i, j])) + \
                                     np.abs((color[1] - obs[1, i, j])) + \
                                     np.abs((color[2] - obs[2, i, j]))
                    if color_distance <= 3:
                        semantic[k, i, j] = 1
                        break
        return semantic


    def navigate(self, target):
        self.target = name2id[target]
        # embed target as part of state
        self.target_map = self.embedding(self.target)
        start_position = [0., 0.]

        rgb_topic = '/rgbd_camera/color/image'
        depth_topic = '/rgbd_camera/depth/image'
        postion_topic = '/state_estimation'
        way_point_topic  = '/way_point'
        current_obs_topic = '/local_map'
        self.color_sub = message_filters.Subscriber(rgb_topic, Image, queue_size = 10, buff_size = 30*480*640)
        self.depth_sub = message_filters.Subscriber(depth_topic, Image, queue_size = 10, buff_size = 40*480*640 ) # increase buffer size to avoid delay (despite queue_size = 1)
        # self.current_obs_sub = message_filters.Subscriber(current_obs_topic, Image, queue_size = 10, buff_size = 40*128*128)
        self.pos_sub = message_filters.Subscriber(postion_topic, Odometry)
        self.way_ponit_pub = rospy.Publisher(way_point_topic, PointStamped, queue_size=1)
        ts = message_filters.ApproximateTimeSynchronizer([self.color_sub, self.depth_sub, self.pos_sub], queue_size = 10, slop = 3) # Take in one color image and one depth image with a limite time gap between message time stamps
        ts.registerCallback(self.color_depth_pos_callback)

    def doReq(self, req):
        """返回图片"""
        target = req.target
        self.target = name2id[target]
        # embed target as part of state
        self.target_map = self.embedding(self.target)
        # 1. 获得局部语义地图
        req.image.encoding = "bgr8"
        current_obs = self.bridge.imgmsg_to_cv2(req.image)
        if not (np.shape(current_obs)[0] == 0):
            cv2.imwrite(os.path.join(self.save_dir, "%s_%s_localmap_%s.png"
                                         % (0, self.eps_len, self.target)),
                                         current_obs)
        if req.image.height == 0 or req.image.width == 0:
            resp = waypointResponse()
            resp.x = -1.0
            resp.y = -1.0
            return resp
        self.current_obs = torch.Tensor(self.rgb_to_semantic(current_obs)).unsqueeze(0).float()
        # 2. 返回节点坐标

        self.view()  # 获取视图
        self.step(-1.)  # 前进一步
        cv2.imwrite(os.path.join(self.save_dir, "%s_%s_obs_%s.png"
                                         % (0, self.eps_len, self.target)),
                            d3_41_colors_rgb[torch.argmax(self.state[0,
                                                          :self.num_channel, ...], dim=0)])

        resp = waypointResponse()
        resp.x = self.way_ponit_pos[0]
        resp.y = self.way_ponit_pos[1]
        return resp


        

