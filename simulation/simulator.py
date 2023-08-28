import sys
sys.path.append('/home/ros/kjx/SSCNav')
import rospy
from octomap_generator.srv import waypoint, waypointRequest, waypointResponse
import cv2 as cv

from HPNav_sim_agent import HPNavSimAgent
from utils.utils import d3_41_colors_rgb, ScalarMeanTracker

import random
import numpy as np
import shutil
import os
import copy
import json
from tensorboardX import SummaryWriter
import torch
from train_agent import parser
import quaternion as nq

args = parser.parse_args()

# new_eval = args.new_eval
new_eval = True
fake_conf = args.fake_conf
discrete = args.discrete
att = args.att
rc = args.rc
unconf = args.unconf
full_map = args.full_map
args.max_step = 500

args.config_paths = './configs/agent_test.yaml'

args.save_dir = './result_sim/exps/'

args.title = '/home/ros/kjx/SSCNav/result_sim'
save_dir = os.path.join(args.save_dir, args.title)
# if os.path.exists(save_dir):
#     assert False, "Dir exists!"
# os.makedirs(save_dir)

configs = None

args.load_json = './data/val.json'

if args.load_json != "":
    with open(args.load_json, 'r') as f:
        configs = json.load(f)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

agent = HPNavSimAgent(
    device=torch.device(args.device),
    config_paths=args.config_paths,
    flip=args.flip,
    save_dir=save_dir,
    # pano = bool(args.pano),
    pano=False,
    user_semantics=bool(args.user_semantics),
    seg_pretrained=args.seg_pretrained,
    cmplt=bool(args.cmplt),
    cmplt_pretrained=args.cmplt_pretrained,
    conf=bool(args.conf),
    conf_pretrained=args.conf_pretrained,
    targets=args.targets,
    aggregate=bool(args.aggregate),
    # aggregate = True,
    memory_size=args.memory_size,
    num_channel=args.num_channel,
    success_threshold=args.success_threshold,
    collision_threshold=args.collision_threshold,
    ignore=args.ignore,
    training=False,
    Q_pretrained=args.Q_pretrained,
    # Q_pretrained = '../result/1.pth',
    offset=args.offset,
    floor_threshold=args.floor_threshold,
    lr=args.lr,
    momentum=args.momentum,
    weight_decay=args.weight_decay,
    gamma=args.gamma,
    batch_size=args.batch_size,
    # batch_size = 1,
    buffer_size=args.buffer_size,
    height=args.height,
    area_x=args.area_x,
    area_z=args.area_z,
    h=args.h,
    w=args.w,
    h_new=args.h_new,
    w_new=args.w_new,
    max_step=args.max_step,
    # max_step = 5,
    navigable_base=args.navigable_base,
    success_reward=args.success_reward,
    step_penalty=args.step_penalty,
    approach_reward=args.approach_reward,
    collision_penalty=args.collision_penalty,
    # max_dist=args.max_dist,
    max_dist=float("inf"),
    scene_types=args.scene_types,
    double_dqn=bool(args.double_dqn),
    TAU=args.TAU,
    preconf=args.preconf,
    seg_threshold=args.seg_threshold,
    min_dist=0.,
    current_position=None,
    new_eval=new_eval,
    shortest=args.shortest,
    fake_conf=fake_conf,
    discrete=discrete,
    att=att,
    rc=rc,
    unconf=unconf,
    full_map=full_map)



if __name__ == "__main__":

    rospy.init_node('sscnav_node')
    server = rospy.Service("Waypoint", waypoint, agent.doReq)
    rospy.spin()


