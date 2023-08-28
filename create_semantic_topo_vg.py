import sys
sys.path.append('/home/ros/kjx/HSPNav')
import math
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from itertools import chain
from sklearn.cluster import MeanShift
from math import *
import json
import pandas as pd
from copy import deepcopy
from utils.utils import d3_41_colors_rgb, compress_semmap
import cv2 as cv
from skimage import morphology
import os

semantic = ['wall', 'floor', 'chair', 'door', 'table', 'picture', 'cabinet', 'cushion', 'window', 'sofa',
            'bed', 'curtain', 'chest_of_drawers', 'plant', 'sink', 'stairs', 'ceiling', 'toilet', 'stool',
            'towel', 'mirror', 'tv_monitor', 'shower', 'column', 'bathtub', 'counter', 'fireplace', 'lighting',
            'beam', 'railing', 'shelving', 'blinds', 'gym_equipment', 'seating', 'board_panel', 'furniture',
            'appliances', 'clothes', 'objects', 'misc']

id2name = { # 拓扑图中存在的物体类别
    3: 'door',
    4: 'table',
    9: 'sofa',
    10: 'bed',
    14: 'sink',
    17: 'toilet',
    22: 'shower',
    24: 'bathtub',
    25: 'counter',
}

name2id = {
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

json_data = None

with open('./knowledge/object_encoding.json','r',encoding='utf8')as fp:
    json_data = json.load(fp) # 结点的嵌入特征 300

FILE = './knowledge/matterport3D_rels.json'
All_relation_and_weight = {}
with open(FILE,'r',encoding='utf8')as fp:
    All_relation_and_weight = json.load(fp) # 关系图中两个结点之间的关系

relation_data = None
with open('./knowledge/relationship_encoding.json','r',encoding='utf8')as fp:
    relation_data = json.load(fp) # 结点边的嵌入特征 300


save_dir = "./object_relation_graphs"

if os.path.exists(save_dir):
    assert False, "Dir exists!"
os.makedirs(save_dir)

def embedding(target):
    # 拓扑图结点的嵌入
    embed = torch.zeros(41, 128, 128)
    embed[target, ...] = 1
    return embed

def sorted_by_theta(points, center):
    x = []
    for point in points:
        theta = math.atan2(point[1] - center[1], point[0] - center[0])
        x.append(theta)
    x = np.array(x)
    return np.argsort(x)


def build_node_contours(mask, center):
    mask = np.array(mask, np.uint8)

    mask = cv.bilateralFilter(mask, 3, 50, 50)

    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    epsilon = 0.01 * cv.arcLength(contours[0], True)
    contours = cv.approxPolyDP(contours[0], epsilon, True) # 多边形拟合

    contours = [contours[i][0].tolist() for i in range(len(contours))]
    if contours is None or len(contours) <= 1:
        return None
    if len(contours) >= 20:
        # index = np.random.randint(0, len(contours),10)
        step = int(len(contours) / 20)
        idx = [x for x in range(0, len(contours), step)]
        contours = [contours[x] for x in idx[:20]]
    elif len(contours) < 20:
        diff = 20 - len(contours)
        i = 0
        while True:
            x = (contours[i][0] + contours[i + 1][0]) / 2
            y = (contours[i][1] + contours[i + 1][1]) / 2
            contours.insert(i + 1, [x, y])
            i = i + 2
            diff -= 1
            if diff == 0:
                break
            elif diff > 0 and i >= (len(contours) - 1):
                i = 0
    index = sorted_by_theta(contours, center)
    contours = [contours[idx] for idx in index]
    return list(chain.from_iterable(contours))

def align_tensors(tensor_a, tensor_b):
    if tensor_a.size()[1] < tensor_b.size()[1]:
        diff_len = tensor_b.size()[1] - tensor_a.size()[1]
        zeros = torch.zeros((tensor_a.size()[0], diff_len, tensor_b.size()[2]))
        if tensor_a.size()[1] == 0:
            tensor_a = torch.zeros((tensor_a.size()[0], diff_len, tensor_b.size()[2]))
        else:
            tensor_a = torch.cat((tensor_a, zeros), dim=1)

    elif tensor_a.size()[1] > tensor_b.size()[1]:
        diff_len = tensor_a.size()[1] - tensor_b.size()[1]
        zeros = torch.zeros((tensor_b.size()[0], diff_len, tensor_a.size()[2]))
        if tensor_b.size()[1] == 0:
            tensor_b = torch.zeros((tensor_b.size()[0], diff_len, tensor_a.size()[2]))
        else:
            tensor_b = torch.cat((tensor_b, zeros), dim=1)
    return tensor_a, tensor_b

def get_relation_weight(relations, object_i, object_j):
    relationship = None
    weight = -1
    if object_j in relations[object_i].keys():
        rel_and_weight = relations[object_i][object_j]
        rel = list(rel_and_weight.keys())[0]
        if relations[object_i][object_j][rel] >= 3:
            relationship = rel
            weight = relations[object_i][object_j][rel]
    return relationship, weight

def compute_topo_for_map(semmap):
    bs = semmap.shape[0]
    for batch in range(bs):
        cats = torch.sum(torch.sum(semmap[batch, ...], dim=2), dim=1).tolist()
        cats = [int(i > 0) for i in cats]  # cats 判断当前语义地图上是否有该类物体存在

        idx = []  # 各个层的聚类中心
        sem_and_center = []
        node_features = []
        for c in range(2, semmap[batch, ...].shape[0] - 1):
            if (not (semantic[c] in json_data)):
                continue
            channel = semmap[batch, c, ...].numpy().astype(bool)
            channel = morphology.remove_small_objects(channel, min_size=50)
            channel = channel.astype(np.uint8)
            ## 图像细化和膨胀
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
            channel = cv.dilate(channel, kernel, iterations = 1)
            # 划分连通域
            num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(channel, connectivity=8)

            if num_labels > 1:
                for l in range(1, num_labels):
                    sem_embed = np.zeros(41)
                    sem_embed[c] = 1
                    sem_embed = sem_embed.tolist()
                    target_node_feature = deepcopy(json_data[semantic[c]]) # 300
                    target_node_feature.extend(sem_embed)# 41

                    locations = np.argwhere(labels == l)
                    mask = np.zeros((np.shape(channel)[0], np.shape(channel)[1]))
                    if len(locations) < 50: # 阴影区域小于10个点不考虑
                        continue
                    for localtion in locations:
                        mask[localtion[0], localtion[1]] = 1.0

                    contour_features = build_node_contours(mask, centroids[l])
                    if contour_features is None:
                        continue
                    target_node_feature.extend(centroids[l])  # 2
                    target_node_feature.extend(contour_features)  # 40
                    node_features.append(target_node_feature) # [383, 1]

                    idx = list(chain(idx, [centroids[l]]))  # 聚类中心，一个位置只能有一个聚类中心
                    center = centroids[l].tolist()
                    center.append(c)
                    sem_and_center.append(center)

        if len(idx) > 125:
            index = random.sample(range(0, len(idx)), 125)
            node_features = [node_features[i] for i in index]
            idx = [idx[i] for i in index]
            sem_and_center = [sem_and_center[i] for i in index]

        edge_idx = []
        edge_features = []
        for i in range(len(idx)):
            for j in range(len(idx)):
                dis = sqrt(pow(idx[i][0] - idx[j][0], 2) + pow(idx[i][1] - idx[j][1], 2)) * 0.05  # 欧式距离
                if i != j and dis < 5.0:
                    # 真实世界中 dis < 1.25m 考虑这样一条边有相邻关系
                    if [i, j] in edge_idx or [j, i] in edge_idx:
                        continue
                    relation = None
                    if semantic[sem_and_center[i][2]] != semantic[sem_and_center[j][2]]:
                        """不对同一类作处理"""
                        object_i = semantic[sem_and_center[i][2]].replace(' ', '_')
                        object_j = semantic[sem_and_center[j][2]].replace(' ', '_')

                        ralation_ij, weight_ij = get_relation_weight(All_relation_and_weight, object_i, object_j)
                        ralation_ji, weight_ji = get_relation_weight(All_relation_and_weight, object_j, object_i)

                        if weight_ij > weight_ji:
                            relation = ralation_ij
                            weight = weight_ij
                        else:
                            relation = ralation_ji
                            weight = weight_ji
                    if relation:
                        edge_idx.append([i, j])
                        # 对边特征进行进一步处理
                        start_x, start_y = idx[i][0], idx[i][1]
                        end_x, end_y = idx[j][0], idx[j][1]
                        relation_embedding = deepcopy(relation_data[relation]) # 300
                        edge_feature = [start_x, start_y, end_x, end_y, dis, weight] + relation_embedding # [306,1]
                        edge_features.append(edge_feature) # list

        node_features = torch.Tensor(node_features)
        edge_features = torch.Tensor(edge_features)
        edge_idx = torch.Tensor(edge_idx).long()

    #################################################################################
        global_map = compress_semmap(semmap[batch]).numpy().astype(int)
        global_map = d3_41_colors_rgb[global_map]
        global_map = global_map[:, :, ::-1]


        w = global_map.shape[0]
        h = global_map.shape[1]
        dpi = 128
        fig = plt.figure(figsize=(w/dpi, h/dpi), dpi=dpi)
        axes = fig.add_axes([0,0,1,1])
        axes.set_axis_off()
        if len(node_features) != 0:
            axes.imshow(global_map)
            nodes = node_features[:, 341:343] # 结点中心坐标
            contour_features = node_features[:, 343:383]
            edge_idxs = edge_idx
            print(len(nodes), len(contour_features), len(edge_idxs))
            idx = np.array(nodes)
            x = idx[:, 0]
            y = idx[:, 1]
            # axes.scatter(x, y, color='r', alpha=1, s=100)
            # for k in range(len(edge_idxs)):
            #     if len(edge_idxs) == 0:
            #         break
            #     i = edge_idxs[k][0]
            #     j = edge_idxs[k][1]
            #     axes.plot([idx[i][0], idx[j][0]], [idx[i][1], idx[j][1]], color='hotpink', alpha=1, linewidth=5.0) # 画边
            # 画轮廓
            for i in range(len(contour_features)):
                for j in range(0, 39, 2):
                    axes.scatter(contour_features[i, j], contour_features[i, j + 1], color='g', alpha=0.8, s=80)
                for m in range(0, 39, 2):
                    if m==38:
                        axes.plot([contour_features[i, m], contour_features[i, 0]],
                                 [contour_features[i, m + 1], contour_features[i, 1]], color='greenyellow', alpha=0.8, linewidth=6.0)

                    else:
                        axes.plot([contour_features[i, m], contour_features[i, m + 2]], [contour_features[i, m + 1], contour_features[i, m + 3]], color='greenyellow', alpha=0.8, linewidth=6.0)

            plt.show()
            fig.savefig(os.path.join(save_dir, 'semantic_topo_%s.png' % (batch)), bbox_inches='tight')


    return node_features, edge_features, edge_idx

if __name__ == "__main__":
    mask = np.random.randint(0,2,(1024,1024))
    c = build_node_contours(mask)
    print(c)



