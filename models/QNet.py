import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ResNet import DeconvBottleneck, Bottleneck
from models.HSPNet import HSPNet
from utils.utils import compress_semmap

layer_infos = [
                  [64, 7, 2, 3],
                  [3, 2, 1],
                  [3, 64, 3, 2, 1, 1],
                  [4, 128, 3, 1, 1, 1],
                  [24, 256, 3, 1, 1, 1],
                  [12, 512, 3, 1, 1, 1],
                  [12, 512, 3, 1, 1, 0, 1],
                  [24, 256, 3, 1, 1, 0, 1],
                  [4, 128, 3, 1, 1, 0, 1],
                  [3, 64, 3, 2, 1, 1, 1],
                  [2, 64, 3, 2, 1, 1, 1]
    ]

def make_uplayer(block, layer_info, in_channels):
    num_layer, out_channels, kernel_size, stride, padding, output_padding, dilation \
        = tuple(layer_info)

    if in_channels != out_channels:
        upsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    else:
        upsample = None
    layers = []
    for i in range(1, num_layer):
        layers.append(block(in_channels, in_channels))
    layers.append(block(in_channels, out_channels, kernel_size,
                        stride, padding, output_padding, dilation, upsample))

    in_channels = out_channels

    return nn.Sequential(*layers)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,\
            padding=dilation, groups=groups, bias=True, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,\
            bias=True)

class BasicBlock(nn.Module):
    """ResNet-18中的双层残差模块"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class QNet(nn.Module):

    def __init__(self, num_channel, conf, rc, hsp_layer_infos, layers=[2, 2, 2, 2], preconf=False):
        super(QNet, self).__init__()
        
        '''
        self.pre_goal = nn.Sequential(
                nn.Conv2d(num_channel, num_channel, kernel_size=3, stride=1,
                padding=1),
                nn.LeakyReLU(inplace=True)
                )
        if conf: 
            self.pre_conf = nn.Sequential(
                nn.Conv2d(num_channel, num_channel, kernel_size=3, stride=1,
                    padding=1),
                nn.LeakyReLU(inplace=True)
                )
        
        '''
        self.conf = conf
        self.rc = rc
        
        self.num_channel = num_channel
        self.attention_map = None

        block = BasicBlock
        self.inplanes = 64
        self.dilation = 1
        assert not preconf, "out-of-date preconf layers"
        self.hsp_net = HSPNet(Bottleneck, DeconvBottleneck, hsp_layer_infos, 1, inp=1) # HSPNet handle the semantic map with size (1, w, h)
        if self.conf and preconf:
            self.preconf = nn.Conv2d(1, 1, kernel_size=1, stride=1)
        else:
            self.preconf = None
        self.conv1 = nn.Conv2d(num_channel*2 + int(self.conf) + int(self.rc) + 1, self.inplanes, kernel_size=7, stride=2,
                padding=3)
        self.relu = nn.LeakyReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)


        self.conv2 = nn.Conv2d(512, 128, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(128, 32, kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(32, 1, kernel_size=1, stride=1)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                        mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, global_graph):

        '''
        obs = x[:, :self.num_channel,...]
        goal = x[:, self.num_channel:2*self.num_channel, ...]
        goal = self.pre_goal(goal)

        if self.conf:
            conf = x[:, 2*self.num_channel:, ...]
            conf = self.pre_conf(conf)
            x = torch.cat((obs, goal, conf), dim=1)
        else:
            x = torch.cat((obs, goal), dim=1)
        '''
        torch.cuda.synchronize()

        self.attention_maps = None
        for i in range(x.shape[0]):
            obs = x[i, :self.num_channel, ...].squeeze(0) # [1, 41, 128, 128]

            semmap = compress_semmap(obs).unsqueeze(0).unsqueeze(0)
            # self.attention_map = torch.ones((1, 1, 128, 128)).to(x.device) * 0.05
            if global_graph[i]["edge_features"].shape[0] == 0:
                self.attention_map = torch.ones((1, 1, 128, 128)).to(x.device)
            else:
                self.attention_map = self.hsp_net(semmap, global_graph[i]["node_features"],
                                    global_graph[i]["edge_features"], global_graph[i]["edge_idx"])
            if not (self.attention_maps is None):
                self.attention_maps = torch.cat((self.attention_maps, self.attention_map), dim=0)
            else:
                self.attention_maps = self.attention_map

        x = torch.cat((x, self.attention_maps), dim=1)
        if self.preconf is not None:
            conf = x[:, -1:, ...]
            normed_conf = self.preconf(conf)
            x = torch.cat((x[:, :-1, ...], normed_conf), dim=1)

        x = self.conv1(x) # [1, 84, 128, 128]
        x = self.relu(x)
        x = self.maxpool(x) # [1, 64, 32, 32]

        x = self.layer1(x) # [1, 64, 32, 32]
        x = self.layer2(x) # [1, 128, 32, 32]
        x = self.layer3(x) # [1, 256, 32, 32]
        x = self.layer4(x) # [1, 512, 32, 32]


        x = self.conv2(x)
        x = self.relu(x) # [1, 128, 32, 32]
        x = F.interpolate(x, scale_factor=2, mode='bilinear',
                align_corners=True) # [1, 128, 64, 64]
        x = self.conv3(x)
        x = self.relu(x) # [1, 32, 64, 64]
        x = F.interpolate(x, scale_factor=2, mode='bilinear',
                align_corners=True) #  [1, 32, 128, 128]
        x = self.conv4(x) # [1, 1, 128, 128]

        return x

class QNet512(nn.Module):

    def __init__(self, num_channel, layers=[2, 2, 2, 2, 2]):
        super(QNet512, self).__init__()

        block = BasicBlock
        self.inplanes = 64
        self.dilation = 1
        
        self.conv1 = nn.Conv2d(num_channel, self.inplanes, kernel_size=7, stride=2,
                padding=3)
        self.relu = nn.LeakyReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
       # self.layer5 = self._make_layer(block, 512, layers[4], stride=2)

        self.conv2 = nn.Conv2d(512, 128, kernel_size=1, stride=1)
        self.conv3 = nn.Conv2d(128, 32, kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(32, 1, kernel_size=1, stride=1)
       # self.conv5 = nn.Conv2d(64, 32, kernel_size=1, stride=1)
       # self.conv6 = nn.Conv2d(32, 1, kernel_size=1, stride=1)
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                        mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):


        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
       # x = self.layer5(x)
     #   print(x.shape)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear',
                align_corners=True)
        x = self.conv3(x)
        x = self.relu(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear',
                align_corners=True)
        x = self.conv4(x)
      #  x = self.relu(x)
      #  x = F.interpolate(x, scale_factor=2, mode="bilinear",
      #          align_corners=True)
      #  x = self.conv5(x)
      #  x = self.relu(x)
      #  x = F.interpolate(x, scale_factor=2, mode="bilinear",
      #          align_corners=True)

      #  x = self.conv6(x)

        return x
class Q_discrete(nn.Module):

    def __init__(self, num_channel, conf, layers=[2, 2, 2, 2], preconf=False):
        super(Q_discrete, self).__init__()
        
        '''
        self.pre_goal = nn.Sequential(
                nn.Conv2d(num_channel, num_channel, kernel_size=3, stride=1,
                padding=1),
                nn.LeakyReLU(inplace=True)
                )
        if conf: 
            self.pre_conf = nn.Sequential(
                nn.Conv2d(num_channel, num_channel, kernel_size=3, stride=1,
                    padding=1),
                nn.LeakyReLU(inplace=True)
                )
        
        '''
        self.conf = conf
        
        self.num_channel = num_channel

        block = BasicBlock
        self.inplanes = 64
        self.dilation = 1
        if self.conf and preconf:
            self.preconf = nn.Conv2d(1, 1, kernel_size=1, stride=1)
        else:
            self.preconf = None
        self.conv1 = nn.Conv2d(num_channel*2 + int(self.conf), self.inplanes, kernel_size=7, stride=2,
                padding=3)
        self.relu = nn.LeakyReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
      #  self.layer5 = self._make_layer(block, 512, layers[4], stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 8)

      #  self.conv2 = nn.Conv2d(512, 128, kernel_size=3, stride=2)
      #  self.conv3 = nn.Conv2d(128, 32, kernel_size=3, stride=2)
      #  self.conv4 = nn.Conv2d(32, 1, kernel_size=3, stride=2)
      #  self.conv5 = nn.Conv2d(128, 64, kernel_size=1, stride=1)
      #  self.conv6 = nn.Conv2d(64, 32, kernel_size=1, stride=1)
      #  self.conv7 = nn.Conv2d(32, 1, kernel_size=1, stride=1) 

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                        mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride))
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):

        '''
        obs = x[:, :self.num_channel,...]
        goal = x[:, self.num_channel:2*self.num_channel, ...]
        goal = self.pre_goal(goal)

        if self.conf:
            conf = x[:, 2*self.num_channel:, ...]
            conf = self.pre_conf(conf)
            x = torch.cat((obs, goal, conf), dim=1)
        else:
            x = torch.cat((obs, goal), dim=1)
'''
        if self.preconf is not None:
            conf = x[:, -1:, ...]
            normed_conf = self.preconf(conf)
            x = torch.cat((x[:, :-1, ...], normed_conf), dim=1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
     #   x = self.layer5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
if __name__ == "__main__":
    Q_net = QNet(41, True, rc=False)
    inp = torch.rand(3, 83, 128, 128)
    output = Q_net(inp)      #[3,1,128,128]

    inp = torch.rand(1, 1, 128, 128)
    conv1 = nn.Conv2d(1, 40, kernel_size=7, stride=2,
                           padding=3)
    conv2 = nn.Conv2d(40, 64, kernel_size=7, stride=2,
                      padding=3)
    relu = nn.LeakyReLU(inplace=True)
    maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    output = conv1(inp)
    output = relu(output)
    output = maxpool(output)
    output = conv2(output)

    # 上采样 1*1*128*128
    output = F.interpolate(output, scale_factor=8, mode='bilinear',
                      align_corners=True)  # [1, 128, 64, 64]

    uplayer1 = make_uplayer(DeconvBottleneck, layer_infos[8], 64)
    uplayer2 = make_uplayer(DeconvBottleneck, layer_infos[9], 64)
    uplayer3 = make_uplayer(DeconvBottleneck, layer_infos[10], 64)

    uplayer_top = DeconvBottleneck(64, 64,
                               kernel_size=3, stride=2, padding=1, output_padding=1, upsample=None)

    x = output
    x = uplayer1(x)
    x = uplayer2(x)
    x = uplayer3(x)
    x = uplayer_top(x)
    
    conv3 = conv1x1(64, 1)
    output = conv3(output)
    print(output.shape)

