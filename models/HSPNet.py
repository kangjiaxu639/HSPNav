import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import math
from torch_geometric.nn import TransformerConv, GCNConv, global_max_pool, global_mean_pool
from torch_geometric.data import Data
from models.GTNet import UniMPBlock
from models.ResNet import DeconvBottleneck, Bottleneck
from torchstat import stat

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
class HSPNet(nn.Module):
    def __init__(self, downblock, upblock, layer_infos, n_classes, inp=None):
        super(HSPNet, self).__init__()

        # first layer's channel #
        in_channels, kernel_size, stride, padding \
            = tuple(layer_infos[0])
        self.in_channels = in_channels

        self.conv1 = nn.Conv2d(n_classes, self.in_channels,
                               kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.LeakyReLU(inplace=True)

        # maxpool
        kernel_size, stride, padding = tuple(layer_infos[1])
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)

        self.transformer = nn.MultiheadAttention(256, 4)
        # downlayers
        self.dlayer1 = self._make_downlayer(downblock, layer_infos[2])
        self.dlayer2 = self._make_downlayer(downblock, layer_infos[3])

        self.graph1 = UniMPBlock(383, 306, 4, dropout = 0.2)  # node_dim=383, edge_dim=4, heads=4 3层图卷积网络 BERT提取的关系特征
        self.graph2 = UniMPBlock(383, 306, 4, dropout = 0.2)
        self.l1 = nn.Linear(383, 256, bias=False)
        self.l2 = nn.Linear(383, 1, bias=False)
        self.channel_change = nn.Sequential(
            nn.Conv2d(662, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128)
        )

        # uplayers
        self.uplayer1 = self._make_uplayer(upblock, layer_infos[8])
        self.uplayer2 = self._make_uplayer(upblock, layer_infos[9])
        self.uplayer3 = self._make_uplayer(upblock, layer_infos[10])

        upsample = None

        self.uplayer_top = upblock(self.in_channels, self.in_channels,
                                   kernel_size=3, stride=2, padding=1, output_padding=1, upsample=upsample)
        if inp is not None:
            self.conv1_1 = nn.Conv2d(self.in_channels, inp, kernel_size=1,
                                     stride=1, bias=False)
            self.sigmoid = nn.Sigmoid()
        else:
            self.sigmoid = None
            self.conv1_1 = nn.Conv2d(self.in_channels, n_classes, kernel_size=1, stride=1, bias=False)

    def _make_downlayer(self, block, layer_info):
        num_layer, out_channels, kernel_size, stride, padding, dilation \
            = tuple(layer_info)

        downsample = nn.Sequential(
            nn.Conv2d(self.in_channels, out_channels,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        layers = []
        layers.append(block(self.in_channels, out_channels, kernel_size,
                            stride, padding, dilation, downsample))
        self.in_channels = out_channels
        for i in range(1, num_layer):
            layers.append(block(self.in_channels, self.in_channels))
        return nn.Sequential(*layers)

    def _make_uplayer(self, block, layer_info):
        num_layer, out_channels, kernel_size, stride, padding, output_padding, dilation \
            = tuple(layer_info)

        if self.in_channels != out_channels:
            upsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            upsample = None
        layers = []
        for i in range(1, num_layer):
            layers.append(block(self.in_channels, self.in_channels))
        layers.append(block(self.in_channels, out_channels, kernel_size,
                            stride, padding, output_padding, dilation, upsample))

        self.in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x, node_features, edge_features, edge_idx):
        """
        Args:
            x: 1, 41, 128, 128
            node_features:
            edge_features:
            edge_idx:
            num:
            topo:

        Returns:

        """
        x = self.conv1(x)  # 1, 64, 64, 64
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)  # 1, 64, 32, 32

        x = self.dlayer1(x)  # 1, 64, 16, 16
        x = self.dlayer2(x)  # 1, 128, 16, 16

        node_features = node_features.to(torch.float32)
        edge_features = edge_features.to(torch.float32)
        edge_idx = edge_idx.to(torch.long)

        gf = self.graph1(node_features, edge_idx.t(),
                         edge_features)
        gf = self.graph2(gf.x, gf.edge_index, gf.edge_attr)
        gf = gf.x.unsqueeze(0)
        gf = self.l1(gf) # [1, 128, 256]
        x, _ = self.transformer(x.transpose(0, 1).flatten(2), gf.transpose(0, 1), gf.transpose(0, 1))  # [1, 128, 16, 16]
        x = x.transpose(0,1)
        x = x.reshape(x.size(0), x.size(1), 16, 16)

        x = self.uplayer1(x)  # 1, 128, 16, 16
        x = self.uplayer2(x)  # 1, 64, 32, 32
        x = self.uplayer3(x)  # 1, 64, 64, 64

        x = self.uplayer_top(x)  # 1, 64, 128, 128

        x = self.conv1_1(x)  # 1, 1, 128, 128
        if self.sigmoid is not None:
            x = self.sigmoid(x)
        return x

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}



