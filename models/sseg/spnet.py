from __future__ import division
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import interpolate
import torch.nn.functional as F
from models.sseg.base import BaseNet
from models.sseg.fcn import FCNHead


class SPNet(BaseNet):
    def __init__(self, backbone, pretrained, nclass, lightweight):
        super(SPNet, self).__init__(backbone, pretrained)

        in_channels = self.backbone.channels[-1]

        self.head = FCNHead(in_channels, nclass, lightweight)
        self.head_bin = SPHead(in_channels, 1, lightweight)

#用于实现SPNet中的模块
class SPHead(nn.Module):
    def __init__(self, in_channels, out_channels, lightweight):
        super(SPHead, self).__init__()
        inter_channels = in_channels // 2
        self.trans_layer = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, 1, 0, bias=False),
                                         nn.BatchNorm2d(inter_channels),
                                         nn.ReLU(True)
                                         )
        self.strip_pool1 = StripPooling(inter_channels, (20, 12))
        self.strip_pool2 = StripPooling(inter_channels, (20, 12))
        self.score_layer = nn.Sequential(nn.Conv2d(inter_channels, inter_channels // 2, 3, 1, 1, bias=False),
                                         nn.BatchNorm2d(inter_channels // 2),
                                         nn.ReLU(True),
                                         nn.Dropout2d(0.1, False),
                                         nn.Conv2d(inter_channels // 2, out_channels, 1))

    def forward(self, x):
        x = self.trans_layer(x) # 2048->1024
        x = self.strip_pool1(x) # 第一个MPM
        x = self.strip_pool2(x) # 第二个MPM
        x = self.score_layer(x) # 经过一个1*1卷积和一个3*3卷积实现预测
        return x


class StripPooling(nn.Module):

    def __init__(self, in_channels, pool_size):
        super(StripPooling, self).__init__()
        #空间池化
        self.pool1 = nn.AdaptiveAvgPool2d(pool_size[0])
        self.pool2 = nn.AdaptiveAvgPool2d(pool_size[1])
        #strip pooling
        self.pool3 = nn.AdaptiveAvgPool2d((1, None))  # 这意味着这个池化层在高度方向上自适应地将输入张量的高度池化到1（高度固定为1），而在宽度方向上不进行池化（宽度保持不变）
        self.pool4 = nn.AdaptiveAvgPool2d((None, 1))

        inter_channels = int(in_channels/4)
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                nn.BatchNorm2d(inter_channels),
                                nn.ReLU(True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                nn.BatchNorm2d(inter_channels),
                                nn.ReLU(True))
        self.conv2_0 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                nn.BatchNorm2d(inter_channels))
        self.conv2_1 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                nn.BatchNorm2d(inter_channels))
        self.conv2_2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                nn.BatchNorm2d(inter_channels))
        self.conv2_3 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (1, 3), 1, (0, 1), bias=False),
                                nn.BatchNorm2d(inter_channels))
        self.conv2_4 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, (3, 1), 1, (1, 0), bias=False),
                                nn.BatchNorm2d(inter_channels))
        self.conv2_5 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                nn.BatchNorm2d(inter_channels),
                                nn.ReLU(True))
        self.conv2_6 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, 1, 1, bias=False),
                                nn.BatchNorm2d(inter_channels),
                                nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(inter_channels*2, in_channels, 1, bias=False),
                                nn.BatchNorm2d(in_channels))


    def forward(self, x):
        _, _, h, w = x.size()
        x1 = self.conv1_1(x)
        x2 = self.conv1_2(x)
        x2_1 = self.conv2_0(x1)
        x2_2 = F.interpolate(self.conv2_1(self.pool1(x1)), (h, w), mode="bilinear", align_corners=False)
        x2_3 = F.interpolate(self.conv2_2(self.pool2(x1)), (h, w), mode="bilinear", align_corners=False)
        x2_4 = F.interpolate(self.conv2_3(self.pool3(x2)), (h, w), mode="bilinear", align_corners=False)
        x2_5 = F.interpolate(self.conv2_4(self.pool4(x2)), (h, w), mode="bilinear", align_corners=False)
        x1 = self.conv2_5(F.relu_(x2_1 + x2_2 + x2_3))
        x2 = self.conv2_6(F.relu_(x2_5 + x2_4))
        out = self.conv3(torch.cat([x1, x2], dim=1))
        return F.relu_(x + out)



