import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from attn_block import ChannelAttention, SpatialAttention

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm3d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class Fusion(nn.Module):
    def __init__(self, in_channels=1024):
        super(Fusion, self).__init__()
        self.in_channels = in_channels
        self.ca = ChannelAttention(in_channels=in_channels, reduction_ratio=16)
        self.sa1 = SpatialAttention(kernel_size=3)
        self.sa2 = SpatialAttention(kernel_size=3)

    def forward(self, x_tv, x_sv):
        # b, c, h, w, d = x_tv.size()

        x_fusion = torch.cat((x_tv, x_sv), dim=1)
        x_fusion = self.ca(x_fusion)
        x_tv, x_sv = x_fusion[:, :self.in_channels // 2], x_fusion[:, self.in_channels // 2:]
        x_tv = self.sa1(x_tv)
        x_sv = self.sa2(x_sv)
        x_fusion = torch.cat((x_tv, x_sv), dim=1)
        return x_fusion

