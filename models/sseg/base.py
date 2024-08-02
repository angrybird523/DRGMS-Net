import torch
from torch import nn
import numpy as np
import math
import torch.nn.functional as F
from models.backbone.ddrnet import DualResNet_imagenet
from torch.nn import BatchNorm2d
import matplotlib.pyplot as plt
class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                              out_chan,
                              kernel_size=ks,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
class StripAttentionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(StripAttentionModule, self).__init__()
        self.conv1 = ConvBNReLU(in_chan, 64, ks=1, stride=1, padding=0)
        self.conv2 = ConvBNReLU(in_chan, 64, ks=1, stride=1, padding=0)
        self.conv3 = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim=1)

        self.init_weight()

    def forward(self, x):
        q = self.conv1(x)
        batchsize, c_middle, h, w = q.size()
        q = F.avg_pool2d(q, [h, 1])
        q = q.view(batchsize, c_middle, -1).permute(0, 2, 1)

        k = self.conv2(x)
        k = k.view(batchsize, c_middle, -1)
        attention_map = torch.bmm(q, k)
        attention_map = self.softmax(attention_map)

        v = self.conv3(x)
        c_out = v.size()[1]
        v = F.avg_pool2d(v, [h, 1])
        v = v.view(batchsize, c_out, -1)

        augmented_feature_map = torch.bmm(v, attention_map)
        augmented_feature_map = augmented_feature_map.view(batchsize, c_out, h, w)
        out = x + augmented_feature_map
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params
def get_backbone(backbone, pretrained):
    if backbone == 'ddrnet':
        backbone = DualResNet_imagenet(pretrained)
    else:
        exit("\nError: BACKBONE \'%s\' is not implemented!\n" % backbone)

    return backbone

class BaseNet(nn.Module):
    def __init__(self, backbone, pretrained):
        super(BaseNet, self).__init__()
        self.backbone_name = backbone
        self.backbone = get_backbone(backbone, pretrained)
        self.relu = nn.ReLU(inplace=True)

        #SPA MOdule
        self.sam = StripAttentionModule(256, 256)

        self.res = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1, bias=False)
        self.conv = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1, bias=False)
    def base_forward(self, x1, x2):
        b, c, h, w = x1.shape

        assert len(x1.shape) == 4, f"Expected x1 to be 4D, got shape {x1.shape}"
        assert len(x2.shape) == 4, f"Expected x2 to be 4D, got shape {x2.shape}"

        #特征提取
        x1, x2 = self.backbone.base_forward(x1, x2)

        out1 = self.head(x1)
        out2 = self.head(x2)

        #上采样 使用双线性插值bilinear将out1和out2的大小上采样至原始输入图像x1和x2的尺寸
        out1 = F.interpolate(out1, size=(h, w), mode='bilinear', align_corners=False)
        out2 = F.interpolate(out2, size=(h, w), mode='bilinear', align_corners=False)


        #---------------------------------------------
        info = self.conv(x1 + x2)

        disappear = self.res(self.relu(x1 - x2))
        appear = self.res(self.relu(x2 - x1))
        out_bin = info + disappear +appear
        out_bin = self.relu(out_bin)

        out_bin = self.sam(out_bin)
        out_bin = self.head_bin(out_bin)
        #------------------------------------

        # # 进行图像差异并进行二值化
        # out_bin = torch.abs(x1 - x2)
        # out_bin = self.head_bin(out_bin)

        out_bin = F.interpolate(out_bin, size=(h, w), mode='bilinear', align_corners=False)
        out_bin = torch.sigmoid(out_bin)

        #输出 函数返回三个输出：‘out1’和‘out2’是两个输入图像语义分割结果，‘out_bin’表示两个输入图像差异二值化结果
        return out1, out2, out_bin.squeeze(1)


#特征增强操作：由于进行了6次不同的增强并且每次将增强的输出添加到原始输出上。由于进行了六次增强所以最终输出除以6
    def forward(self, x1, x2, tta=False):

        if self.backbone_name == 'ddrnet' in self.backbone_name:

            if not tta:
                return self.base_forward(x1, x2)
            else:
                out1, out2, out_bin = self.base_forward(x1, x2)
                out1 = F.softmax(out1, dim=1)
                out2 = F.softmax(out2, dim=1)
                out_bin = out_bin.unsqueeze(1)
                origin_x1 = x1.clone()
                origin_x2 = x2.clone()

                x1 = origin_x1.flip(2)
                x2 = origin_x2.flip(2)
                cur_out1, cur_out2, cur_out_bin = self.base_forward(x1, x2)
                out1 += F.softmax(cur_out1, dim=1).flip(2)
                out2 += F.softmax(cur_out2, dim=1).flip(2)
                out_bin += cur_out_bin.unsqueeze(1).flip(2)

                x1 = origin_x1.flip(3)
                x2 = origin_x2.flip(3)
                cur_out1, cur_out2, cur_out_bin = self.base_forward(x1, x2)
                out1 += F.softmax(cur_out1, dim=1).flip(3)
                out2 += F.softmax(cur_out2, dim=1).flip(3)
                out_bin += cur_out_bin.unsqueeze(1).flip(3)

                x1 = origin_x1.transpose(2, 3).flip(3)
                x2 = origin_x2.transpose(2, 3).flip(3)
                cur_out1, cur_out2, cur_out_bin = self.base_forward(x1, x2)
                out1 += F.softmax(cur_out1, dim=1).flip(3).transpose(2, 3)
                out2 += F.softmax(cur_out2, dim=1).flip(3).transpose(2, 3)
                out_bin += cur_out_bin.unsqueeze(1).flip(3).transpose(2, 3)

                x1 = origin_x1.flip(3).transpose(2, 3)
                x2 = origin_x2.flip(3).transpose(2, 3)
                cur_out1, cur_out2, cur_out_bin = self.base_forward(x1, x2)
                out1 += F.softmax(cur_out1, dim=1).transpose(2, 3).flip(3)
                out2 += F.softmax(cur_out2, dim=1).transpose(2, 3).flip(3)
                out_bin += cur_out_bin.unsqueeze(1).transpose(2, 3).flip(3)

                x1 = origin_x1.flip(2).flip(3)
                x2 = origin_x2.flip(2).flip(3)
                cur_out1, cur_out2, cur_out_bin = self.base_forward(x1, x2)
                out1 += F.softmax(cur_out1, dim=1).flip(3).flip(2)
                out2 += F.softmax(cur_out2, dim=1).flip(3).flip(2)
                out_bin += cur_out_bin.unsqueeze(1).flip(3).flip(2)

                out1 /= 6.0
                out2 /= 6.0
                out_bin /= 6.0

                return out1, out2, out_bin.squeeze(1)



