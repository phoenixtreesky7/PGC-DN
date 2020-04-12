# -*- coding: utf-8 -*-
import numpy as np

from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision


class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, act_func=nn.ReLU(inplace=True)):
        super(VGGBlock, self).__init__()
        self.act_func = act_func
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_func(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act_func(out)

        return out


class UNet(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt

        nb_filter = [32, 64, 128, 256, 512, 512, 512, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(opt.input_nc, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], 1, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))
        x4_0 = self.conv4_0(self.pool(x3_0))

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))

        output = self.final(x0_4)
        return output


class NestedUNet(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt

        nb_filter = [32, 64, 128, 256, 512, 512, 512, 512]

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(opt.input_nc, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])
        self.conv5_0 = VGGBlock(nb_filter[4], nb_filter[5], nb_filter[5])
        self.conv6_0 = VGGBlock(nb_filter[5], nb_filter[6], nb_filter[6])
        self.conv7_0 = VGGBlock(nb_filter[6], nb_filter[7], nb_filter[7])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv4_1 = VGGBlock(nb_filter[4]+nb_filter[5], nb_filter[4], nb_filter[4])
        self.conv5_1 = VGGBlock(nb_filter[5]+nb_filter[6], nb_filter[5], nb_filter[5])
        self.conv6_1 = VGGBlock(nb_filter[6]+nb_filter[7], nb_filter[6], nb_filter[6])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_2 = VGGBlock(nb_filter[3]*2+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv4_2 = VGGBlock(nb_filter[4]*2+nb_filter[5], nb_filter[4], nb_filter[4])
        self.conv5_2 = VGGBlock(nb_filter[5]*2+nb_filter[6], nb_filter[5], nb_filter[5])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_3 = VGGBlock(nb_filter[2]*3+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_3 = VGGBlock(nb_filter[3]*3+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv4_3 = VGGBlock(nb_filter[4]*3+nb_filter[5], nb_filter[4], nb_filter[4])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_4 = VGGBlock(nb_filter[1]*4+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_4 = VGGBlock(nb_filter[2]*4+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_4 = VGGBlock(nb_filter[3]*4+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_5 = VGGBlock(nb_filter[0]*5+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_5 = VGGBlock(nb_filter[1]*5+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_5 = VGGBlock(nb_filter[2]*5+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_6 = VGGBlock(nb_filter[0]*6+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_6 = VGGBlock(nb_filter[1]*6+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_7 = VGGBlock(nb_filter[0]*7+nb_filter[1], nb_filter[0], nb_filter[0])


        if self.opt.deepsupervision:
            self.final5 = nn.Conv2d(nb_filter[0], opt.output_nc, kernel_size=1)
            self.final6 = nn.Conv2d(nb_filter[0], opt.output_nc, kernel_size=1)
            self.final7 = nn.Conv2d(nb_filter[0], opt.output_nc, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], opt.output_nc, kernel_size=1)


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        x5_0 = self.conv5_0(self.pool(x4_0))
        x4_1 = self.conv4_1(torch.cat([x4_0, self.up(x5_0)], 1))
        x3_2 = self.conv3_2(torch.cat([x3_0, x3_1, self.up(x4_1)], 1))
        x2_3 = self.conv2_3(torch.cat([x2_0, x2_1, x2_2, self.up(x3_2)], 1))
        x1_4 = self.conv1_4(torch.cat([x1_0, x1_1, x1_2, x1_3, self.up(x2_3)], 1))
        x0_5 = self.conv0_5(torch.cat([x0_0, x0_1, x0_2, x0_3, x0_4, self.up(x1_4)], 1))

        x6_0 = self.conv6_0(self.pool(x5_0))
        x5_1 = self.conv5_1(torch.cat([x5_0, self.up(x6_0)], 1))
        x4_2 = self.conv4_2(torch.cat([x4_0, x4_1, self.up(x5_1)], 1))
        x3_3 = self.conv3_3(torch.cat([x3_0, x3_1, x3_2, self.up(x4_2)], 1))
        x2_4 = self.conv2_4(torch.cat([x2_0, x2_1, x2_2, x2_3, self.up(x3_3)], 1))
        x1_5 = self.conv1_5(torch.cat([x1_0, x1_1, x1_2, x1_3, x1_4, self.up(x2_4)], 1))
        x0_6 = self.conv0_6(torch.cat([x0_0, x0_1, x0_2, x0_3, x0_4, x0_5, self.up(x1_5)], 1))

        x7_0 = self.conv7_0(self.pool(x6_0))
        x6_1 = self.conv6_1(torch.cat([x6_0, self.up(x7_0)], 1))
        x5_2 = self.conv5_2(torch.cat([x5_0, x5_1, self.up(x6_1)], 1))
        x4_3 = self.conv4_3(torch.cat([x4_0, x4_1, x4_2, self.up(x5_2)], 1))
        x3_4 = self.conv3_4(torch.cat([x3_0, x3_1, x3_2, x3_3, self.up(x4_3)], 1))
        x2_5 = self.conv2_5(torch.cat([x2_0, x2_1, x2_2, x2_3, x2_4, self.up(x3_4)], 1))
        x1_6 = self.conv1_6(torch.cat([x1_0, x1_1, x1_2, x1_3, x1_4, x1_5, self.up(x2_5)], 1))
        x0_7 = self.conv0_7(torch.cat([x0_0, x0_1, x0_2, x0_3, x0_4, x0_5, x0_6, self.up(x1_6)], 1))

        if self.opt.deepsupervision:
            output5 = self.final5(x0_5)
            output6 = self.final6(x0_6)
            output7 = self.final7(x0_7)

            return [output5, output6, output7]

        else:
            output = self.final(x0_7)
            return x0_7
