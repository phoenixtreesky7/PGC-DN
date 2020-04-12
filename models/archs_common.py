# -*- coding: utf-8 -*-

import numpy as np



from torch import nn

from torch.nn import functional as F

import torch

from torchvision import models

import torchvision

import models.ca_common as ca_conv



from models.GCnet import ContextBlock2d as GCB

from models.GCnet import MS_ContextBlock2d as MSGCB

from models.GCnet import DIFMS_ContextBlock2d as DIFMSGCB

#from apnb.utils.apnb import APNB

##############################################################################

##  BACKBONE: VGG, RN_bottleneck_d2, RN_bottleneck_u2, RN_dilation_d2, RN_dilation_u2, RN_bottleneck

##############################################################################

class VGGBlock(nn.Module):

    def __init__(self, in_channel, middle_channel, out_channel, activation=nn.ReLU(inplace=True)):

        super(VGGBlock, self).__init__()

        self.activation = activation

        self.conv1 = nn.Conv2d(in_channel, middle_channel, 3, padding=1)

        self.bn1 = nn.InstanceNorm2d(middle_channel)

        self.conv2 = nn.Conv2d(middle_channel, out_channel, 3, padding=1)

        self.bn2 = nn.InstanceNorm2d(out_channel)



    def forward(self, x):

        out = self.conv1(x)

        out = self.bn1(out)

        out = self.activation(out)



        #out = self.conv2(out)

        #out = self.bn2(out)

        #out = self.activation(out)



        return out

###############################

##        Resnet block       ##

###############################

class RN_basic_d2(nn.Module):

    def __init__(self, in_channel, middle_channel, out_channel, activation=nn.ReLU(inplace=True)):

        super(RN_basic_d2, self).__init__()

        trans = [nn.InstanceNorm2d(in_channel), activation, nn.Conv2d(in_channel, middle_channel, kernel_size=3, stride=2, padding=1, bias=False)]



        bottleneck = [nn.InstanceNorm2d(middle_channel), activation, nn.Conv2d(middle_channel, middle_channel//4, kernel_size=1, stride=1, bias=False)]

        bottleneck += [nn.InstanceNorm2d(middle_channel//4), activation, nn.Conv2d(middle_channel//4, middle_channel//4, kernel_size=3, stride=1, padding=1, bias=False)]

        bottleneck += [nn.InstanceNorm2d(middle_channel//4), activation, nn.Conv2d(middle_channel//4, out_channel, kernel_size=1, stride=1, bias=False)]



        #self.relu = nn.ReLU(True)

        self.ex = nn.Sequential(*trans)

        self.dc = nn.Sequential(*bottleneck)



    def forward(self, x):

        out = self.ex(x)

        out = self.dc(out) + out



        return out





class RN_basic_u2(nn.Module):

    def __init__(self, in_channel, middle_channel, out_channel, activation=nn.ReLU(inplace=True)):

        super(RN_basic_u2, self).__init__()

        trans = [nn.InstanceNorm2d(in_channel), activation, nn.Conv2d(in_channel, middle_channel, kernel_size=1, stride=1, bias=False)]



        bottleneck = [nn.InstanceNorm2d(middle_channel), activation, nn.Conv2d(middle_channel, middle_channel//4, kernel_size=1, stride=1, bias=False)]

        bottleneck += [nn.InstanceNorm2d(middle_channel//4), activation, nn.Conv2d(middle_channel//4, middle_channel//4, kernel_size=3, stride=1, padding=1, bias=False)]    

        bottleneck += [nn.InstanceNorm2d(middle_channel//4), activation, nn.Conv2d(middle_channel//4, out_channel, kernel_size=1, stride=1, bias=False)]



        #self.relu = nn.ReLU(True)

        self.ex = nn.Sequential(*trans)

        self.dc = nn.Sequential(*bottleneck)



    def forward(self, x):

        out = self.ex(x)

        out = self.dc(out) + out



        return out



##### NO Dilation #####

class RN_bottleneck_d2(nn.Module):

    def __init__(self, in_channel, middle_channel, out_channel, scale, activation=nn.ReLU(inplace=True)):

        super(RN_bottleneck_d2, self).__init__()

        self.scale = scale



        trans = [nn.Conv2d(in_channel, middle_channel, kernel_size=3, stride=2, padding=1, bias=False)]

        self.gcb = GCB(middle_channel, middle_channel)

        bottleneck = [nn.InstanceNorm2d(middle_channel), nn.ReLU(True), nn.Conv2d(middle_channel, middle_channel//4, kernel_size=1, stride=1, bias=False)]

        bottleneck += [nn.InstanceNorm2d(middle_channel//4), nn.ReLU(True), nn.Conv2d(middle_channel//4, middle_channel//4, kernel_size=3, stride=1, padding=1, bias=False)]    

        bottleneck += [nn.InstanceNorm2d(out_channel//4), nn.ReLU(True), nn.Conv2d(middle_channel//4, out_channel, kernel_size=1, stride=1, bias=False)]



        #self.relu = nn.ReLU(True)

        self.ex = nn.Sequential(*trans)

        self.dc = nn.Sequential(*bottleneck)



    def forward(self, x):

        out = self.ex(x)

        out_gc = self.gcb(out)

        out = self.dc(out_gc) + out_gc



        return out





class RN_bottleneck_u2(nn.Module):

    def __init__(self, in_channel, middle_channel, out_channel, scale, activation=nn.ReLU(inplace=True)):

        super(RN_bottleneck_u2, self).__init__()

        trans = [nn.Conv2d(in_channel, middle_channel, kernel_size=1, stride=1, bias=False)]



        self.gcb = GCB(middle_channel, middle_channel)

        bottleneck = [nn.InstanceNorm2d(middle_channel), nn.ReLU(True), nn.Conv2d(middle_channel, middle_channel//4, kernel_size=1, stride=1, bias=False)]

        bottleneck += [nn.InstanceNorm2d(middle_channel//4), nn.ReLU(True), nn.Conv2d(middle_channel//4, middle_channel//4, kernel_size=3, stride=1, padding=1, bias=False)]    

        bottleneck += [nn.InstanceNorm2d(middle_channel//4), nn.ReLU(True), nn.Conv2d(middle_channel//4, out_channel, kernel_size=1, stride=1, bias=False)]



        #self.relu = nn.ReLU(True)

        self.ex = nn.Sequential(*trans)

        self.dc = nn.Sequential(*bottleneck)



    def forward(self, x):

        out = self.ex(x)

        out_gc = self.gcb(out)

        out = self.dc(out_gc) + out_gc



        return out

#####  NO GC  ######

class RN_dilation_d2(nn.Module):

    def __init__(self, in_channel, middle_channel, out_channel, activation=nn.ReLU(inplace=True)):

        super(RN_dilation_d2, self).__init__()

        trans = [nn.InstanceNorm2d(in_channel), activation, nn.Conv2d(in_channel, middle_channel, kernel_size=3, stride=2, padding=1, bias=False)]



        bottleneck = [nn.InstanceNorm2d(middle_channel), activation, nn.Conv2d(middle_channel, middle_channel//4, kernel_size=1, stride=1, bias=False)]

        bottleneck += [nn.InstanceNorm2d(middle_channel//4), activation, nn.Conv2d(middle_channel//4, middle_channel//4, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)]

        bottleneck += [nn.InstanceNorm2d(middle_channel//4), activation, nn.Conv2d(middle_channel//4, out_channel, kernel_size=1, stride=1, bias=False)]



        #self.relu = nn.ReLU(True)

        self.ex = nn.Sequential(*trans)

        self.dc = nn.Sequential(*bottleneck)



    def forward(self, x):

        out = self.ex(x)

        out = self.dc(out) + out



        return out





class RN_dilation_u2(nn.Module):

    def __init__(self, in_channel, middle_channel, out_channel, activation=nn.ReLU(inplace=True)):

        super(RN_dilation_u2, self).__init__()

        trans = [nn.InstanceNorm2d(in_channel), activation, nn.Conv2d(in_channel, middle_channel, kernel_size=1, stride=1, bias=False)]



        bottleneck = [nn.InstanceNorm2d(middle_channel), activation, nn.Conv2d(middle_channel, middle_channel//4, kernel_size=1, stride=1, bias=False)]

        bottleneck += [nn.InstanceNorm2d(middle_channel//4), activation, nn.Conv2d(middle_channel//4, middle_channel//4, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)]    

        bottleneck += [nn.InstanceNorm2d(middle_channel//4), activation, nn.Conv2d(middle_channel//4, out_channel, kernel_size=1, stride=1, bias=False)]



        #self.relu = nn.ReLU(True)

        self.ex = nn.Sequential(*trans)

        self.dc = nn.Sequential(*bottleneck)



    def forward(self, x):

        out = self.ex(x)

        out = self.dc(out) + out



        return out



######  GC in ResNet v1  #####

class RN_GCB_d2(nn.Module):

    def __init__(self, in_channel, middle_channel, out_channel, activation=nn.ReLU(inplace=True)):

        super(RN_GCB_d2, self).__init__()



        trans = [nn.Conv2d(in_channel, middle_channel, kernel_size=3, stride=2, padding=1, bias=False)]

        bottleneck = [GCB(middle_channel, middle_channel)]

        bottleneck += [nn.InstanceNorm2d(middle_channel), nn.ReLU(True), nn.Conv2d(middle_channel, middle_channel//4, kernel_size=1, stride=1, bias=False)]

        bottleneck += [nn.InstanceNorm2d(middle_channel//4), nn.ReLU(True), nn.Conv2d(middle_channel//4, middle_channel//4, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)]    

        bottleneck += [nn.InstanceNorm2d(out_channel//4), nn.ReLU(True), nn.Conv2d(middle_channel//4, out_channel, kernel_size=1, stride=1, bias=False)]



        self.relu = nn.ReLU(True)

        self.ex = nn.Sequential(*trans)

        self.dc = nn.Sequential(*bottleneck)



    def forward(self, x):

        out = self.ex(x)

        out = self.relu(self.dc(out) + out)



        return out





class RN_GCB_u2(nn.Module):

    def __init__(self, in_channel, middle_channel, out_channel, activation=nn.ReLU(inplace=True)):

        super(RN_GCB_u2, self).__init__()

        trans = [nn.Conv2d(in_channel, middle_channel, kernel_size=1, stride=1, bias=False)]



        bottleneck = [GCB(middle_channel, middle_channel)]

        bottleneck += [nn.InstanceNorm2d(middle_channel), nn.ReLU(True), nn.Conv2d(middle_channel, middle_channel//4, kernel_size=1, stride=1, bias=False)]

        bottleneck += [nn.InstanceNorm2d(middle_channel//4), nn.ReLU(True), nn.Conv2d(middle_channel//4, middle_channel//4, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)]    

        bottleneck += [nn.InstanceNorm2d(middle_channel//4), nn.ReLU(True), nn.Conv2d(middle_channel//4, out_channel, kernel_size=1, stride=1, bias=False)]



        self.relu = nn.ReLU(True)

        self.ex = nn.Sequential(*trans)

        self.dc = nn.Sequential(*bottleneck)



    def forward(self, x):

        out = self.ex(x)

        out = self.relu(self.dc(out) + out)



        return out



######  GC out ResNet v2  #####

class RN_GCB_d2_v2(nn.Module):

    def __init__(self, in_channel, middle_channel, out_channel, activation=nn.ReLU(inplace=True)):

        super(RN_GCB_d2_v2, self).__init__()



        trans = [nn.Conv2d(in_channel, middle_channel, kernel_size=3, stride=2, padding=1, bias=False)]

        self.gcb = GCB(middle_channel, middle_channel)

        bottleneck = [nn.InstanceNorm2d(middle_channel), nn.ReLU(True), nn.Conv2d(middle_channel, middle_channel//4, kernel_size=1, stride=1, bias=False)]

        bottleneck += [nn.InstanceNorm2d(middle_channel//4), nn.ReLU(True), nn.Conv2d(middle_channel//4, middle_channel//4, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)]    

        bottleneck += [nn.InstanceNorm2d(out_channel//4), nn.ReLU(True), nn.Conv2d(middle_channel//4, out_channel, kernel_size=1, stride=1, bias=False)]



        #self.relu = nn.ReLU(True)

        self.ex = nn.Sequential(*trans)

        self.dc = nn.Sequential(*bottleneck)



    def forward(self, x):

        out = self.ex(x)

        out_gc = self.gcb(out)

        out = self.dc(out_gc) + out_gc



        return out





class RN_GCB_u2_v2(nn.Module):

    def __init__(self, in_channel, middle_channel, out_channel, activation=nn.ReLU(inplace=True)):

        super(RN_GCB_u2_v2, self).__init__()

        trans = [nn.Conv2d(in_channel, middle_channel, kernel_size=1, stride=1, bias=False)]



        self.gcb = GCB(middle_channel, middle_channel)

        bottleneck = [nn.InstanceNorm2d(middle_channel), nn.ReLU(True), nn.Conv2d(middle_channel, middle_channel//4, kernel_size=1, stride=1, bias=False)]

        bottleneck += [nn.InstanceNorm2d(middle_channel//4), nn.ReLU(True), nn.Conv2d(middle_channel//4, middle_channel//4, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)]    

        bottleneck += [nn.InstanceNorm2d(middle_channel//4), nn.ReLU(True), nn.Conv2d(middle_channel//4, out_channel, kernel_size=1, stride=1, bias=False)]



        #self.relu = nn.ReLU(True)

        self.ex = nn.Sequential(*trans)

        self.dc = nn.Sequential(*bottleneck)



    def forward(self, x):

        out = self.ex(x)

        out_gc = self.gcb(out)

        out = self.dc(out_gc) + out_gc



        return out



#####  MSGC out ResNet v2  #####

class RN_MSGCB_d2_v2(nn.Module):

    def __init__(self, in_channel, middle_channel, out_channel, activation=nn.ReLU(inplace=True)):

        super(RN_MSGCB_d2_v2, self).__init__()



        trans = [nn.Conv2d(in_channel, middle_channel, kernel_size=3, stride=2, padding=1, bias=False)]

        self.gcb = MSGCB(middle_channel, middle_channel)

        bottleneck = [nn.InstanceNorm2d(middle_channel), nn.ReLU(True), nn.Conv2d(middle_channel, middle_channel//4, kernel_size=1, stride=1, bias=False),

                  nn.InstanceNorm2d(middle_channel//4)]

        bottleneck += [nn.InstanceNorm2d(middle_channel//4), nn.ReLU(True), nn.Conv2d(middle_channel//4, middle_channel//4, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)]    

        bottleneck += [nn.InstanceNorm2d(out_channel//4), nn.ReLU(True), nn.Conv2d(middle_channel//4, out_channel, kernel_size=1, stride=1, bias=False)]



        #self.relu = nn.ReLU(True)

        self.ex = nn.Sequential(*trans)

        self.dc = nn.Sequential(*bottleneck)



    def forward(self, x):

        out = self.ex(x)

        out_gc = self.gcb(out)

        out = self.dc(out_gc) + out_gc



        return out



class RN_MSGCB_u2_v2(nn.Module):

    def __init__(self, in_channel, middle_channel, out_channel, activation=nn.ReLU(inplace=True)):

        super(RN_MSGCB_u2_v2, self).__init__()

        trans = [nn.Conv2d(in_channel, middle_channel, kernel_size=1, stride=1, bias=False)]



        self.gcb = MSGCB(middle_channel, middle_channel)

        bottleneck = [nn.InstanceNorm2d(middle_channel), nn.ReLU(True), nn.Conv2d(middle_channel, middle_channel//4, kernel_size=1, stride=1, bias=False),

                  nn.InstanceNorm2d(middle_channel//4)]

        bottleneck += [nn.InstanceNorm2d(middle_channel//4), nn.ReLU(True), nn.Conv2d(middle_channel//4, middle_channel//4, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)]    

        bottleneck += [nn.InstanceNorm2d(middle_channel//4), nn.ReLU(True), nn.Conv2d(middle_channel//4, out_channel, kernel_size=1, stride=1, bias=False)]



        #self.relu = nn.ReLU(True)

        self.ex = nn.Sequential(*trans)

        self.dc = nn.Sequential(*bottleneck)



    def forward(self, x):

        out = self.ex(x)

        out_gc = self.gcb(out)

        out = self.dc(out_gc) + out_gc



        return out



#####  DIFMSGC out ResNet v2  #####

class RN_DIFMSGCB_d2_v2(nn.Module):

    def __init__(self, in_channel, middle_channel, out_channel, scale, activation=nn.ReLU(inplace=True)):

        super(RN_DIFMSGCB_d2_v2, self).__init__()

        self.scale = scale



        trans = [nn.Conv2d(in_channel, middle_channel, kernel_size=3, stride=2, padding=1, bias=False)]

        self.gcb = DIFMSGCB(middle_channel, middle_channel, scale)

        bottleneck = [nn.InstanceNorm2d(middle_channel), nn.ReLU(True), nn.Conv2d(middle_channel, middle_channel//4, kernel_size=1, stride=1, bias=False)]

        bottleneck += [nn.InstanceNorm2d(middle_channel//4), nn.ReLU(True), nn.Conv2d(middle_channel//4, middle_channel//4, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)]    

        bottleneck += [nn.InstanceNorm2d(out_channel//4), nn.ReLU(True), nn.Conv2d(middle_channel//4, out_channel, kernel_size=1, stride=1, bias=False)]



        #self.relu = nn.ReLU(True)

        self.ex = nn.Sequential(*trans)

        self.dc = nn.Sequential(*bottleneck)



    def forward(self, x):

        out = self.ex(x)

        out_gc = self.gcb(out)

        out = self.dc(out_gc) + out_gc



        return out



class RN_DIFMSGCB_u2_v2(nn.Module):

    def __init__(self, in_channel, middle_channel, out_channel, scale, activation=nn.ReLU(inplace=True)):

        super(RN_DIFMSGCB_u2_v2, self).__init__()

        trans = [nn.Conv2d(in_channel, middle_channel, kernel_size=1, stride=1, bias=False)]



        self.gcb = DIFMSGCB(middle_channel, middle_channel, scale)

        bottleneck = [nn.InstanceNorm2d(middle_channel), nn.ReLU(True), nn.Conv2d(middle_channel, middle_channel//4, kernel_size=1, stride=1, bias=False)]

        bottleneck += [nn.InstanceNorm2d(middle_channel//4), nn.ReLU(True), nn.Conv2d(middle_channel//4, middle_channel//4, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)]    

        bottleneck += [nn.InstanceNorm2d(middle_channel//4), nn.ReLU(True), nn.Conv2d(middle_channel//4, out_channel, kernel_size=1, stride=1, bias=False)]



        #self.relu = nn.ReLU(True)

        self.ex = nn.Sequential(*trans)

        self.dc = nn.Sequential(*bottleneck)



    def forward(self, x):

        out = self.ex(x)

        out_gc = self.gcb(out)

        out = self.dc(out_gc) + out_gc



        return out




#######  ResNet block in ResNet Group ######

class RN_bottleneck(nn.Module):

    def __init__(self, input_channel, middle_channel, output_channel, norm_layer=nn.InstanceNorm2d, activation=nn.ReLU(True), use_dropout=False):

        super(RN_bottleneck, self).__init__()

        #self.conv1x1 = nn.Conv2d(input_channel, output_channel, 1)

        self.conv_block = self.build_conv_block(input_channel, middle_channel, output_channel, norm_layer, activation, use_dropout)

        self.relu = activation



    def build_conv_block(self, input_channel, middle_channel, output_channel, norm_layer, activation, use_dropout):

        conv_block = []

        conv_block += [norm_layer(input_channel), activation, nn.Conv2d(input_channel, middle_channel//4, kernel_size=1, padding=0)]

        conv_block += [norm_layer(middle_channel//4), activation, nn.Conv2d(middle_channel//4, middle_channel//4, kernel_size=3, padding=1)]

        conv_block += [norm_layer(middle_channel//4), activation, nn.Conv2d(middle_channel//4, output_channel, kernel_size=1, padding=0)]



        return nn.Sequential(*conv_block)



    def forward(self, x):

        out = self.conv_block(x)

        #x = self.conv1x1(x)

        out = out + x

        return out






###############################################

###############################################

##                                           ##

##           Dilated Residual U-Net          ##

##                                           ##

###############################################

###############################################

#####  No GC block in U-net  #####

class UNet3_Concate_Sym(nn.Module):

    def __init__(self, opt):

        super(UNet3_Concate_Sym,self).__init__()



        self.opt = opt



        nb_filter = [ 64, 128, 256, 512, 512, 512, 512]



        self.conv1_0 = RN_dilation_d2(nb_filter[0], nb_filter[1], nb_filter[1])

        self.conv2_0 = RN_dilation_d2(nb_filter[1], nb_filter[2], nb_filter[2])

        self.conv3_0 = RN_dilation_d2(nb_filter[2], nb_filter[3], nb_filter[3])



        ResAttentionNet_G = []

        for i in range(4):

                ResAttentionNet_G += [RN_bottleneck(nb_filter[3], nb_filter[3], nb_filter[3], norm_layer=nn.InstanceNorm2d)]



        self.trans = nn.Sequential(*ResAttentionNet_G)



        self.conv2_1 = RN_dilation_u2(nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv1_2 = RN_dilation_u2(nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_3 = RN_dilation_u2(nb_filter[1], nb_filter[0], nb_filter[0])





        model_upsample3 = [nn.InstanceNorm2d(nb_filter[3]), nn.ReLU(True), nn.ConvTranspose2d(nb_filter[3] * 2, nb_filter[3], kernel_size=3, stride=2, padding=1, output_padding=1), nn.InstanceNorm2d(nb_filter[3]), nn.ReLU(True)]

        self.up3 = nn.Sequential(*model_upsample3)



        model_upsample2 = [nn.ConvTranspose2d(nb_filter[2] * 2, nb_filter[2], kernel_size=3, stride=2, padding=1, output_padding=1), nn.InstanceNorm2d(nb_filter[2]), nn.ReLU(True)]

        self.up2 = nn.Sequential(*model_upsample2)



        model_upsample1 = [nn.ConvTranspose2d(nb_filter[1] * 2, nb_filter[1], kernel_size=3, stride=2, padding=1, output_padding=1), nn.InstanceNorm2d(nb_filter[1]), nn.ReLU(True)]

        self.up1 = nn.Sequential(*model_upsample1) 



        self.final = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1)

        self.relu = nn.ReLU(True)





    def forward(self, input):



        x1_0 = self.conv1_0(input)  # 128 128

        #xa1_0 = self.attention_1(x1_0)

        x2_0 = self.conv2_0(x1_0)  # 64 256

        #xa2_0 = self.attention_2(x2_0)

        x3_0 = self.conv3_0(x2_0)  # 32 512

        #xa3_0 = self.attention_3(x3_0)



        x3_0t = self.relu(self.trans(x3_0) + x3_0)

        x3_0t = self.relu(self.trans(x3_0t) + x3_0t)  # 32 512



        x2_1 = self.conv2_1(self.up3(torch.cat([x3_0t, x3_0], 1)))  # 64 256

        x1_2 = self.conv1_2(self.up2(torch.cat([x2_0, x2_1], 1)))  # 128 128

        x0_3 = self.conv0_3(self.up1(torch.cat([x1_0, x1_2], 1)))  # 256 64

        #x0_4 = self.final(torch.cat([xa0_0, x0_3], 1)) 



        #output = self.final(x0_4)

        return x0_3




######  MSGC block  #####

class MSGCUNet3_Concate_Sym(nn.Module):

    def __init__(self, opt):

        super(MSGCUNet3_Concate_Sym,self).__init__()



        self.opt = opt



        nb_filter = [64, 128, 256, 512, 512, 512, 512]



        self.conv1_0 = RN_MSGCB_d2_v2(nb_filter[0], nb_filter[1], nb_filter[1])

        self.conv2_0 = RN_MSGCB_d2_v2(nb_filter[1], nb_filter[2], nb_filter[2])

        self.conv3_0 = RN_MSGCB_d2_v2(nb_filter[2], nb_filter[3], nb_filter[3])



        ResAttentionNet_G = []

        for i in range(4):

                ResAttentionNet_G += [RN_bottleneck(nb_filter[3], nb_filter[3], nb_filter[3], norm_layer=nn.InstanceNorm2d)]



        self.trans = nn.Sequential(*ResAttentionNet_G)



        self.conv2_1 = RN_MSGCB_u2_v2(nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv1_2 = RN_MSGCB_u2_v2(nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_3 = RN_MSGCB_u2_v2(nb_filter[1], nb_filter[0], nb_filter[0])





        model_upsample3 = [nn.ConvTranspose2d(nb_filter[3] * 2, nb_filter[3], kernel_size=3, stride=2, padding=1, output_padding=1), nn.InstanceNorm2d(nb_filter[3]), nn.ReLU(True)]

        self.up3 = nn.Sequential(*model_upsample3)

        model_upsample2 = [nn.ConvTranspose2d(nb_filter[2] * 2, nb_filter[2], kernel_size=3, stride=2, padding=1, output_padding=1), nn.InstanceNorm2d(nb_filter[2]), nn.ReLU(True)]

        self.up2 = nn.Sequential(*model_upsample2)

        model_upsample1 = [nn.ConvTranspose2d(nb_filter[1] * 2, nb_filter[1], kernel_size=3, stride=2, padding=1, output_padding=1), nn.InstanceNorm2d(nb_filter[1]), nn.ReLU(True)]

        self.up1 = nn.Sequential(*model_upsample1)



        #self.final = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1)

        self.relu = nn.ReLU(True)

        self.norm = nn.InstanceNorm2d(nb_filter[0])







    def forward(self, input):



        x1_0 = self.conv1_0(input)  # 128 128

        #xa1_0 = self.attention_1(x1_0)

        x2_0 = self.conv2_0(x1_0)  # 64 256

        #xa2_0 = self.attention_2(x2_0)

        x3_0 = self.conv3_0(x2_0)  # 32 512

        #xa3_0 = self.attention_3(x3_0)



        x3_0t = self.relu(self.trans(x3_0) + x3_0)

        x3_0t = self.relu(self.trans(x3_0t) + x3_0t)  # 32 512



        x2_1 = self.conv2_1(self.up3(torch.cat([x3_0t, x3_0], 1)))  # 64 256

        x1_2 = self.conv1_2(self.up2(torch.cat([x2_0, x2_1], 1)))  # 128 128

        x0_3 = self.conv0_3(self.up1(torch.cat([x1_0, x1_2], 1)))  # 256 64

        x0_3 = self.relu(self.norm(x0_3))

        #x0_4 = self.final(torch.cat([xa0_0, x0_3], 1)) 



        #output = self.final(x0_4)

        return x0_3









######  GC block out of ResNet v2  #####

class GCUNet3_Concate_Sym_outRNv2(nn.Module):

    def __init__(self, opt):

        super(GCUNet3_Concate_Sym_outRNv2,self).__init__()



        self.opt = opt



        nb_filter = [64, 128, 256, 512, 512, 512, 512]



        self.conv1_0 = RN_GCB_d2_v2(nb_filter[0], nb_filter[1], nb_filter[1])

        self.conv2_0 = RN_GCB_d2_v2(nb_filter[1], nb_filter[2], nb_filter[2])

        self.conv3_0 = RN_GCB_d2_v2(nb_filter[2], nb_filter[3], nb_filter[3])



        ResAttentionNet_G = []

        for i in range(4):

                ResAttentionNet_G += [RN_bottleneck(nb_filter[3], nb_filter[3], nb_filter[3], norm_layer=nn.InstanceNorm2d)]



        self.trans = nn.Sequential(*ResAttentionNet_G)



        self.conv2_1 = RN_GCB_u2_v2(nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv1_2 = RN_GCB_u2_v2(nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_3 = RN_GCB_u2_v2(nb_filter[1], nb_filter[0], nb_filter[0])





        model_upsample3 = [nn.ConvTranspose2d(nb_filter[3] * 2, nb_filter[3], kernel_size=3, stride=2, padding=1, output_padding=1), nn.InstanceNorm2d(nb_filter[3]), nn.ReLU(True)]

        self.up3 = nn.Sequential(*model_upsample3)

        model_upsample2 = [nn.ConvTranspose2d(nb_filter[2] * 2, nb_filter[2], kernel_size=3, stride=2, padding=1, output_padding=1), nn.InstanceNorm2d(nb_filter[2]), nn.ReLU(True)]

        self.up2 = nn.Sequential(*model_upsample2)

        model_upsample1 = [nn.ConvTranspose2d(nb_filter[1] * 2, nb_filter[1], kernel_size=3, stride=2, padding=1, output_padding=1), nn.InstanceNorm2d(nb_filter[1]), nn.ReLU(True)]

        self.up1 = nn.Sequential(*model_upsample1) 



        self.relu = nn.ReLU(True)

        self.norm = nn.InstanceNorm2d(nb_filter[0])



    def forward(self, input):



        x1_0 = self.conv1_0(input)  # 128 128

        #xa1_0 = self.attention_1(x1_0)

        x2_0 = self.conv2_0(x1_0)  # 64 256

        #xa2_0 = self.attention_2(x2_0)

        x3_0 = self.conv3_0(x2_0)  # 32 512

        #xa3_0 = self.attention_3(x3_0)



        x3_0t = self.relu(self.trans(x3_0) + x3_0)

        x3_0t = self.relu(self.trans(x3_0t) + x3_0t)  # 32 512



        x2_1 = self.conv2_1(self.up3(torch.cat([x3_0t, x3_0], 1)))  # 64 256

        x1_2 = self.conv1_2(self.up2(torch.cat([x2_0, x2_1], 1)))  # 128 128

        x0_3 = self.conv0_3(self.up1(torch.cat([x1_0, x1_2], 1)))  # 256 64

        x0_3 = self.relu(self.norm(x0_3))

        #x0_4 = self.final(torch.cat([xa0_0, x0_3], 1)) 



        #output = self.final(x0_4)

        return x0_3









class DIFMSGCUNet3_Concate_Sym(nn.Module):

    def __init__(self, opt):

        super(DIFMSGCUNet3_Concate_Sym,self).__init__()



        self.opt = opt



        nb_filter = [64, 128, 256, 512, 512, 512, 512]



        self.conv1_0 = RN_DIFMSGCB_d2_v2(nb_filter[0], nb_filter[1], nb_filter[1], scale=2)

        self.conv2_0 = RN_DIFMSGCB_d2_v2(nb_filter[1], nb_filter[2], nb_filter[2], scale=1)

        self.conv3_0 = RN_DIFMSGCB_d2_v2(nb_filter[2], nb_filter[3], nb_filter[3], scale=0)



        ResAttentionNet_G = []

        for i in range(4):

                ResAttentionNet_G += [RN_bottleneck(nb_filter[3], nb_filter[3], nb_filter[3], norm_layer=nn.InstanceNorm2d)]



        self.trans = nn.Sequential(*ResAttentionNet_G)



        self.conv2_1 = RN_DIFMSGCB_u2_v2(nb_filter[3], nb_filter[2], nb_filter[2], scale=0)

        self.conv1_2 = RN_DIFMSGCB_u2_v2(nb_filter[2], nb_filter[1], nb_filter[1], scale=1)

        self.conv0_3 = RN_DIFMSGCB_u2_v2(nb_filter[1], nb_filter[0], nb_filter[0], scale=2)





        model_upsample3 = [nn.ConvTranspose2d(nb_filter[3] * 2, nb_filter[3], kernel_size=3, stride=2, padding=1, output_padding=1), nn.InstanceNorm2d(nb_filter[3]), nn.ReLU(True)]

        self.up3 = nn.Sequential(*model_upsample3)

        model_upsample2 = [nn.ConvTranspose2d(nb_filter[2] * 2, nb_filter[2], kernel_size=3, stride=2, padding=1, output_padding=1), nn.InstanceNorm2d(nb_filter[2]), nn.ReLU(True)]

        self.up2 = nn.Sequential(*model_upsample2)

        model_upsample1 = [nn.ConvTranspose2d(nb_filter[1] * 2, nb_filter[1], kernel_size=3, stride=2, padding=1, output_padding=1), nn.InstanceNorm2d(nb_filter[1]), nn.ReLU(True)]

        self.up1 = nn.Sequential(*model_upsample1)



        #self.final = nn.Conv2d(nb_filter[1], nb_filter[0], kernel_size=1)

        self.relu = nn.ReLU(True)

        self.norm = nn.InstanceNorm2d(nb_filter[0])







    def forward(self, input):



        x1_0 = self.conv1_0(input)  # 128 128

        #xa1_0 = self.attention_1(x1_0)

        x2_0 = self.conv2_0(x1_0)  # 64 256

        #xa2_0 = self.attention_2(x2_0)

        x3_0 = self.conv3_0(x2_0)  # 32 512

        #xa3_0 = self.attention_3(x3_0)



        x3_0t = self.relu(self.trans(x3_0) + x3_0)

        x3_0t = self.relu(self.trans(x3_0t) + x3_0t)  # 32 512



        x2_1 = self.conv2_1(self.up3(torch.cat([x3_0t, x3_0], 1)))  # 64 256

        x1_2 = self.conv1_2(self.up2(torch.cat([x2_0, x2_1], 1)))  # 128 128

        x0_3 = self.conv0_3(self.up1(torch.cat([x1_0, x1_2], 1)))  # 256 64

        x0_3 = self.relu(self.norm(x0_3))

        #x0_4 = self.final(torch.cat([xa0_0, x0_3], 1)) 



        #output = self.final(x0_4)

        return x0_3





