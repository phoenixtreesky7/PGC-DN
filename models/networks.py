### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 

### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

import torch

import torch.nn as nn

import functools

from torch.autograd import Variable

import numpy as np

#from models import common

import torch.nn.functional as F

from functools import reduce

from options.train_options import TrainOptions

#from .archs import NestedUNet as UNet_plus

import math

from models import archs_common as archs


#from gcnet import GCnet



from gcnet.GCnet import ContextBlock2d as GCB

from gcnet.GCnet import MS_ContextBlock2d as MSGCB

from gcnet.GCnet import WMS_ContextBlock2d as WMSGCB

from gcnet.GCnet import DIFMS_ContextBlock2d as DIFMSGCB

opt = TrainOptions().parse()



###############################################################################

# Functions

###############################################################################

def weights_init(m):

    classname = m.__class__.__name__

    if classname.find('Conv') != -1:

        m.weight.data.normal_(0.0, 0.02)

    elif classname.find('BatchNorm2d') != -1:

        m.weight.data.normal_(1.0, 0.02)

        m.bias.data.fill_(0)



def get_norm_layer(norm_type='instance'):

    if norm_type == 'batch':

        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)

    elif norm_type == 'instance':

        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)

    else:

        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)

    return norm_layer



def define_G(input_nc, output_nc, sub_model, full_attention, ngf, netG, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1, 

             n_resblocks=4, norm='instance', gpu_ids=[]):    

    norm_layer = get_norm_layer(norm_type=norm)     

    if netG == 'global':    

        netG = GlobalGenerator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)   



    
    elif netG == 'gc_drun_pl': 

        netG = GC_DRUN_PL(input_nc, output_nc, sub_model, full_attention, ngf, 

            n_downsample_global, n_blocks_global, n_local_enhancers, n_resblocks, norm_layer)

    elif netG == 'msgc_drun_pl': 

        netG = MSGC_DRUN_PL(input_nc, output_nc, sub_model, full_attention, ngf, 

            n_downsample_global, n_blocks_global, n_local_enhancers, n_resblocks, norm_layer)

    elif netG == 'difmsgc_drun_pl': 

        netG = DIFMSGC_DRUN_PL(input_nc, output_nc, sub_model, full_attention, ngf, 

            n_downsample_global, n_blocks_global, n_local_enhancers, n_resblocks, norm_layer)

    elif netG == 'onemsgc_drun_pl': 

        netG = ONEMSGC_DRUN_PL(input_nc, output_nc, sub_model, full_attention, ngf,n_downsample_global, n_blocks_global, n_local_enhancers, n_resblocks, norm_layer)
    else: 

        raise('generator not implemented!')

    #print(netG)

    if len(gpu_ids) > 0:

        assert(torch.cuda.is_available())   

        netG.cuda(gpu_ids[0])

    netG.apply(weights_init)

    return netG



def define_D(input_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1, getIntermFeat=False, gpu_ids=[]):        

    norm_layer = get_norm_layer(norm_type=norm)   

    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat)   

    #print(netD)

    if len(gpu_ids) > 0:

        assert(torch.cuda.is_available())

        netD.cuda(gpu_ids[0])

    netD.apply(weights_init)

    return netD



def print_network(net):

    if isinstance(net, list):

        net = net[0]

    num_params = 0

    for param in net.parameters():

        num_params += param.numel()

    print(net)

    print('Total number of parameters: %d' % num_params)



#########################################################################

# Losses

#########################################################################

class GANLoss(nn.Module):

    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,

                 tensor=torch.FloatTensor):

        super(GANLoss, self).__init__()

        self.real_label = target_real_label

        self.fake_label = target_fake_label

        self.real_label_var = None

        self.fake_label_var = None

        self.Tensor = tensor

        if use_lsgan:

            self.loss = nn.MSELoss()

        else:

            self.loss = nn.BCELoss()



    def get_target_tensor(self, input, target_is_real):

        target_tensor = None

        if target_is_real:

            create_label = ((self.real_label_var is None) or

                            (self.real_label_var.numel() != input.numel()))

            if create_label:

                real_tensor = self.Tensor(input.size()).fill_(self.real_label)

                self.real_label_var = Variable(real_tensor, requires_grad=False)

            target_tensor = self.real_label_var

        else:

            create_label = ((self.fake_label_var is None) or

                            (self.fake_label_var.numel() != input.numel()))

            if create_label:

                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)

                self.fake_label_var = Variable(fake_tensor, requires_grad=False)

            target_tensor = self.fake_label_var

        return target_tensor



    def __call__(self, input, target_is_real):

        if isinstance(input[0], list):

            loss = 0

            for input_i in input:

                pred = input_i[-1]

                target_tensor = self.get_target_tensor(pred, target_is_real)

                loss += self.loss(pred, target_tensor)

            return loss

        else:            

            target_tensor = self.get_target_tensor(input[-1], target_is_real)

            return self.loss(input[-1], target_tensor)



class VGGLoss(nn.Module):

    def __init__(self, gpu_ids):

        super(VGGLoss, self).__init__()        

        self.vgg = Vgg19().cuda()

        self.criterion = nn.L1Loss()

        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        



    def forward(self, x, y):              

        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        loss = 0

        for i in range(len(x_vgg)):

            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        

        return loss





########################################################################

# Generator

########################################################################

####################
###    GC-DRU    ###
####################
### GC block before ResNet v2; dilated ResNet U-net + pyramid pooling

class GC_DRUN_PL(nn.Module): #32f

    def __init__(self, input_nc, output_nc, sub_model, full_attention, ngf=32, n_downsample_global=3, n_blocks_global=8, 

                 n_local_enhancers=1, n_resblocks=8, norm_layer=nn.InstanceNorm2d, padding_type='reflect'):

        super(GC_DRUN_PL, self).__init__()

        self.n_local_enhancers = n_local_enhancers

        self.sub_model = sub_model

        

        ###### global generator model #####           

        ngf_global = ngf * (2**(n_local_enhancers-1))

        print('ngf_global= %d' % (ngf_global))



        ###### Low-level Feature Extractors #####

        model_downsample1 = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0), norm_layer(ngf_global), nn.ReLU(True)]

        model_downsample2 = [nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1), norm_layer(ngf_global * 2), nn.ReLU(True)]



        self.model_ds1 = nn.Sequential(*model_downsample1)

        self.model_ds2 = nn.Sequential(*model_downsample2)



        ###### Non-local attention #####

        model_non_local_attention = [GCB(ngf * 2, ngf * 2)]

        model_non_local_attention += [ResnetBlock(ngf_global * 2, ngf_global * 2, ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

        #model_non_local_attention += [ResnetBlock(ngf * 2, ngf * 2, ngf * 2, padding_type=padding_type, norm_layer=norm_layer)]

        self.model_gcb = nn.Sequential(*model_non_local_attention)





        ##### Attention U-Net #####

        #rdaunet = [ResnetBlock(ngf_global * 2, ngf_global * 2, ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

        if sub_model == 'attention_unet3':

            rdaunet = [archs.GCUNet3_Concate_Sym_outRNv2(opt)]



        else:

            raise('sub_model not implemented!') 



        #rdaunet += [ResnetBlock(ngf_global * 2, ngf_global * 2, ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]



        self.model = nn.Sequential(*rdaunet)



        ###### Upsampling #####

        model_upsample = [nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1), norm_layer(ngf_global), nn.ReLU(True)]

        self.model_us = nn.Sequential(*model_upsample)



        model_final = [nn.ReflectionPad2d(3), nn.Conv2d(ngf_global * 2, ngf_global, kernel_size=7, padding=0)]

        self.model_f = nn.Sequential(*model_final)



        self.model_dh1 = Dehaze1()



    def forward(self, input):

        output_ds1 = self.model_ds1(input)

        output_gcb = self.model_ds2(output_ds1)

        output_gcb = self.model_gcb(output_gcb)

        output_gcb = self.model(output_gcb)

        output_gcb = self.model_gcb(output_gcb)

        output_us = self.model_us(output_gcb)

        tmp = torch.cat((output_us, output_ds1), 1)

        output_f = self.model_f(tmp)



        tmp = torch.cat((output_f, input), 1)

        output_dh = self.model_dh1(tmp)

        tmp = torch.cat((output_f, output_dh), 1)

        output_sub_us = self.model_dh1(tmp)

        return output_dh, output_dh, output_sub_us






###############################################
#####  PGC-DRU (PGC block in every stage) #####
###############################################
## different scales MSGC block + Dilated ResNet U-net + pyramid pooling

class DIFMSGC_DRUN_PL(nn.Module): #32f

    def __init__(self, input_nc, output_nc, sub_model, full_attention, ngf=32, n_downsample_global=3, n_blocks_global=8, 

                 n_local_enhancers=1, n_resblocks=8, norm_layer=nn.InstanceNorm2d, padding_type='reflect'):

        super(DIFMSGC_DRUN_PL, self).__init__()

        self.n_local_enhancers = n_local_enhancers

        self.sub_model = sub_model

        

        ###### global generator model #####           

        ngf_global = ngf * (2**(n_local_enhancers-1))

        print('ngf_global= %d' % (ngf_global))



        ###### Low-level Feature Extractors #####

        model_downsample1 = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0), norm_layer(ngf_global), nn.ReLU(True)]

        model_downsample2 = [nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1), norm_layer(ngf_global * 2), nn.ReLU(True)]



        self.model_ds1 = nn.Sequential(*model_downsample1)

        self.model_ds2 = nn.Sequential(*model_downsample2)



        ##### Attention U-Net #####

        #rdaunet = [ResnetBlock(ngf_global * 2, ngf_global * 2, ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

        if sub_model == 'attention_unet3':

            rdaunet = [archs.DIFMSGCUNet3_Concate_Sym(opt)]



        else:

            raise('sub_model not implemented!') 



        #rdaunet += [ResnetBlock(ngf_global * 2, ngf_global * 2, ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]



        self.model = nn.Sequential(*rdaunet)



        ###### Non-local attention #####

        model_non_local_attention = [MSGCB(ngf * 2, ngf * 2)]

        model_non_local_attention += [ResnetBlock(ngf_global * 2, ngf_global * 2, ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

        #model_non_local_attention += [ResnetBlock(ngf * 2, ngf * 2, ngf * 2, padding_type=padding_type, norm_layer=norm_layer)]

        self.model_msgcba = nn.Sequential(*model_non_local_attention)



        ###### Upsampling #####

        model_upsample = [nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1), norm_layer(ngf_global), nn.ReLU(True)]

        self.model_us = nn.Sequential(*model_upsample)



        model_final = [nn.ReflectionPad2d(3), nn.Conv2d(ngf_global * 2, ngf_global, kernel_size=7, padding=0)]

        self.model_f = nn.Sequential(*model_final)



        self.model_dh1 = Dehaze1()



    def forward(self, input):

        output_ds1 = self.model_ds1(input)

        output_msgcba = self.model_ds2(output_ds1)

        output_msgcba = self.model_msgcba(output_msgcba)

        output_msgcba = self.model(output_msgcba)

        output_msgcba = self.model_msgcba(output_msgcba)

        output_us = self.model_us(output_msgcba)

        tmp = torch.cat((output_us, output_ds1), 1)

        output_f = self.model_f(tmp)



        tmp = torch.cat((output_f, input), 1)

        output_dh = self.model_dh1(tmp)

        tmp = torch.cat((output_f, output_dh), 1)

        output_sub_us = self.model_dh1(tmp)

        return output_dh, output_dh, output_sub_us


#################################################################
#####  PGC-DRU (PGC blocks only in the first stage (paper)) #####
#################################################################

## different scales MSGC block + Dilated ResNet U-net + pyramid pooling

class ONEMSGC_DRUN_PL(nn.Module): #32f

    def __init__(self, input_nc, output_nc, sub_model, full_attention, ngf=32, n_downsample_global=3, n_blocks_global=8, 

                 n_local_enhancers=1, n_resblocks=8, norm_layer=nn.InstanceNorm2d, padding_type='reflect'):

        super(ONEMSGC_DRUN_PL, self).__init__()

        self.n_local_enhancers = n_local_enhancers

        self.sub_model = sub_model

        

        ###### global generator model #####           

        ngf_global = ngf * (2**(n_local_enhancers-1))

        print('ngf_global= %d' % (ngf_global))



        ###### Low-level Feature Extractors #####

        model_downsample1 = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0), norm_layer(ngf_global), nn.ReLU(True)]

        model_downsample2 = [nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1), norm_layer(ngf_global * 2), nn.ReLU(True)]



        self.model_ds1 = nn.Sequential(*model_downsample1)

        self.model_ds2 = nn.Sequential(*model_downsample2)



        ##### Attention U-Net #####

        #rdaunet = [ResnetBlock(ngf_global * 2, ngf_global * 2, ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

        if sub_model == 'attention_unet3':

            rdaunet = [archs.GCUNet3_Concate_Sym_outRNv2(opt)]



        else:

            raise('sub_model not implemented!') 



        #rdaunet += [ResnetBlock(ngf_global * 2, ngf_global * 2, ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]



        self.model = nn.Sequential(*rdaunet)



        ###### Non-local attention #####

        model_non_local_attention = [WMSGCB(ngf * 2, ngf * 2, 3, opt.gc_w)]

        model_non_local_attention += [ResnetBlock(ngf_global * 2, ngf_global * 2, ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

        #model_non_local_attention += [ResnetBlock(ngf * 2, ngf * 2, ngf * 2, padding_type=padding_type, norm_layer=norm_layer)]

        self.model_msgcba = nn.Sequential(*model_non_local_attention)



        ###### Upsampling #####

        model_upsample = [nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1), norm_layer(ngf_global), nn.ReLU(True)]

        self.model_us = nn.Sequential(*model_upsample)



        model_final = [nn.ReflectionPad2d(3), nn.Conv2d(ngf_global * 2, ngf_global, kernel_size=7, padding=0)]

        self.model_f = nn.Sequential(*model_final)



        self.model_dh1 = Dehaze1()



    def forward(self, input):

        output_ds1 = self.model_ds1(input)

        output_msgcba = self.model_ds2(output_ds1)

        output_msgcba = self.model_msgcba(output_msgcba)

        output_msgcba = self.model(output_msgcba)

        output_msgcba = self.model_msgcba(output_msgcba)

        output_us = self.model_us(output_msgcba)

        tmp = torch.cat((output_us, output_ds1), 1)

        output_f = self.model_f(tmp)



        tmp = torch.cat((output_f, input), 1)

        output_dh = self.model_dh1(tmp)

        tmp = torch.cat((output_f, output_dh), 1)

        output_sub_us = self.model_dh1(tmp)

        return output_dh, output_dh, output_sub_us



class Dehaze1(nn.Module):

    def __init__(self):

        super(Dehaze1, self).__init__()



        self.relu = nn.LeakyReLU(0.2, inplace=True)



        self.tanh = nn.Tanh()



        self.refine1 = nn.Conv2d( 32+3, 20, kernel_size=3, stride=1, padding=1)

        self.refine2 = nn.Conv2d(20, 20, kernel_size=3, stride=1, padding=1)



        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm

        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm

        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm

        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1, stride=1, padding=0)  # 1mm



        self.refine3 = nn.Conv2d(20+4, 3, kernel_size=3,stride=1,padding=1)



        self.upsample = F.upsample_nearest



        self.batch1 = nn.InstanceNorm2d(100, affine=True)



    def forward(self, x):

        dehaze = self.relu((self.refine1(x)))

        dehaze = self.relu((self.refine2(dehaze)))

        shape_out = dehaze.data.size()

        # print(shape_out)

        shape_out = shape_out[2:4]



        x101 = F.avg_pool2d(dehaze, 32)



        x102 = F.avg_pool2d(dehaze, 16)



        x103 = F.avg_pool2d(dehaze, 8)



        x104 = F.avg_pool2d(dehaze, 4)

        x1010 = self.upsample(self.relu(self.conv1010(x101)),size=shape_out)

        x1020 = self.upsample(self.relu(self.conv1020(x102)),size=shape_out)

        x1030 = self.upsample(self.relu(self.conv1030(x103)),size=shape_out)

        x1040 = self.upsample(self.relu(self.conv1040(x104)),size=shape_out)



        dehaze = torch.cat((x1010, x1020, x1030, x1040, dehaze), 1)

        dehaze= self.tanh(self.refine3(dehaze))



        return dehaze




class GlobalGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, 

                 padding_type='reflect'):

        assert(n_blocks >= 0)

        super(GlobalGenerator, self).__init__()        

        activation = nn.ReLU(True)        



        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]

        ### downsample

        for i in range(n_downsampling):

            mult = 2**i

            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),

                      norm_layer(ngf * mult * 2), activation]



        ### resnet blocks

        mult = 2**n_downsampling

        for i in range(n_blocks):

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        

        ### upsample         

        for i in range(n_downsampling):

            mult = 2**(n_downsampling - i)

            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),

                       norm_layer(int(ngf * mult / 2)), activation]

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]        

        self.model = nn.Sequential(*model)

            

    def forward(self, input):

        return self.model(input)             

        

# Define a resnet block

class ResnetBlock(nn.Module):

    '''

    def __init__(self, input_channel, mid_channel, output_channel, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):

        super(ResnetBlock, self).__init__()

        self.conv_block = self.build_conv_block(input_channel, mid_channel, output_channel, padding_type, norm_layer, activation, use_dropout)



    def build_conv_block(self, input_channel, mid_channel, output_channel, padding_type, norm_layer, activation, use_dropout):

        conv_block = []

        p = 0

        if padding_type == 'reflect':

            conv_block += [nn.ReflectionPad2d(1)]

        elif padding_type == 'replicate':

            conv_block += [nn.ReplicationPad2d(1)]

        elif padding_type == 'zero':

            p = 1

        else:

            raise NotImplementedError('padding [%s] is not implemented' % padding_type)



        conv_block += [nn.Conv2d(input_channel, mid_channel, kernel_size=3, padding=p),

                       norm_layer(mid_channel),

                       activation]

        if use_dropout:

            conv_block += [nn.Dropout(0.5)]



        p = 0

        if padding_type == 'reflect':

            conv_block += [nn.ReflectionPad2d(1)]

        elif padding_type == 'replicate':

            conv_block += [nn.ReplicationPad2d(1)]

        elif padding_type == 'zero':

            p = 1

        else:

            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(mid_channel, output_channel, kernel_size=3, padding=p),

                       norm_layer(output_channel)]



        return nn.Sequential(*conv_block)



    def forward(self, x):

        out = x + self.conv_block(x)

        return out



'''    

    def __init__(self, input_channel, middle_channel, output_channel, padding_type, norm_layer=nn.InstanceNorm2d, activation=nn.ReLU(True), use_dropout=False):

        super(ResnetBlock, self).__init__()

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



class MultiscaleDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, 

                 use_sigmoid=False, num_D=3, getIntermFeat=False):

        super(MultiscaleDiscriminator, self).__init__()

        self.num_D = num_D

        self.n_layers = n_layers

        self.getIntermFeat = getIntermFeat

     

        for i in range(num_D):

            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)

            if getIntermFeat:                                

                for j in range(n_layers+2):

                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                                   

            else:

                setattr(self, 'layer'+str(i), netD.model)



        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)



    def singleD_forward(self, model, input):

        if self.getIntermFeat:

            result = [input]

            for i in range(len(model)):

                result.append(model[i](result[-1]))

            return result[1:]

        else:

            return [model(input)]



    def forward(self, input):        

        num_D = self.num_D

        result = []

        input_downsampled = input

        for i in range(num_D):

            if self.getIntermFeat:

                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]

            else:

                model = getattr(self, 'layer'+str(num_D-1-i))

            result.append(self.singleD_forward(model, input_downsampled))

            if i != (num_D-1):

                input_downsampled = self.downsample(input_downsampled)

        return result

        

# Defines the PatchGAN discriminator with the specified arguments.

class NLayerDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):

        super(NLayerDiscriminator, self).__init__()

        self.getIntermFeat = getIntermFeat

        self.n_layers = n_layers



        kw = 4

        padw = int(np.ceil((kw-1.0)/2))

        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]



        nf = ndf

        for n in range(1, n_layers):

            nf_prev = nf

            nf = min(nf * 2, 512)

            sequence += [[

                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),

                norm_layer(nf), nn.LeakyReLU(0.2, True)

            ]]



        nf_prev = nf

        nf = min(nf * 2, 512)

        sequence += [[

            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),

            norm_layer(nf),

            nn.LeakyReLU(0.2, True)

        ]]



        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]



        if use_sigmoid:

            sequence += [[nn.Sigmoid()]]



        if getIntermFeat:

            for n in range(len(sequence)):

                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))

        else:

            sequence_stream = []

            for n in range(len(sequence)):

                sequence_stream += sequence[n]

            self.model = nn.Sequential(*sequence_stream)



    def forward(self, input):

        if self.getIntermFeat:

            res = [input]

            for n in range(self.n_layers+2):

                model = getattr(self, 'model'+str(n))

                res.append(model(res[-1]))

            return res[1:]

        else:

            return self.model(input)        



from torchvision import models

class Vgg19(torch.nn.Module):

    def __init__(self, requires_grad=False):

        super(Vgg19, self).__init__()

        vgg_pretrained_features = models.vgg19(pretrained=True).features

        self.slice1 = torch.nn.Sequential()

        self.slice2 = torch.nn.Sequential()

        self.slice3 = torch.nn.Sequential()

        self.slice4 = torch.nn.Sequential()

        self.slice5 = torch.nn.Sequential()

        for x in range(2):

            self.slice1.add_module(str(x), vgg_pretrained_features[x])

        for x in range(2, 7):

            self.slice2.add_module(str(x), vgg_pretrained_features[x])

        for x in range(7, 12):

            self.slice3.add_module(str(x), vgg_pretrained_features[x])

        for x in range(12, 21):

            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        for x in range(21, 30):

            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:

            for param in self.parameters():

                param.requires_grad = False



    def forward(self, X):

        h_relu1 = self.slice1(X)

        h_relu2 = self.slice2(h_relu1)        

        h_relu3 = self.slice3(h_relu2)        

        h_relu4 = self.slice4(h_relu3)        

        h_relu5 = self.slice5(h_relu4)                

        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]

        return out







'''

#####  MSGC block: no weight #####

## MSGC block + Dilated ResNet U-net + pyramid pooling

class MSGC_DRUN_PL(nn.Module): #32f

    def __init__(self, input_nc, output_nc, sub_model, full_attention, ngf=32, n_downsample_global=3, n_blocks_global=8, 

                 n_local_enhancers=1, n_resblocks=8, norm_layer=nn.InstanceNorm2d, padding_type='reflect'):

        super(MSGC_DRUN_PL, self).__init__()

        self.n_local_enhancers = n_local_enhancers

        self.sub_model = sub_model

        

        ###### global generator model #####           

        ngf_global = ngf * (2**(n_local_enhancers-1))

        print('ngf_global= %d' % (ngf_global))



        ###### Low-level Feature Extractors #####

        model_downsample1 = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0), norm_layer(ngf_global), nn.ReLU(True)]

        model_downsample2 = [nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1), norm_layer(ngf_global * 2), nn.ReLU(True)]

        self.model_ds1 = nn.Sequential(*model_downsample1)

        self.model_ds2 = nn.Sequential(*model_downsample2)



        ###### Non-local attention #####

        

        model_non_local_attention = [MSGCB(ngf * 2, ngf * 2)]

        model_non_local_attention += [ResnetBlock(ngf_global * 2, ngf_global * 2, ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

        #model_non_local_attention += [ResnetBlock(ngf * 2, ngf * 2, ngf * 2, padding_type=padding_type, norm_layer=norm_layer)]

        self.model_msgcb = nn.Sequential(*model_non_local_attention)



        ##### Attention U-Net #####

        #rdaunet = [ResnetBlock(ngf_global * 2, ngf_global * 2, ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

        if sub_model == 'attention_unet3':

            rdaunet = [archs.MSGCUNet3_Concate_Sym(opt)]



        else:

            raise('sub_model not implemented!') 



        #rdaunet += [ResnetBlock(ngf_global * 2, ngf_global * 2, ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]



        self.model = nn.Sequential(*rdaunet) 



        ###### Upsampling #####

        model_upsample = [nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1), norm_layer(ngf_global), nn.ReLU(True)]

        self.model_us = nn.Sequential(*model_upsample)



        model_final = [nn.ReflectionPad2d(3), nn.Conv2d(ngf_global * 2, ngf_global, kernel_size=7, padding=0)]

        self.model_f = nn.Sequential(*model_final)



        self.model_dh1 = Dehaze1()



    def forward(self, input):

        output_ds1 = self.model_ds1(input)

        output_msgcb = self.model_ds2(output_ds1)

        output_msgcb = self.model_msgcb(output_msgcb)

        output_msgcb = self.model(output_msgcb)

        output_msgcb = self.model_msgcb(output_msgcb)

        output_us = self.model_us(output_msgcb)

        tmp = torch.cat((output_us, output_ds1), 1)

        output_f = self.model_f(tmp)



        tmp = torch.cat((output_f, input), 1)

        output_dh = self.model_dh1(tmp)

        tmp = torch.cat((output_f, output_dh), 1)

        output_sub_us = self.model_dh1(tmp)

        return output_dh, output_dh, output_sub_us



'''