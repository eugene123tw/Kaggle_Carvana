
from net.model.loss import *
from net.model.blocks import *

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



def make_linear_bn_prelu(in_channels, out_channels):
    return [
        nn.Linear(in_channels, out_channels, bias=False),
        nn.BatchNorm1d(out_channels),
        nn.PReLU(out_channels),
    ]


def make_conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1):
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]

def make_dilationconv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, dilation=1):
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, dilation=dilation ,bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]


class UNet256(nn.Module):
    def __init__(self, in_shape, num_classes):
        super(UNet256, self).__init__()
        in_channels, height, width = in_shape

        self.conv1d = nn.Sequential(
            *make_conv_bn_relu(in_channels, 16, kernel_size=3, stride=1, padding=1),
            *make_conv_bn_relu(16, 32, kernel_size=3, stride=2, padding=1),
        )

        self.conv2d = nn.Sequential(
            *make_conv_bn_relu(32, 64, kernel_size=3, stride=1, padding=1),
            *make_conv_bn_relu(64, 128, kernel_size=3, stride=1, padding=1),
        )

        self.conv3d = nn.Sequential(
            *make_conv_bn_relu(128, 256, kernel_size=3, stride=1, padding=1),
            *make_conv_bn_relu(256, 512, kernel_size=3, stride=1, padding=1),
        )

        self.conv4d = nn.Sequential(
            *make_conv_bn_relu(512, 512, kernel_size=3, stride=1, padding=1),
            *make_conv_bn_relu(512, 512, kernel_size=3, stride=1, padding=1),
        )

        self.conv5d = nn.Sequential(
            *make_conv_bn_relu(512, 512, kernel_size=1, stride=1, padding=0),
        )

        self.conv4u = nn.Sequential(
            *make_conv_bn_relu(1024, 512, kernel_size=3, stride=1, padding=1),
            *make_conv_bn_relu(512, 512, kernel_size=3, stride=1, padding=1),
        )

        self.conv3u = nn.Sequential(
            *make_conv_bn_relu(1024, 512, kernel_size=3, stride=1, padding=1),
            *make_conv_bn_relu(512, 128, kernel_size=3, stride=1, padding=1),
        )

        self.conv2u = nn.Sequential(
            *make_conv_bn_relu(256, 128, kernel_size=3, stride=1, padding=1),
            *make_conv_bn_relu(128, 32, kernel_size=3, stride=1, padding=1),
        )

        self.conv1u = nn.Sequential(
            *make_conv_bn_relu(64, 64, kernel_size=3, stride=1, padding=1),
            *make_conv_bn_relu(64, 32, kernel_size=3, stride=1, padding=1),
        )

        self.conv0u = nn.Sequential(
            *make_conv_bn_relu(32, 32, kernel_size=3, stride=1, padding=1),
        )

        self.last = nn.Conv2d(32, num_classes, kernel_size=1, stride=1, padding=0)


    def forward(self, x):

        conv1d = self.conv1d(x)                                 # 512, C=64
        out    = F.max_pool2d(conv1d, kernel_size=2, stride=2)  # 256, C=64

        conv2d = self.conv2d(out)                               # 256, C=128
        out    = F.max_pool2d(conv2d, kernel_size=2, stride=2)  # 128, C=128

        conv3d = self.conv3d(out)                               # 128, C=256
        out    = F.max_pool2d(conv3d, kernel_size=2, stride=2)  #  64, C=256

        conv4d = self.conv4d(out)                               # 64, C=512
        out    = F.max_pool2d(conv4d, kernel_size=2, stride=2)  # 32, C=512

        conv5d = self.conv5d(out)                               # 32, C=512
        out    = conv5d                                         # 32, C=512

        out    = F.upsample_bilinear(out,scale_factor=2)        # 64, C=512
        out    = torch.cat([out , conv4d],1)                                   # 64, C=512
        out    = self.conv4u(out)                               # 64, C=256


        out    = F.upsample_bilinear(out,scale_factor=2)        # 128, C=256
        out    = torch.cat([out , conv3d],1)                                  # 128, C=256
        out    = self.conv3u(out)                               # 128, C=128


        out    = F.upsample_bilinear(out,scale_factor=2)        # 256, C=128
        out = torch.cat([out, conv2d],1)                                    # 256, C=128
        out    = self.conv2u(out)                               # 256, C=64


        out    = F.upsample_bilinear(out,scale_factor=2)        # 512, C=64
        out = torch.cat([out, conv1d],1)                                    # 512, C=64
        out    = self.conv1u(out)                               # 512, C=64

        out = F.upsample_bilinear(out, scale_factor=2)  # 128
        out = self.conv0u(out)

        logits = self.last(out)

        return logits



class UNet512_2 (nn.Module):

    def __init__(self, in_shape, num_classes):
        super(UNet512_2, self).__init__()
        in_channels, height, width = in_shape


        #512
        self.down0a = nn.Sequential(
            *make_conv_bn_relu(in_channels, 16, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(16, 16, kernel_size=3, stride=1, padding=1 ),
        )
        #256


        #UNet512_2 ------------------------------------------------------------------------
        #256
        self.down0 = nn.Sequential(
            *make_conv_bn_relu(16, 32, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(32, 32, kernel_size=3, stride=1, padding=1 ),
        )
        #128

        self.down1 = nn.Sequential(
            *make_conv_bn_relu(32, 64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(64, 64, kernel_size=3, stride=1, padding=1 ),
        )
        #64

        self.down2 = nn.Sequential(
            *make_conv_bn_relu(64,  128, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(128, 128, kernel_size=3, stride=1, padding=1 ),
        )
        #32

        self.down3 = nn.Sequential(
            *make_conv_bn_relu(128, 256, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(256, 256, kernel_size=3, stride=1, padding=1 ),
        )
        #16

        self.down4 = nn.Sequential(
            *make_conv_bn_relu(256,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(512,512, kernel_size=3, stride=1, padding=1 ),
        )
        #8

        self.center = nn.Sequential(
            *make_conv_bn_relu(512, 1024, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(1024,1024, kernel_size=3, stride=1, padding=1 ),
        )

        #16
        self.up4 = nn.Sequential(
            *make_conv_bn_relu(512+1024,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     512,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     512,512, kernel_size=3, stride=1, padding=1 ),
            #nn.Dropout(p=0.10),
        )
        #16

        self.up3 = nn.Sequential(
            *make_conv_bn_relu(256+512,256, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    256,256, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    256,256, kernel_size=3, stride=1, padding=1 ),
        )
        #32

        self.up2 = nn.Sequential(
            *make_conv_bn_relu(128+256,128, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    128,128, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    128,128, kernel_size=3, stride=1, padding=1 ),
        )
        #64

        self.up1 = nn.Sequential(
            *make_conv_bn_relu( 64+128,64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     64,64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     64,64, kernel_size=3, stride=1, padding=1 ),
        )
        #128

        self.up0 = nn.Sequential(
            *make_conv_bn_relu( 32+64,32, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    32,32, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    32,32, kernel_size=3, stride=1, padding=1 ),
        )
        #128
        #-------------------------------------------------------------------------

        self.up0a = nn.Sequential(
            *make_conv_bn_relu( 16+32,16, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    16,16, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    16,16, kernel_size=3, stride=1, padding=1 ),
        )
        #128

        self.classify = nn.Conv2d(16, num_classes, kernel_size=1, stride=1, padding=0 )


    def forward(self, x):

        #512
        down0a = self.down0a(x)
        out    = F.max_pool2d(down0a, kernel_size=2, stride=2) #64

        down0 = self.down0(out)
        out   = F.max_pool2d(down0, kernel_size=2, stride=2) #64

        down1 = self.down1(out)
        out   = F.max_pool2d(down1, kernel_size=2, stride=2) #64

        down2 = self.down2(out)
        out   = F.max_pool2d(down2, kernel_size=2, stride=2) #32

        down3 = self.down3(out)
        out   = F.max_pool2d(down3, kernel_size=2, stride=2) #16

        down4 = self.down4(out)
        out   = F.max_pool2d(down4, kernel_size=2, stride=2) # 8

        out   = self.center(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #16
        out   = torch.cat([down4, out],1)
        out   = self.up4(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #32
        out   = torch.cat([down3, out],1)
        out   = self.up3(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #64
        out   = torch.cat([down2, out],1)
        out   = self.up2(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #128
        out   = torch.cat([down1, out],1)
        out   = self.up1(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #256
        out   = torch.cat([down0, out],1)
        out   = self.up0(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #512
        out   = torch.cat([down0a, out],1)
        out   = self.up0a(out)

        out   = self.classify(out)

        return out



class UNet512_shallow (nn.Module):

    def __init__(self, in_shape, num_classes=1):
        super(UNet512_shallow, self).__init__()
        in_channels, height, width = in_shape


        #512
        self.down0a = nn.Sequential(
            *make_conv_bn_relu(in_channels, 16, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(16, 16, kernel_size=3, stride=1, padding=1 ),
        )
        #256


        #UNet512_2 ------------------------------------------------------------------------
        #256
        self.down0 = nn.Sequential(
            *make_conv_bn_relu(16, 32, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(32, 32, kernel_size=3, stride=1, padding=1 ),
        )
        #128

        self.down1 = nn.Sequential(
            *make_conv_bn_relu(32, 64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(64, 64, kernel_size=3, stride=1, padding=1 ),
        )
        #64

        self.down2 = nn.Sequential(
            *make_conv_bn_relu(64,  128, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(128, 128, kernel_size=3, stride=1, padding=1 ),
        )
        #32

        self.center = nn.Sequential(
            *make_conv_bn_relu(128, 256, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(256, 256, kernel_size=3, stride=1, padding=1 ),
        )
        # 64

        self.up2 = nn.Sequential(
            *make_conv_bn_relu(256 + 128, 128, kernel_size=3, stride=1, padding=1),
            *make_conv_bn_relu(128, 128, kernel_size=3, stride=1, padding=1),
        )
        # 128

        self.up1 = nn.Sequential(
            *make_conv_bn_relu(128 + 64, 64, kernel_size=3, stride=1, padding=1),
            *make_conv_bn_relu(64, 64, kernel_size=3, stride=1, padding=1),
        )

        # 256
        self.up0 = nn.Sequential(
            *make_conv_bn_relu(64 + 32, 32, kernel_size=3, stride=1, padding=1),
            *make_conv_bn_relu(32, 32, kernel_size=3, stride=1, padding=1),
        )

        # 512
        self.up0a = nn.Sequential(
            *make_conv_bn_relu( 16+32,16, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    16,16, kernel_size=3, stride=1, padding=1 ),
        )

        self.classify = nn.Conv2d(16, num_classes, kernel_size=1, stride=1, padding=0 )


    def forward(self, x):

        #512
        down0a = self.down0a(x)
        out    = F.max_pool2d(down0a, kernel_size=2, stride=2) #64

        down0 = self.down0(out)
        out   = F.max_pool2d(down0, kernel_size=2, stride=2) #64

        down1 = self.down1(out)
        out   = F.max_pool2d(down1, kernel_size=2, stride=2) #64

        down2 = self.down2(out)
        out   = F.max_pool2d(down2, kernel_size=2, stride=2) #32

        out   = self.center(out)

        out   = F.upsample(out, scale_factor=2, mode='bilinear') #64
        out   = torch.cat([down2, out],1)
        out   = self.up2(out)

        out   = F.upsample(out, scale_factor=2, mode='bilinear') #128
        out   = torch.cat([down1, out],1)
        out   = self.up1(out)

        out   = F.upsample(out, scale_factor=2, mode='bilinear') #256
        out   = torch.cat([down0, out],1)
        out   = self.up0(out)

        out   = F.upsample(out, scale_factor=2, mode='bilinear') #512
        out   = torch.cat([down0a, out],1)
        out   = self.up0a(out)

        out   = self.classify(out)

        return out



class UNet1024 (nn.Module):
    def __init__(self, in_shape):
        super(UNet1024, self).__init__()
        C,H,W = in_shape
        #assert(C==3)

        #1024
        self.down1 = StackEncoder(  C,   24, kernel_size=3)   #512
        self.down2 = StackEncoder( 24,   64, kernel_size=3)   #256
        self.down3 = StackEncoder( 64,  128, kernel_size=3)   #128
        self.down4 = StackEncoder(128,  256, kernel_size=3)   # 64
        self.down5 = StackEncoder(256,  512, kernel_size=3)   # 32
        self.down6 = StackEncoder(512,  768, kernel_size=3)   # 16

        self.center = nn.Sequential(
            ConvBnRelu2d(768, 768, kernel_size=3, padding=1, stride=1 ),
        )

        # 8
        # x_big_channels, x_channels, y_channels
        self.up6 = StackDecoder(768,  768, 512, kernel_size=3)  # 16
        self.up5 = StackDecoder( 512, 512, 256, kernel_size=3)  # 32
        self.up4 = StackDecoder( 256, 256, 128, kernel_size=3)  # 64
        self.up3 = StackDecoder( 128, 128,  64, kernel_size=3)  #128
        self.up2 = StackDecoder(  64,  64,  24, kernel_size=3)  #256
        self.up1 = StackDecoder(  24,  24,  24, kernel_size=3)  #512
        self.classify = nn.Conv2d(24, 1, kernel_size=1, padding=0, stride=1, bias=True)


    def forward(self, x):

        out = x                       #;print('x    ',x.size())
                                      #
        down1,out = self.down1(out)  ##;print('down1',down1.size())  #256
        down2,out = self.down2(out)   #;print('down2',down2.size())  #128
        down3,out = self.down3(out)   #;print('down3',down3.size())  #64
        down4,out = self.down4(out)   #;print('down4',down4.size())  #32
        down5,out = self.down5(out)   #;print('down5',down5.size())  #16
        down6,out = self.down6(out)   #;print('down6',down6.size())  #8
        pass                          #;print('out  ',out.size())

        out = self.center(out)
        out = self.up6(down6, out)
        out = self.up5(down5, out)
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        out = self.up2(down2, out)
        out = self.up1(down1, out)
        #1024

        out = self.classify(out)
        out = torch.squeeze(out, dim=1)
        return out



class UNet_pyramid(nn.Module):

    def __init__(self, in_shape):
        super(UNet_pyramid, self).__init__()
        in_channels, height, width = in_shape
        self.height = height
        self.width  = width


        #512
        self.down0a = nn.Sequential(
            *make_conv_bn_relu(in_channels, 16, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(16, 16, kernel_size=3, stride=1, padding=1 ),
        )
        #256


        #UNet512_2 ------------------------------------------------------------------------
        #256
        self.down0 = nn.Sequential(
            *make_conv_bn_relu(16, 64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(64, 64, kernel_size=3, stride=1, padding=1 ),
        )
        #128

        self.down1 = nn.Sequential(
            *make_conv_bn_relu(64, 64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(64, 64, kernel_size=3, stride=1, padding=1 ),
        )
        #64

        self.down2 = nn.Sequential(
            *make_conv_bn_relu(64,  64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(64, 64, kernel_size=3, stride=1, padding=1 ),
        )
        #32

        self.center = nn.Sequential(
            *make_conv_bn_relu(64, 64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(64, 64, kernel_size=3, stride=1, padding=1 ),
        )
        # 64

        self.up2 = nn.Sequential(
            *make_conv_bn_relu(64, 64, kernel_size=3, stride=1, padding=1),
            *make_conv_bn_relu(64, 64, kernel_size=3, stride=1, padding=1),
        )
        # 128

        self.up1 = nn.Sequential(
            *make_conv_bn_relu(64, 64, kernel_size=3, stride=1, padding=1),
            *make_conv_bn_relu(64, 64, kernel_size=3, stride=1, padding=1),
        )

        # 256
        self.up0 = nn.Sequential(
            *make_conv_bn_relu(64, 64, kernel_size=3, stride=1, padding=1),
            *make_conv_bn_relu(64, 64, kernel_size=3, stride=1, padding=1),
        )

        # 512
        self.up0a = nn.Sequential(
            *make_conv_bn_relu( 64,64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu( 64,64, kernel_size=3, stride=1, padding=1 ),
        )

        self.lateral0 = nn.Sequential(
            *make_conv_bn_relu(64, 64, kernel_size=1, stride=1, padding=0),
        )

        self.lateral1 = nn.Sequential(
            *make_conv_bn_relu(64, 64, kernel_size=1, stride=1, padding=0),
        )

        self.lateral2 = nn.Sequential(
            *make_conv_bn_relu(64, 64, kernel_size=1, stride=1, padding=0),
        )

        self.mask0 = nn.Sequential(
            *make_dilationconv_bn_relu(64, 1, kernel_size=3, stride=1, padding=12, dilation=12),
        )

        self.mask1 = nn.Sequential(
            *make_dilationconv_bn_relu(64, 1, kernel_size=3, stride=1, padding=18, dilation=18),
        )

        self.mask2 = nn.Sequential(
            *make_dilationconv_bn_relu(64, 1, kernel_size=3, stride=1, padding=24, dilation=24),
        )

        self.final = nn.Sequential(
            *make_conv_bn_relu(64, 1, kernel_size=1, stride=1, padding=0),
        )

        self.classify = nn.Conv2d(1, 1, kernel_size=1, padding=0, stride=1)

    def forward(self, x):

        #512
        down0a = self.down0a(x)
        out    = F.max_pool2d(down0a, kernel_size=2, stride=2) #64

        down0 = self.down0(out)
        out   = F.max_pool2d(down0, kernel_size=2, stride=2) #64

        down1 = self.down1(out)
        out   = F.max_pool2d(down1, kernel_size=2, stride=2) #64

        down2 = self.down2(out)
        out   = F.max_pool2d(down2, kernel_size=2, stride=2) #32

        out   = self.center(out)

        out   = F.upsample(out, scale_factor=2, mode='nearest') #64
        out   = out + self.lateral2(down2)
        perdict2 = self.mask2(out)
        perdict2 = F.upsample(perdict2, size=(self.height, self.width))
        out   = self.up2(out)

        out   = F.upsample(out, scale_factor=2, mode='nearest') #128
        out = out + self.lateral1(down1)
        perdict1 = self.mask1(out)
        perdict1 = F.upsample(perdict1, size=(self.height, self.width))
        out   = self.up1(out)

        out   = F.upsample(out, scale_factor=2, mode='nearest') #256
        out = out + self.lateral0(down0)
        perdict0 = self.mask0(out)
        perdict0 = F.upsample(perdict0, size=(self.height, self.width))
        out   = self.up0(out)

        out   = F.upsample(out, scale_factor=2, mode='nearest') #512
        out   = self.up0a(out)

        out   = self.final(out)

        out   = out+perdict0+perdict1+perdict2
        out   = self.classify(out)


        return out

class UNet_pyramid_1 (nn.Module):

    def __init__(self, in_shape):
        super(UNet_pyramid_1, self).__init__()
        in_channels, height, width = in_shape
        self.height = height
        self.width  = width

        #512
        self.down0a = nn.Sequential(
            *make_conv_bn_relu(in_channels, 16, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(16, 16, kernel_size=3, stride=1, padding=1 ),
        )
        #256


        #UNet512_2 ------------------------------------------------------------------------
        #256
        self.down0 = nn.Sequential(
            *make_conv_bn_relu(16, 32, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(32, 32, kernel_size=3, stride=1, padding=1 ),
        )
        #128

        self.down1 = nn.Sequential(
            *make_conv_bn_relu(32, 64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(64, 64, kernel_size=3, stride=1, padding=1 ),
        )
        #64

        self.down2 = nn.Sequential(
            *make_conv_bn_relu(64,  128, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(128, 128, kernel_size=3, stride=1, padding=1 ),
        )
        #32

        self.center = nn.Sequential(
            *make_conv_bn_relu(128, 256, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(256, 256, kernel_size=3, stride=1, padding=1 ),
        )
        # 64

        self.up2 = nn.Sequential(
            *make_conv_bn_relu(256 + 128, 128, kernel_size=3, stride=1, padding=1),
            *make_conv_bn_relu(128, 128, kernel_size=3, stride=1, padding=1),
        )
        # 128

        self.up1 = nn.Sequential(
            *make_conv_bn_relu(128 + 64, 64, kernel_size=3, stride=1, padding=1),
            *make_conv_bn_relu(64, 64, kernel_size=3, stride=1, padding=1),
        )

        # 256
        self.up0 = nn.Sequential(
            *make_conv_bn_relu(64 + 32, 32, kernel_size=3, stride=1, padding=1),
            *make_conv_bn_relu(32, 32, kernel_size=3, stride=1, padding=1),
        )

        # 512
        self.up0a = nn.Sequential(
            *make_conv_bn_relu( 16+32,16, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    16,16, kernel_size=3, stride=1, padding=1 ),
        )

        self.classify = nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0 )

        self.lateral2 = nn.Sequential(
            *make_conv_bn_relu(128, 1, kernel_size=1, stride=1, padding=0),
        )

        self.lateral1 = nn.Sequential(
            *make_conv_bn_relu(64, 1, kernel_size=1, stride=1, padding=0),
        )

        self.lateral0 = nn.Sequential(
            *make_conv_bn_relu(32, 1, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):

        #512
        down0a = self.down0a(x)
        out    = F.max_pool2d(down0a, kernel_size=2, stride=2) #64

        down0 = self.down0(out)
        out   = F.max_pool2d(down0, kernel_size=2, stride=2) #64

        down1 = self.down1(out)
        out   = F.max_pool2d(down1, kernel_size=2, stride=2) #64

        down2 = self.down2(out)
        out   = F.max_pool2d(down2, kernel_size=2, stride=2) #32

        out   = self.center(out)

        out   = F.upsample(out, scale_factor=2, mode='bilinear') #64
        out   = torch.cat([down2, out],1)
        out   = self.up2(out)
        predict2 = self.lateral2(out)
        predict2 = F.upsample(predict2, size=(self.height, self.width))

        out   = F.upsample(out, scale_factor=2, mode='bilinear') #128
        out   = torch.cat([down1, out],1)
        out   = self.up1(out)
        predict1 = self.lateral1(out)
        predict1 = F.upsample(predict1, size=(self.height, self.width))

        out   = F.upsample(out, scale_factor=2, mode='bilinear') #256
        out   = torch.cat([down0, out],1)
        out   = self.up0(out)
        predict0 = self.lateral0(out)
        predict0 = F.upsample(predict0, size=(self.height, self.width))

        out   = F.upsample(out, scale_factor=2, mode='bilinear') #512
        out   = torch.cat([down0a, out],1)
        out   = self.up0a(out)

        out   = self.classify(out)
        out   = out + predict2 + predict1 + predict0

        return out


class UNet_pyramid_1024 (nn.Module):
    def __init__(self, in_shape):
        super(UNet_pyramid_1024, self).__init__()
        C,H,W = in_shape
        #assert(C==3)
        self.height, self.width = H, W

        #1024
        self.down1 = StackEncoder(  C,   24, kernel_size=3)   #512
        self.down2 = StackEncoder( 24,   64, kernel_size=3)   #256
        self.down3 = StackEncoder( 64,  128, kernel_size=3)   #128
        self.down4 = StackEncoder(128,  256, kernel_size=3)   # 64
        self.down5 = StackEncoder(256,  512, kernel_size=3)   # 32
        self.down6 = StackEncoder(512,  768, kernel_size=3)   # 16

        self.center = nn.Sequential(
            ConvBnRelu2d(768, 768, kernel_size=3, padding=1, stride=1 ),
        )

        # 8
        # x_big_channels, x_channels, y_channels
        self.up6 = StackDecoder(768,  768, 512, kernel_size=3)  # 16
        self.up5 = StackDecoder( 512, 512, 256, kernel_size=3)  # 32
        self.up4 = StackDecoder( 256, 256, 128, kernel_size=3)  # 64
        self.up3 = StackDecoder( 128, 128,  64, kernel_size=3)  #128
        self.up2 = StackDecoder(  64,  64,  24, kernel_size=3)  #256
        self.up1 = StackDecoder(  24,  24,  24, kernel_size=3)  #512

        self.lateral6 = nn.Sequential(
            *make_conv_bn_relu(512, 1, kernel_size=1, stride=1, padding=0),
        )
        self.lateral5 = nn.Sequential(
            *make_conv_bn_relu(256, 1, kernel_size=1, stride=1, padding=0),
        )
        self.lateral4 = nn.Sequential(
            *make_conv_bn_relu(128, 1, kernel_size=1, stride=1, padding=0),
        )
        self.lateral3 = nn.Sequential(
            *make_conv_bn_relu(64, 1, kernel_size=1, stride=1, padding=0),
        )
        self.lateral2 = nn.Sequential(
            *make_conv_bn_relu(24, 1, kernel_size=1, stride=1, padding=0),
        )


        self.classify = nn.Conv2d(24, 1, kernel_size=1, padding=0, stride=1, bias=True)


    def forward(self, x):

        out = x                       #;print('x    ',x.size())
                                      #
        down1,out = self.down1(out)  ##;print('down1',down1.size())  #256
        down2,out = self.down2(out)   #;print('down2',down2.size())  #128
        down3,out = self.down3(out)   #;print('down3',down3.size())  #64
        down4,out = self.down4(out)   #;print('down4',down4.size())  #32
        down5,out = self.down5(out)   #;print('down5',down5.size())  #16
        down6,out = self.down6(out)   #;print('down6',down6.size())  #8
        pass                          #;print('out  ',out.size())

        out = self.center(out)
        out = self.up6(down6, out)
        predict6 = self.lateral6(out)
        predict6 = F.upsample(predict6, size=(self.height, self.width))

        out = self.up5(down5, out)
        predict5 = self.lateral5(out)
        predict5 = F.upsample(predict5, size=(self.height, self.width))

        out = self.up4(down4, out)
        predict4 = self.lateral4(out)
        predict4 = F.upsample(predict4, size=(self.height, self.width))

        out = self.up3(down3, out)
        predict3 = self.lateral3(out)
        predict3 = F.upsample(predict3, size=(self.height, self.width))

        out = self.up2(down2, out)
        predict2 = self.lateral2(out)
        predict2 = F.upsample(predict2, size=(self.height, self.width))

        out = self.up1(down1, out)
        #1024

        out = self.classify(out)
        out = out + predict6 + predict5 + predict4 + predict3 + predict2
        out = torch.squeeze(out, dim=1)
        return out

class UNet_pyramid_1024_2 (nn.Module):
    def __init__(self, in_shape):
        super(UNet_pyramid_1024_2, self).__init__()
        C,H,W = in_shape
        #assert(C==3)
        self.height, self.width = H, W

        self.down1 = StackEncoder(C, 24, kernel_size=3)  # 512
        self.down2 = StackEncoder(24, 48, kernel_size=3)  # 256
        self.down3 = StackEncoder(48, 64, kernel_size=3)  # 128
        self.down4 = StackEncoder(64, 128, kernel_size=3)  # 64
        self.down5 = StackEncoder(128, 256, kernel_size=3)  # 32
        self.down6 = StackEncoder(256, 512, kernel_size=3)  # 16

        self.center = nn.Sequential(
            ConvBnRelu2d(512, 512, kernel_size=3, padding=1, stride=1),
        )

        self.up6 = StackDecoder(512, 512, 256, kernel_size=3)  # 16
        self.up5 = StackDecoder(256, 256, 128, kernel_size=3)  # 32
        self.up4 = StackDecoder(128, 128, 64, kernel_size=3)  # 64
        self.up3 = StackDecoder(64, 64, 48, kernel_size=3)  # 128
        self.up2 = StackDecoder(48, 48, 24, kernel_size=3)  # 256
        self.up1 = StackDecoder(24, 24, 24, kernel_size=3)  # 512




        self.lateral6 = nn.Sequential(
            *make_conv_bn_relu(256, 1, kernel_size=1, stride=1, padding=0),
        )
        self.lateral5 = nn.Sequential(
            *make_conv_bn_relu(128, 1, kernel_size=1, stride=1, padding=0),
        )
        self.lateral4 = nn.Sequential(
            *make_conv_bn_relu(64, 1, kernel_size=1, stride=1, padding=0),
        )
        self.lateral3 = nn.Sequential(
            *make_conv_bn_relu(48, 1, kernel_size=1, stride=1, padding=0),
        )
        self.lateral2 = nn.Sequential(
            *make_conv_bn_relu(24, 1, kernel_size=1, stride=1, padding=0),
        )


        self.classify = nn.Conv2d(24, 1, kernel_size=1, padding=0, stride=1, bias=True)


    def forward(self, x):

        out = x                       #;print('x    ',x.size())
                                      #
        down1,out = self.down1(out)  ##;print('down1',down1.size())  #256
        down2,out = self.down2(out)   #;print('down2',down2.size())  #128
        down3,out = self.down3(out)   #;print('down3',down3.size())  #64
        down4,out = self.down4(out)   #;print('down4',down4.size())  #32
        down5,out = self.down5(out)   #;print('down5',down5.size())  #16
        down6,out = self.down6(out)   #;print('down6',down6.size())  #8
        pass                          #;print('out  ',out.size())

        out = self.center(out)
        out = self.up6(down6, out)
        predict6 = self.lateral6(out)
        predict6 = F.upsample(predict6, size=(self.height, self.width))

        out = self.up5(down5, out)
        predict5 = self.lateral5(out)
        predict5 = F.upsample(predict5, size=(self.height, self.width))

        out = self.up4(down4, out)
        predict4 = self.lateral4(out)
        predict4 = F.upsample(predict4, size=(self.height, self.width))

        out = self.up3(down3, out)
        predict3 = self.lateral3(out)
        predict3 = F.upsample(predict3, size=(self.height, self.width))

        out = self.up2(down2, out)
        predict2 = self.lateral2(out)
        predict2 = F.upsample(predict2, size=(self.height, self.width))

        out = self.up1(down1, out)
        #1024

        out = self.classify(out)
        out = out + predict6 + predict5 + predict4 + predict3 + predict2
        out = torch.squeeze(out, dim=1)
        return out


if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    # https://discuss.pytorch.org/t/print-autograd-graph/692/8
    batch_size = 2
    C, H, W = 3, 1024, 1024


    if 1:
        inputs = torch.randn(batch_size, C, H, W)
        labels = torch.FloatTensor(batch_size, H, W).random_(1)
        # net = UNet512_2(in_shape=(C,H,W), num_classes=1).cuda().train()
        # net = UNet1024(in_shape=(C,H,W)).cuda().train()
        net = UNet_pyramid_1024_2(in_shape=(C,H,W)).cuda().train()

        x = Variable(inputs).cuda()
        y = Variable(labels).cuda()
        logits = net.forward(x)
        loss = BCELoss2d()(logits, y)
        loss.backward()

        print('logits')
        print(logits)

        print('loss')
        print(loss)
    pass