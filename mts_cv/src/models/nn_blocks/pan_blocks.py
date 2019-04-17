import torch
from torch.nn import functional as F
from torch import nn
from .common_blocks import ConvBnRelu


class GAUBlockUnet(nn.Module):
    
    def __init__(self, in_ch_encoder, in_ch_decoder, out_ch, conv_type='default'):  #
        super(GAUBlockUnet, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(in_ch_decoder,
                       out_ch,
                       kernel_size=1,
                       stride=1,
                       padding=0,
                       add_relu=False,
                       conv_type=conv_type),
            nn.Sigmoid()
        )
        self.conv1_up = ConvBnRelu(in_ch_decoder,
                                   out_ch,
                                   kernel_size=1,
                                   stride=1,
                                   padding=0,
                                   conv_type=conv_type)
        
        self.conv2 = ConvBnRelu(in_ch_encoder,
                                out_ch,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                conv_type=conv_type)
    
    # x: low level feature
    # y: high level feature
    def forward(self, x, y):
        h, w = x.size(2), x.size(3)
        y_up = F.interpolate(y, size=(h, w), mode='bilinear', align_corners=True)
        y_up = self.conv1_up(y_up)
        x = self.conv2(x)
        y = self.conv1(y)
        # print(y.shape)
        z = torch.mul(x, y)
        return y_up + z


class GAUBlock(nn.Module):
    
    def __init__(self, in_ch, out_ch):  #
        super(GAUBlock, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(out_ch, out_ch, kernel_size=1, stride=1, padding=0, add_relu=False),
            nn.Sigmoid()
        )
        # self.conv1_up = ConvBnRelu(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

        self.conv2 = ConvBnRelu(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
    
    # x: low level feature
    # y: high level feature
    def forward(self, x, y):
        h, w = x.size(2), x.size(3)
        y_up = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)(y)
        # y_up = self.conv1_up(y_up)
        x = self.conv2(x)
        y = self.conv1(y)
        # print(y.shape)
        z = torch.mul(x, y)
        return y_up + z


class FPABlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, depthwise=False, conv_type='default'):
        super(FPABlock, self).__init__()
        # global pooling branch
        self.branch1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(in_channels, out_channels, kernel_size=1,
                       stride=1, padding=0, conv_type=conv_type)
        )
        # midddle branch
        self.mid = nn.Sequential(
            ConvBnRelu(in_channels, out_channels, kernel_size=1,
                       stride=1, padding=0, conv_type=conv_type)
        )
        self.down1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBnRelu(in_channels, 1, kernel_size=7, stride=1,
                       padding=3, depthwise=depthwise, conv_type=conv_type)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBnRelu(1, 1, kernel_size=5, stride=1, padding=2,
                       depthwise=depthwise, conv_type=conv_type)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            ConvBnRelu(1, 1, kernel_size=3, stride=1, padding=1,
                       depthwise=depthwise, conv_type=conv_type),
            ConvBnRelu(1, 1, kernel_size=3, stride=1, padding=1,
                       depthwise=depthwise, conv_type=conv_type),
        )
        self.conv2 = ConvBnRelu(1, 1, kernel_size=5, stride=1, padding=2,
                                depthwise=depthwise, conv_type=conv_type)
        self.conv1 = ConvBnRelu(1, 1, kernel_size=7, stride=1, padding=3,
                                depthwise=depthwise, conv_type=conv_type)
    
    def forward(self, x):
        h, w = x.size(2), x.size(3)
        b1 = self.branch1(x)
        b1 = F.interpolate(b1, size=(h, w), mode='bilinear', align_corners=True)
        
        mid = self.mid(x)
        
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x3 = F.interpolate(x3, size=(h // 4, w // 4), mode='bilinear', align_corners=True)
        
        x2 = self.conv2(x2)
        x = x2 + x3
        x = F.interpolate(x, size=(h // 2, w // 2), mode='bilinear', align_corners=True)
        
        x1 = self.conv1(x1)
        x = x + x1
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        
        x = torch.mul(x, mid)
        x = x + b1
        return x
