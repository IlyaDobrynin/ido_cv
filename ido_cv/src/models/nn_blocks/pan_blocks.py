import torch
from torch.nn import functional as F
from torch import nn
from .common_blocks import Conv
from .common_blocks import ConvBnRelu


class GAUBlockUnet(nn.Module):
    
    def __init__(self, in_ch_enc, in_ch_dec, out_ch, conv_type='default', bn_type='default',
                 depthwise=False, upscale_mode='bilinear'):
        super(GAUBlockUnet, self).__init__()

        self.upscale_mode = upscale_mode
        if self.upscale_mode == 'bilinear':
            self.align_corners = True
        else:
            self.align_corners = None

        conv_bn_relu_params = dict(
            out_channels=out_ch,
            conv_type=conv_type,
            bn_type=bn_type,
            depthwise=depthwise
        )
        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(in_channels=in_ch_dec, add_relu=False, kernel_size=1, **conv_bn_relu_params),
            nn.Sigmoid()
        )
        self.conv1_up = ConvBnRelu(in_channels=in_ch_dec, kernel_size=1, **conv_bn_relu_params)
        self.conv2 = ConvBnRelu(in_channels=in_ch_enc, kernel_size=3, padding=1, **conv_bn_relu_params)
    
    # x: low level feature
    # y: high level feature
    def forward(self, x, y):
        h, w = x.size(2), x.size(3)
        y_up = F.interpolate(y, size=(h, w), mode=self.upscale_mode, align_corners=self.align_corners)
        y_up = self.conv1_up(y_up)
        x = self.conv2(x)
        y = self.conv1(y)
        # print(y.shape)
        z = torch.mul(x, y)
        return y_up + z


class GAUBlock(nn.Module):
    
    def __init__(self, in_ch, out_ch, conv_type='default', bn_type='default', depthwise=False,
                 upscale_mode='bilinear', align_corners=True):
        super(GAUBlock, self).__init__()

        self.upscale_mode = upscale_mode
        self.align_corners = align_corners
        if self.upscale_mode == 'bilinear':
            self.align_corners = True
        else:
            self.align_corners = False

        conv_bn_relu_params = dict(
            out_channels=out_ch,
            conv_type=conv_type,
            bn_type=bn_type,
            depthwise=depthwise
        )
        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(in_channels=out_ch, kernel_size=1, add_relu=False, **conv_bn_relu_params),
            nn.Sigmoid()
        )
        self.conv2 = ConvBnRelu(in_channels=in_ch, kernel_size=3, padding=1, **conv_bn_relu_params)
    
    # x: low level feature
    # y: high level feature
    def forward(self, x, y):
        h, w = x.size(2), x.size(3)
        y_up = F.interpolate(y, size=(h, w), mode=self.upscale_mode, align_corners=self.align_corners)
        x = self.conv2(x)
        y = self.conv1(y)
        z = torch.mul(x, y)
        return y_up + z


class FPABlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, depthwise=False, conv_type='default',
                 upscale_mode='bilinear', align_corners=True):
        super(FPABlock, self).__init__()

        self.upscale_mode = upscale_mode
        self.align_corners = align_corners

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
        upscale_parameters = dict(
            mode=self.upscale_mode,
            align_corners=self.align_corners
        )
        b1 = F.interpolate(b1, size=(h, w), **upscale_parameters)
        
        mid = self.mid(x)
        
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x3 = F.interpolate(x3, size=(h // 4, w // 4), **upscale_parameters)
        
        x2 = self.conv2(x2)
        x = x2 + x3
        x = F.interpolate(x, size=(h // 2, w // 2), **upscale_parameters)
        
        x1 = self.conv1(x1)
        x = x + x1
        x = F.interpolate(x, size=(h, w), **upscale_parameters)
        
        x = torch.mul(x, mid)
        x = x + b1
        return x
