import torch
from torch import nn
from torch.nn import functional as F

from .common_blocks import ConvRelu
from .common_blocks import ConvBnRelu
from .common_blocks import PartialConv2d
from .common_blocks import DepthwiseConv2d
from .se_blocks import SCSEBlock
from .custom_layers.sync_batchnorm import SynchronizedBatchNorm2d


class DecoderBlock(nn.Module):
    """Paramaters for Deconvolution were chosen to avoid artifacts, following
    link https://distill.pub/2016/deconv-checkerboard/
    """
    
    def __init__(self, in_skip_ch, in_dec_ch, out_channels, dropout_rate=0.2,
                 bn_type='default', conv_type='default'):
        super(DecoderBlock, self).__init__()
        in_channels = in_skip_ch + in_dec_ch
        self.conv_bn_relu = ConvBnRelu(out_channels,
                                       out_channels,
                                       kernel_size=3,
                                       padding=1,
                                       bn_type=bn_type,
                                       conv_type=conv_type)
        # self.conv2 = ConvBnRelu(out_channels, out_channels, bn_type=bn_type, conv_type=conv_type)
        if conv_type == 'default':
            ConvBlock = nn.Conv2d
        elif conv_type == 'partial':
            ConvBlock = PartialConv2d
        else:
            raise ValueError(
                'Wrong type of convolution: {}. Should be "default" or "partial"'.format(
                    conv_type
                )
            )
        self.first_conv = ConvBlock(in_channels=in_channels,
                                    out_channels=out_channels,
                                    kernel_size=1)
        self.conv = ConvBlock(in_channels=out_channels,
                              out_channels=out_channels,
                              kernel_size=3,
                              padding=1)
        if bn_type == 'default':
            BatchNorm = nn.BatchNorm2d
        elif bn_type == 'sync':
            BatchNorm = SynchronizedBatchNorm2d
        else:
            raise ValueError(
                'Wrong type if bn: {}. Should be "default" or "sync"'
            )
        self.bn = BatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_rate)
    
    def forward(self, x, skip):
        scale_factor = int(skip[0].shape[-1] // x.shape[-1])
        if scale_factor > 1:
            x = F.interpolate(x, scale_factor=scale_factor, mode='nearest')
        skip.append(x)
        x = torch.cat(skip, 1)
        x = self.dropout(x)
        x = self.first_conv(x)
        out = self.conv_bn_relu(x)
        out = self.conv(out)
        out = out + x
        out = self.bn(out)
        out = self.relu(out)
        return out
    
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, BatchNorm, depthwise=False, add_relu=True,
                 add_se=False, bn_type='default', conv_type='default'):
        super(ResidualBlock, self).__init__()
        self.add_relu = add_relu
        self.add_se = add_se
        
        self.start_bn = BatchNorm(in_channels)
        self.start_relu = nn.ReLU(inplace=True)
        
        self.conv_bn_relu = ConvBnRelu(in_channels, out_channels, kernel_size=3, padding=1,
                                       conv_type=conv_type, bn_type=bn_type)
        conv_parameters = dict(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=3, padding=1)
        if depthwise:
            self.conv = DepthwiseConv2d(conv_type=conv_type, **conv_parameters)
        else:
            if conv_type == 'default':
                self.conv = nn.Conv2d(**conv_parameters)
            elif conv_type == 'partial':
                self.conv = PartialConv2d(**conv_parameters)
            else:
                raise ValueError(
                    'Wrong type of convolution: {}. Should be "default" or "partial"'.format(
                        conv_type
                    )
                )
        if self.add_se:
            self.se_block = SCSEBlock(out_channels, reduction=16)
        self.final_bn = BatchNorm(out_channels)
        self.final_relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        out = self.start_bn(x)
        out = self.start_relu(out)
        out = self.conv_bn_relu(out)
        out = self.conv(out)
        out = out + x
        out = self.final_bn(out)
        if self.add_se:
            out = self.se_block(out)
        if self.add_relu:
            out = self.final_relu(out)
        return out


class DecoderBlockResidual(nn.Module):
    """Paramaters for Deconvolution were chosen to avoid artifacts, following
    link https://distill.pub/2016/deconv-checkerboard/
    """
    
    def __init__(self, in_skip_ch, in_dec_ch, inside_channels, dropout_rate=0.2, se_include=False,
                 depthwise=False, bn_type='default', conv_type='default'):
        super(DecoderBlockResidual, self).__init__()
        self.se_include = se_include
        conv_parameters = dict(
            in_channels=in_skip_ch + in_dec_ch,
            out_channels=inside_channels,
            kernel_size=3,
            padding=1
        )
        if depthwise:
            self.conv = DepthwiseConv2d(conv_type=conv_type, **conv_parameters)
        else:
            if conv_type == 'default':
                self.conv = nn.Conv2d(**conv_parameters)
            elif conv_type == 'partial':
                self.conv = PartialConv2d(**conv_parameters)
            else:
                raise ValueError(
                    'Wrong type of convolution: {}. Should be "default" or "partial"'.format(
                        conv_type
                    )
                )
        if bn_type == 'default':
            BatchNorm = nn.BatchNorm2d
        elif bn_type == 'sync':
            BatchNorm = SynchronizedBatchNorm2d
        else:
            raise ValueError(
                'Wrong type if bn: {}. Should be "default" or "sync"'
            )
        # print(in_skip_ch + inside_channels)
        self.residual_1 = ResidualBlock(in_channels=inside_channels,
                                        out_channels=inside_channels,
                                        BatchNorm=BatchNorm,
                                        depthwise=depthwise,
                                        add_se=se_include,
                                        bn_type=bn_type,
                                        conv_type=conv_type)
        self.residual_2 = ResidualBlock(in_channels=inside_channels,
                                        out_channels=inside_channels,
                                        BatchNorm=BatchNorm,
                                        depthwise=depthwise,
                                        add_relu=False,
                                        add_se=se_include,
                                        bn_type=bn_type,
                                        conv_type=conv_type)
        if self.se_include:
            self.se_layers = SCSEBlock(inside_channels, reduction=16)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_rate)
    
    def forward(self, x, skip):
        h, w = skip[0].size()[2], skip[0].size()[3],
        x = F.interpolate(x, size=(h, w), mode='nearest')
        skip.append(x)
        x = torch.cat(skip, 1)
        x = self.dropout(x)
        x = self.conv(x)
        out = self.residual_1(x)
        out = self.residual_2(out)
        out = self.relu(out)
        return out
