import gc
from torch import nn

from .custom_layers.sync_batchnorm import SynchronizedBatchNorm2d
from .custom_layers.partial_conv import PartialConv2d


class DepthwiseConv2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=1, dilation=1, bias=True, conv_type='default'):
        super(DepthwiseConv2d, self).__init__()
        if conv_type == 'default':
            conv = nn.Conv2d
        elif conv_type == 'partial':
            conv = PartialConv2d
        else:
            raise ValueError(
                'Wrong type of convolution: {}. Should be "default" or "partial"'.format(
                    conv_type
                )
            )
        self.depthwise = conv(in_channels,
                              in_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              groups=in_channels,
                              bias=bias)
        self.pointwise = conv(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
    

class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                 dilation=1, bias=True, depthwise=False, conv_type='default'):
        super(ConvRelu, self).__init__()
        conv_parameters = dict(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias
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
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class ConvBnRelu(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 dilation=1, bias=True, add_relu=True, depthwise=False, conv_type='default',
                 bn_type='default'):
        super(ConvBnRelu, self).__init__()
        conv_parameters = dict(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias
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
        self.add_relu = add_relu
        self.bn = BatchNorm(out_channels)
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.add_relu:
            x = self.activation(x)
        return x
