import torch
from torch import nn
from torch.nn import functional as F
from .common_blocks import Conv
from .common_blocks import BatchNorm
from .common_blocks import ConvBnRelu
from .se_blocks import SCSEBlock


class DecoderBlock(nn.Module):
    """Paramaters for Deconvolution were chosen to avoid artifacts, following
    link https://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_skip_ch, in_dec_ch, out_channels, dropout_rate=0.2, depthwise=False,
                 upscale_mode='nearest', bn_type='default', conv_type='default', se_include=False,
                 se_reduction=16):
        super(DecoderBlock, self).__init__()
        in_channels = in_skip_ch + in_dec_ch
        self.se_include = se_include
        self.upscale_mode = upscale_mode

        # Initialize layers
        conv_params = dict(
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            depthwise=depthwise,
            conv_type=conv_type
        )
        self.conv_bn_relu = ConvBnRelu(in_channels=in_channels, bn_type=bn_type, **conv_params)
        self.conv = Conv(in_channels=out_channels, **conv_params)
        self.bn = BatchNorm(out_channels, bn_type=bn_type)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        if self.se_include:
            self.se_block = SCSEBlock(out_channels, reduction=se_reduction)

    def forward(self, x, skip):
        h, w = skip[0].size()[2], skip[0].size()[3]
        x = F.interpolate(x, size=(h, w), mode=self.upscale_mode)
        skip.append(x)
        x = torch.cat(skip, 1)
        x = self.dropout(x)
        x = self.conv_bn_relu(x)
        x = self.conv(x)
        x = self.bn(x)
        if self.se_include:
            x = self.se_block(x) + x
        x = self.relu(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, depthwise=False, bn_type='default',
                 conv_type='default', add_se=False, se_reduction=16):
        super(ResidualBlock, self).__init__()
        self.add_se = add_se

        # Initialize layers
        conv_params = dict(
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            depthwise=depthwise,
            conv_type=conv_type
        )
        self.conv_bn_relu_1 = ConvBnRelu(in_channels=in_channels, bn_type=bn_type, **conv_params)
        self.conv_bn_relu_2 = ConvBnRelu(in_channels=out_channels, bn_type=bn_type, **conv_params)
        self.conv = Conv(in_channels=out_channels, **conv_params)
        self.bn = BatchNorm(out_channels, bn_type=bn_type)
        if self.add_se:
            self.se_block = SCSEBlock(out_channels, reduction=se_reduction)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv_bn_relu_1(x)
        out = self.conv_bn_relu_2(out)
        out = self.conv(out)
        out = self.bn(out)
        if self.add_se:
            out = self.se_block(out)
        out = out + residual
        out = self.relu(out)
        return out


class IdentityBlock(nn.Module):
    def __init__(self, in_channels, out_channels, depthwise=False, bn_type='default',
                 conv_type='default', add_se=False, se_reduction=16):
        super(IdentityBlock, self).__init__()
        self.add_se = add_se

        # Initialize layers
        conv_params = dict(
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            depthwise=depthwise,
            conv_type=conv_type
        )
        self.conv_f = Conv(in_channels=in_channels, **conv_params)
        self.conv_bn_relu_1 = ConvBnRelu(in_channels=out_channels, bn_type=bn_type, **conv_params)
        self.conv_bn_relu_2 = ConvBnRelu(in_channels=out_channels, bn_type=bn_type, **conv_params)
        self.conv_l = Conv(in_channels=out_channels, **conv_params)
        self.bn = BatchNorm(out_channels, bn_type=bn_type)
        if self.add_se:
            self.se_block = SCSEBlock(out_channels, reduction=se_reduction)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_f(x)
        residual = x
        out = self.conv_bn_relu_1(x)
        out = self.conv_bn_relu_2(out)
        out = self.conv_l(out)
        out = self.bn(out)
        if self.add_se:
            out = self.se_block(out)
        out = out + residual
        out = self.relu(out)
        return out


class DecoderBlockResidual(nn.Module):
    """Paramaters for Deconvolution were chosen to avoid artifacts, following
    link https://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_skip_ch, in_dec_ch, out_channels, residual_depth=2, dropout_rate=0.2,
                 se_include=False, depthwise=False, upscale_mode='nearest', bn_type='default',
                 conv_type='default'):
        super(DecoderBlockResidual, self).__init__()
        self.se_include = se_include
        self.upscale_mode = upscale_mode
        self.conv = Conv(
            in_channels=in_skip_ch + in_dec_ch,
            out_channels=out_channels,
            kernel_size=1,
            conv_type=conv_type,
            depthwise=depthwise,
        )
        residual_parameters = dict(
            in_channels=out_channels,
            out_channels=out_channels,
            depthwise=depthwise,
            add_se=se_include,
            bn_type=bn_type,
            conv_type=conv_type
        )
        self.residual_layers = nn.ModuleList([
            ResidualBlock(**residual_parameters) for _ in range(residual_depth)
        ])
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x, skip):
        h, w = skip[0].size()[2], skip[0].size()[3]
        x = F.interpolate(x, size=(h, w), mode=self.upscale_mode)
        skip.append(x)
        x = torch.cat(skip, dim=1)
        x = self.conv(x)
        x = self.dropout(x)
        for i, res_layer in enumerate(self.residual_layers):
            x = res_layer(x)
        return x
