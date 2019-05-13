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


class DecoderBlockResidual(nn.Module):
    """Paramaters for Deconvolution were chosen to avoid artifacts, following
    link https://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_skip_ch, in_dec_ch, out_channels, dropout_rate=0.2, se_include=False,
                 depthwise=False, upscale_mode='nearest', bn_type='default', conv_type='default'):
        super(DecoderBlockResidual, self).__init__()
        self.se_include = se_include
        self.upscale_mode = upscale_mode
        conv_parameters = dict(
            in_channels=in_skip_ch + in_dec_ch,
            out_channels=out_channels,
            kernel_size=1,
            conv_type=conv_type,
            depthwise=depthwise,
        )
        self.conv = Conv(
            **conv_parameters
        )
        # self.conv = ConvBnRelu(
        #     bn_type=bn_type,
        #     **conv_parameters
        # )
        residual_parameters = dict(
            in_channels=out_channels,
            out_channels=out_channels,
            depthwise=depthwise,
            add_se=se_include,
            bn_type=bn_type,
            conv_type=conv_type
        )
        self.residual_1 = ResidualBlock(
            **residual_parameters
        )
        self.residual_2 = ResidualBlock(
            **residual_parameters
        )
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x, skip):
        h, w = skip[0].size()[2], skip[0].size()[3]
        x = F.interpolate(x, size=(h, w), mode=self.upscale_mode)
        skip.append(x)
        x = torch.cat(skip, 1)
        x = self.dropout(x)
        x = self.conv(x)
        out = self.residual_1(x)
        out = self.residual_2(out)
        return out


# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, depthwise=False, add_relu=True,
#                  add_se=False, bn_type='default', conv_type='default'):
#         super(ResidualBlock, self).__init__()
#         self.add_relu = add_relu
#         self.add_se = add_se
#
#         self.start_bn = BatchNorm(in_channels, bn_type=bn_type)
#         self.start_relu = nn.ReLU(inplace=True)
#
#         self.conv_bn_relu = ConvBnRelu(in_channels, out_channels, kernel_size=3, padding=1,
#                                        conv_type=conv_type, bn_type=bn_type)
#         conv_parameters = dict(in_channels=out_channels, out_channels=out_channels,
#                                kernel_size=3, padding=1)
#         self.conv = Conv(conv_type=conv_type, depthwise=depthwise, **conv_parameters)
#         if self.add_se:
#             self.se_block = SCSEBlock(out_channels, reduction=16)
#         self.final_bn = BatchNorm(out_channels, bn_type=bn_type)
#         self.final_relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         out = self.start_bn(x)
#         out = self.start_relu(out)
#         out = self.conv_bn_relu(out)
#         out = self.conv(out)
#         out = self.final_bn(out)
#         if self.add_se:
#             out = self.se_block(out)
#         out = out + x
#         if self.add_relu:
#             out = self.final_relu(out)
#         return out


# class DecoderBlockResidual(nn.Module):
#     """Paramaters for Deconvolution were chosen to avoid artifacts, following
#     link https://distill.pub/2016/deconv-checkerboard/
#     """
#
#     def __init__(self, in_skip_ch, in_dec_ch, out_channels, dropout_rate=0.2, se_include=False,
#                  depthwise=False, upscale_mode='nearest', bn_type='default', conv_type='default'):
#         super(DecoderBlockResidual, self).__init__()
#         self.se_include = se_include
#         self.upscale_mode = upscale_mode
#         conv_parameters = dict(
#             in_channels=in_skip_ch + in_dec_ch,
#             out_channels=inside_channels,
#             kernel_size=3,
#             padding=1
#         )
#         self.conv = Conv(conv_type=conv_type, depthwise=depthwise, **conv_parameters)
#         self.residual_1 = ResidualBlock(in_channels=inside_channels,
#                                         out_channels=out_channels,
#                                         depthwise=depthwise,
#                                         add_se=se_include,
#                                         bn_type=bn_type,
#                                         conv_type=conv_type)
#         self.residual_2 = ResidualBlock(in_channels=out_channels,
#                                         out_channels=out_channels,
#                                         depthwise=depthwise,
#                                         add_relu=False,
#                                         add_se=se_include,
#                                         bn_type=bn_type,
#                                         conv_type=conv_type)
#         self.relu = nn.ReLU(inplace=True)
#         self.dropout = nn.Dropout2d(p=dropout_rate)
#
#     def forward(self, x, skip):
#         h, w = skip[0].size()[2], skip[0].size()[3],
#         x = F.interpolate(x, size=(h, w), mode=self.upscale_mode)
#         skip.append(x)
#         x = torch.cat(skip, 1)
#         x = self.dropout(x)
#         x = self.conv(x)
#         out = self.residual_1(x)
#         out = self.residual_2(out)
#         out = self.relu(out)
#         return out
