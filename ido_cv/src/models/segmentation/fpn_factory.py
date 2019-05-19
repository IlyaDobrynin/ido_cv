# -*- coding: utf-8 -*-
"""
Module implements Feature Pyramid Network (FPN) autobuilder

"""
from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary

from ..nn_blocks.common_blocks import Conv
from ..nn_blocks.common_blocks import ConvBnRelu
from ..nn_blocks.encoders import EncoderCommon
from ..nn_blocks.se_blocks import SCSEBlock


class FPNFactory(EncoderCommon):
    """
        FPN auto builder class
        (https://arxiv.org/abs/1612.03144)

        Used papers and repos:
        Pretrained classifiers:     https://github.com/Cadene/pretrained-models.pytorch
        Squeeze & Excitation:       https://arxiv.org/abs/1709.01507v1
        Partial convolution:        https://arxiv.org/pdf/1811.11718.pdf

        Arguments:
            backbone:           name of the u-net encoder line.
                                Should be in backbones.backbone_factory.backbones.keys()
            depth:              depth of the u-net encoder. Should be in [1, 5] interval.
            num_classes:        amount of output classes to predict
            num_filters:        number of filters in first convolution layer
            pretrained:         name of the pretrain weights. 'imagenet' or None
            num_input_channels: amount of input channels
            dropout_rate:       decoder dropout rate
            conv_type:          decoder conv layer type:
                                    'default' - sunple nn.Conv2d
                                    'partial' - Partial convolution
            residual:           Flag to include residual decoder block instead of default
                                Should be one of 'vortex', 'dilation', 'fpa', 'fpa_dilation' or None
            gau:                Flag to include PAN-like skip-connection
            se_decoder:         Flag to include squeeze & excitation layers in decoder line
        """

    def __init__(self, backbone, depth=4, num_classes=1, num_filters=32, pretrained='imagenet',
                 unfreeze_encoder=True, custom_enc_start=False, num_input_channels=3,
                 dropout_rate=0.2, upscale_mode='nearest', depthwise=False,
                 bn_type='default', conv_type='default', residual=False, gau=False,
                 se_decoder=False):

        super(FPNFactory, self).__init__(backbone=backbone,
                                         pretrained=pretrained,
                                         depth=depth,
                                         unfreeze_encoder=unfreeze_encoder,
                                         custom_enc_start=custom_enc_start,
                                         num_input_channels=num_input_channels,
                                         bn_type=bn_type,
                                         conv_type=conv_type,
                                         depthwise=depthwise)
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.num_input_channels = num_input_channels
        self.dropout_rate = dropout_rate
        self.up_mode = upscale_mode
        self.residual = residual
        self.gau = gau
        self.se_decoder = se_decoder

        self.first_decoder_conv = self._get_first_decoder_conv()
        self.second_decoder_conv = self._get_second_decoder_conv()
        self.third_decoder_conv = self._get_third_decoder_conv()
        self.final_decoder_conv = self._get_final_decoder_conv()

        if self.se_decoder:
            self.se_layers = self._get_se_blocks()
            self.se_final = SCSEBlock(channel=self.num_filters, reduction=16)

        self.final_agg_conv = Conv(in_channels=self.num_filters * 2 * self.depth,
                                   out_channels=self.num_filters,
                                   kernel_size=1,
                                   conv_type=self.conv_type,
                                   depthwise=self.depthwise)
        self.final_layers = nn.Sequential(
            OrderedDict(
                [
                    ("conv_bn_relu_fin_1", ConvBnRelu(in_channels=self.num_filters,
                                                      out_channels=self.num_filters,
                                                      kernel_size=3,
                                                      padding=1,
                                                      depthwise=self.depthwise,
                                                      bn_type=self.bn_type,
                                                      conv_type=self.conv_type)),
                    ("conv_bn_relu_fin_2", ConvBnRelu(in_channels=self.num_filters,
                                                      out_channels=self.num_filters,
                                                      kernel_size=3,
                                                      padding=1,
                                                      depthwise=self.depthwise,
                                                      bn_type=self.bn_type,
                                                      conv_type=self.conv_type))
                ]
            )
        )
        self.final_conv = nn.Conv2d(in_channels=self.num_filters,
                                    out_channels=self.num_classes,
                                    kernel_size=3,
                                    padding=1)

    def _get_first_decoder_conv(self):
        first_conv_list = nn.ModuleList([])
        for i in range(self.depth):
            first_conv_layer = Conv(in_channels=self.encoder_filters[i],
                                    out_channels=self.num_filters * 3,
                                    kernel_size=1,
                                    conv_type=self.conv_type,
                                    depthwise=self.depthwise)
            # first_conv_layer = ConvBnRelu(in_channels=self.encoder_filters[i],
            #                               out_channels=self.num_filters * 3,
            #                               kernel_size=1,
            #                               bn_type=self.bn_type,
            #                               conv_type=self.conv_type)

            first_conv_list.append(first_conv_layer)
        return first_conv_list

    def _get_second_decoder_conv(self):
        second_conv_list = nn.ModuleList([])
        for i in range(self.depth):
            if i < (self.depth - 1):
                second_conv_layer = Conv(in_channels=self.num_filters * 3,
                                         out_channels=self.num_filters * 3,
                                         kernel_size=3,
                                         padding=1,
                                         conv_type=self.conv_type,
                                         depthwise=self.depthwise)
                # second_conv_layer = ConvBnRelu(in_channels=self.num_filters * 3,
                #                                out_channels=self.num_filters * 3,
                #                                kernel_size=3,
                #                                padding=1,
                #                                bn_type=self.bn_type,
                #                                conv_type=self.conv_type)
                second_conv_list.append(second_conv_layer)
        return second_conv_list

    def _get_third_decoder_conv(self):
        third_conv_list = nn.ModuleList([])
        for i in range(self.depth):
            third_conv_layer = Conv(in_channels=self.num_filters * 3,
                                    out_channels=self.num_filters * 2,
                                    kernel_size=1,
                                    conv_type=self.conv_type,
                                    depthwise=self.depthwise)
            # third_conv_layer = ConvBnRelu(in_channels=self.num_filters * 3,
            #                               out_channels=self.num_filters * 2,
            #                               kernel_size=1,
            #                               bn_type=self.bn_type,
            #                               conv_type=self.conv_type)
            third_conv_list.append(third_conv_layer)
        return third_conv_list

    def _get_final_decoder_conv(self):
        final_conv_list = nn.ModuleList([])
        for i in range(self.depth):
            final_conv_layer = nn.Sequential(
                OrderedDict(
                    [
                        ('conv_bn_relu_fin_1', ConvBnRelu(in_channels=self.num_filters * 2,
                                                          out_channels=self.num_filters * 2,
                                                          kernel_size=3,
                                                          padding=1,
                                                          depthwise=self.depthwise,
                                                          bn_type=self.bn_type,
                                                          conv_type=self.conv_type)),
                        ('conv_bn_relu_fin_1', ConvBnRelu(in_channels=self.num_filters * 2,
                                                          out_channels=self.num_filters * 2,
                                                          kernel_size=3,
                                                          padding=1,
                                                          depthwise=self.depthwise,
                                                          bn_type=self.bn_type,
                                                          conv_type=self.conv_type))
                    ]
                )
            )
            final_conv_list.append(final_conv_layer)
        return final_conv_list

    def _get_se_blocks(self):
        se_list = nn.ModuleList([])
        for i in range(self.depth):
            se_block = SCSEBlock(channel=self.num_filters * 2,
                                 reduction=16)
            se_list.append(se_block)
        return se_list

    def _make_decoder_forward(self, encoder_list):
        """

        :param encoder_list:
        :return:
        """

        # Make first convolution layers lane
        first_conv_list = list()
        for i, enc_feat in enumerate(encoder_list):
            first_conv_layer = self.first_decoder_conv[i](enc_feat)
            first_conv_list.append(first_conv_layer)

        # Make second convolution layers lane in reverse
        second_conv_list_reversed = list()
        for i in range(self.depth):
            rev_i = (self.depth - 1) - i
            if rev_i == (self.depth - 1):
                second_conv_list_reversed.append(first_conv_list[rev_i])
            else:
                h, w = first_conv_list[rev_i].size()[2], first_conv_list[rev_i].size()[3],
                up_x = F.interpolate(second_conv_list_reversed[i - 1],
                                     size=(h, w),
                                     mode=self.up_mode)
                second_conv_layer = first_conv_list[rev_i] + up_x
                second_conv_layer = self.second_decoder_conv[i - 1](second_conv_layer)
                second_conv_list_reversed.append(second_conv_layer)

        # Make final convolution layers lane
        h, w = second_conv_list_reversed[-1].size(2), second_conv_list_reversed[-1].size(3)
        final_conv_list = list()
        for i in range(self.depth):
            rev_i = (self.depth - 1) - i
            final_conv_layer_start = second_conv_list_reversed[rev_i]
            final_conv_layer_start = self.third_decoder_conv[i](final_conv_layer_start)
            final_conv_layer = self.final_decoder_conv[i](final_conv_layer_start)
            if self.se_decoder:
                final_conv_layer = self.se_layers[i](final_conv_layer)
            if self.residual:
                final_conv_layer = final_conv_layer + final_conv_layer_start
            if i != 0:
                final_conv_layer = F.interpolate(final_conv_layer, size=(h, w), mode=self.up_mode)
            final_conv_list.append(final_conv_layer)
        return final_conv_list

    def forward(self, x):
        """ Defines the computation performed at every call.

        :param x: Input Tensor
        :return Output Tensor
        """
        h, w = x.size()[2], x.size()[3]
        encoder_list = self._make_encoder_forward(x)
        final_conv_list = self._make_decoder_forward(encoder_list)
        x = torch.cat(final_conv_list, dim=1)
        x = F.interpolate(x, size=(h, w), mode=self.up_mode)
        x = self.final_agg_conv(x)
        out = self.final_layers(x)
        if self.se_decoder:
            out = self.se_final(out)
        if self.residual:
            out = x + out
        out = self.final_conv(out)
        return out


if __name__ == '__main__':
    backbone_name = 'resnet18'
    input_size = (3, 256, 256)
    model = FPNFactory(
        backbone=backbone_name, depth=5, num_classes=1, num_filters=32, pretrained='imagenet',
        num_input_channels=3, residual=True, gau=False, se_decoder=True
    )
    print(model)
    print(model.state_dict())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    summary(model, input_size=input_size)