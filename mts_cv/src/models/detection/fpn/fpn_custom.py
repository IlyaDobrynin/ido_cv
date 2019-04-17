# -*- coding: utf-8 -*-
"""
Module implements Feature Pyramid Network (FPN) autobuilder

"""
from collections import OrderedDict
from torch import nn
from torch.nn import functional as F

from ...nn_blocks.common_blocks import ConvBnRelu
from ...nn_blocks.encoders import EncoderCommon
from ...nn_blocks.common_blocks import PartialConv2d
from ...nn_blocks.se_blocks import SCSEBlock


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
            pretrained:         name of the pretrain weights. 'imagenet' or None
            fpn_features:       number of filters in ffpn inner convolutions
            unfreeze_encoder:   Flag to make encoder trainable
            num_input_channels: amount of input channels
            dropout_rate:       decoder dropout rate
            bn_type:            batch normalization layers type:
                                    'default' - simple nn.BatchNorm2d
                                    'sync' - Synchronized batchnorm
            conv_type:          decoder conv layer type:
                                    'default' - simple nn.Conv2d
                                    'partial' - Partial convolution
            residual:           Flag to include residual decoder block instead of default
                                Should be one of 'vortex', 'dilation', 'fpa', 'fpa_dilation' or None
            se_decoder:         Flag to include squeeze & excitation layers in decoder line
        """
    
    def __init__(self, backbone, depth=5, pretrained='imagenet', unfreeze_encoder=True,
                 fpn_features=256, include_last_conv_layers=True, num_input_channels=3,
                 dropout_rate=0.2, bn_type='default', conv_type='default', up_mode='nearest',
                 residual=False, se_decoder=False):
        
        super(FPNFactory, self).__init__(backbone=backbone,
                                         pretrained=pretrained,
                                         depth=depth,
                                         unfreeze_encoder=unfreeze_encoder)
        if conv_type == 'default':
            self.ConvBlock = nn.Conv2d
        elif conv_type == 'partial':
            self.ConvBlock = PartialConv2d
        else:
            raise ValueError(
                'Wrong type of convolution: {}. Should be "default" or "partial"'.format(
                    conv_type
                )
            )

        self.fpn_features = fpn_features
        self.include_last_conv_layers = include_last_conv_layers
        self.num_input_channels = num_input_channels
        self.dropout_rate = dropout_rate
        self.up_mode = up_mode
        self.bn_type = bn_type
        self.conv_type = conv_type
        self.residual = residual
        self.se_decoder = se_decoder
        
        self.first_decoder_conv = self._get_first_decoder_conv(num_features=fpn_features)
        self.second_decoder_conv = self._get_second_decoder_conv(num_features=fpn_features)
        
        if self.include_last_conv_layers:
            self.third_decoder_conv = self._get_third_decoder_conv(num_features=fpn_features)
            self.fourth_decoder_conv = self._get_fourth_decoder_conv(num_features=fpn_features)
        
            if self.se_decoder:
                self.se_layers = self._get_se_blocks()
    
    def _get_first_decoder_conv(self, num_features):
        """ Function makes first level of convolutions
        
        :return:
        """
        first_conv_list = nn.ModuleList([])
        for i in range(self.depth):
            first_conv_layer = self.ConvBlock(in_channels=self.encoder_filters[i],
                                              out_channels=num_features,
                                              kernel_size=1)
            first_conv_list.append(first_conv_layer)
        return first_conv_list
    
    def _get_second_decoder_conv(self, num_features):
        """ Function makes second level of convolutions
        
        :return:
        """
        second_conv_list = nn.ModuleList([])
        for i in range(self.depth - 1):
            second_conv_layer = self.ConvBlock(in_channels=num_features,
                                               out_channels=num_features,
                                               kernel_size=3,
                                               padding=1)
            second_conv_list.append(second_conv_layer)
        return second_conv_list
    
    def _get_third_decoder_conv(self, num_features):
        """ Function makes third level of convolutions
        
        :return:
        """
        third_conv_list = nn.ModuleList([])
        for i in range(self.depth):
            third_conv_layer = self.ConvBlock(in_channels=num_features,
                                              # out_channels=int(num_features / 2),
                                              out_channels=num_features,
                                              kernel_size=1)
            third_conv_list.append(third_conv_layer)
        return third_conv_list
    
    def _get_fourth_decoder_conv(self, num_features):
        final_conv_list = nn.ModuleList([])
        for i in range(self.depth):
            final_conv_layer = nn.Sequential(
                OrderedDict(
                    [
                        (
                            'conv_bn_relu_fin_1', ConvBnRelu(
                                # in_channels=int(num_features / 2),
                                # out_channels=int(num_features / 2),
                                in_channels=num_features,
                                out_channels=num_features,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                dilation=1,
                                bn_type=self.bn_type,
                                conv_type=self.conv_type
                            )
                        ),
                        (
                            'conv_bn_relu_fin_1', ConvBnRelu(
                                # in_channels=int(num_features / 2),
                                # out_channels=int(num_features / 2),
                                in_channels=num_features,
                                out_channels=num_features,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                dilation=1,
                                bn_type=self.bn_type,
                                conv_type=self.conv_type)
                        )
                    ]
                )
            )
            final_conv_list.append(final_conv_layer)
        return final_conv_list
    
    def _get_se_blocks(self):
        se_list = nn.ModuleList([])
        for i in range(self.depth):
            se_block = SCSEBlock(
                channel=self.fpn_features,
                reduction=16
            )
            se_list.append(se_block)
        return se_list
    
    def _make_decoder_forward(self, encoder_list):
        """ Function combines convolution layers alltogether to make fpn output features

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
        for i in range(self.depth):                                                         # 0, 1, 2, 3, 4
            rev_i = (self.depth - 1) - i                                                    # 4, 3, 2, 1, 0
            if rev_i == (self.depth - 1):                                                   # if idx = 4
                second_conv_list_reversed.append(first_conv_list[rev_i])                    # append P5
            else:
                h, w = first_conv_list[rev_i].size()[2], first_conv_list[rev_i].size()[3]
                up_x = F.interpolate(second_conv_list_reversed[i - 1],
                                     size=(h, w),
                                     mode=self.up_mode)
                second_conv_layer = first_conv_list[rev_i] + up_x
                second_conv_layer = self.second_decoder_conv[i - 1](second_conv_layer)
                second_conv_list_reversed.append(second_conv_layer)
        
        if self.include_last_conv_layers:
            # Make third convolution layers lane
            third_conv_list = list()
            for i in range(self.depth):
                rev_i = (self.depth - 1) - i
                third_conv_layer_in = second_conv_list_reversed[rev_i]
                third_conv_layer_out = self.third_decoder_conv[i](third_conv_layer_in)
                third_conv_list.append(third_conv_layer_out)
                
            # Make fourth convolution layers lane
            out_list = list()
            for i in range(self.depth):
                fourth_conv_layer = self.fourth_decoder_conv[i](third_conv_list[i])
                if self.se_decoder:
                    fourth_conv_layer = self.se_layers[i](fourth_conv_layer)
                if self.residual:
                    fourth_conv_layer = fourth_conv_layer + third_conv_list[i]
                out_list.append(fourth_conv_layer)
        else:
            out_list = list()
            for i in range(self.depth):  # 0, 1, 2, 3, 4
                rev_i = (self.depth - 1) - i
                out_list.append(second_conv_list_reversed[rev_i])

        return out_list
    
    def forward(self, x):
        """ Defines the computation performed at every call.

        :param x: Input Tensor
        :return Output Tensor
        """
        _, encoder_list = self._make_encoder_forward(x)
        fourth_conv_list = self._make_decoder_forward(encoder_list)
        # print([i.shape for i in fourth_conv_list])
        P1 = fourth_conv_list[0]
        P2 = fourth_conv_list[1]
        P3 = fourth_conv_list[2]
        P4 = fourth_conv_list[3]
        P5 = fourth_conv_list[4]
        return P3, P4, P5
