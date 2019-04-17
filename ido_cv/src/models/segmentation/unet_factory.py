# -*- coding: utf-8 -*-
"""
Module implements U-net autobuilder class

"""
from collections import OrderedDict
import gc
import torch
from torch import nn
from torch.nn import functional as F
from ..backbones import backbone_factory
from ..nn_blocks.classic_unet_blocks import DecoderBlock
from ..nn_blocks.classic_unet_blocks import DecoderBlockResidual
from ..nn_blocks.encoders import EncoderCommon
from ..nn_blocks.common_blocks import ConvBnRelu
from ..nn_blocks.common_blocks import PartialConv2d
from ..nn_blocks.pan_blocks import FPABlock
from ..nn_blocks.pan_blocks import GAUBlockUnet
from ..nn_blocks.vortex_block import VortexPooling


class UnetFactory(EncoderCommon):
    """
    U-net auto builder class
    (https://arxiv.org/abs/1505.04597.pdf)

    Used papers and repos:
    Pretrained classifiers:     https://github.com/Cadene/pretrained-models.pytorch
    Dilation bottleneck:        https://github.com/lyakaap/Kaggle-Carvana-3rd-place-solution
    FPA bottleneck and GAU:     https://arxiv.org/pdf/1805.10180.pdf
    Vortex bottleneck:          https://arxiv.org/abs/1804.06242v1
    Hypercolumn:                https://arxiv.org/abs/1411.5752
    Squeeze & Excitation:       https://arxiv.org/abs/1709.01507v1
    Partial convolution:        https://arxiv.org/pdf/1811.11718.pdf
    
    Arguments:
        backbone:           name of the u-net encoder line.
                            Should be in backbones.backbone_factory.backbones.keys()
        depth:              depth of the u-net encoder. Should be in [1, 5] interval.
        num_classes:        amount of output classes to predict
        num_filters:        number of filters in first convolution layer
        pretrained:         name of the pretrain weights. 'imagenet' or None
        unfreeze_encoder:   Flag to unfreeze encoder weights
        num_input_channels: amount of input channels
        dropout_rate:       model dropout rate
        conv_type:          decoder conv layer type:
                                'default' - sunple nn.Conv2d
                                'partial' - Partial convolution
        depthwise:          Flag to use depthwise separable convolution instead of simple
                            torch.nn.Conv2d
        residual:           Flag to include residual decoder block instead of default
        mid_block:          Type of middle bottleneck block.
                                - 'vortex'
                                - 'dilation'
                                - 'fpa'
                                - 'fpa_dilation'
                                - None
        dilate_depth:       Optional argument if mid_block=='dilation'. Amount of dilation depth.
        gau:                Flag to include PAN-like skip-connection
        hypercolumn:        Flag to include hypercolumn
        se_decoder:         Flag to include squeeze & excitation layers in decoder line
    """

    def __init__(self, backbone, depth=4, num_classes=1, num_filters=32, pretrained='imagenet',
                 unfreeze_encoder=True, num_input_channels=3, dropout_rate=0.2, bn_type='default',
                 conv_type='default', depthwise=False, residual=False, mid_block=None,
                 dilate_depth=1, gau=False, hypercolumn=False, se_decoder=False):

        super(UnetFactory, self).__init__(backbone=backbone,
                                          pretrained=pretrained,
                                          depth=depth,
                                          unfreeze_encoder=unfreeze_encoder)

        assert backbone in backbone_factory.BACKBONES.keys(), \
            "Wrong name of backbone: {}. " \
            "Should be in backbones.backbone_factory.backbones.keys()".format(backbone)
        
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
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.num_input_channels = num_input_channels
        self.dropout_rate = dropout_rate
        self.bn_type = bn_type
        self.conv_type = conv_type
        self.depthwise = depthwise
        self.residual = residual
        self.mid_block = mid_block
        self.gau = gau
        self.hypercolumn = hypercolumn
        self.se_decoder = se_decoder

        self.decoder_filters = []
        for i in range(self.depth):
            self.decoder_filters.append(self.num_filters * (2 ** i))
        self.decoder_layers = self._get_decoder()

        if self.mid_block is not None:
            if self.mid_block == 'dilation':
                self.dilate_depth = dilate_depth
                self.dilation_layers = self._get_dilation_layers()
            elif self.mid_block == 'fpa':
                self.fpa = FPABlock(
                    in_channels=self.encoder_filters[self.depth - 1],
                    out_channels=self.encoder_filters[self.depth - 1],
                    depthwise=self.depthwise,
                    conv_type=self.conv_type
                )
            elif self.mid_block == 'fpa_dilation':
                self.dilate_depth = dilate_depth
                self.dilation_layers = self._get_dilation_layers()
                self.fpa = FPABlock(
                    in_channels=self.encoder_filters[self.depth - 1],
                    out_channels=self.encoder_filters[self.depth - 1],
                    depthwise=self.depthwise,
                    conv_type=self.conv_type
                )
                self.fpa_dil_conv = self.ConvBlock(
                    in_channels=self.encoder_filters[self.depth - 1] * 2,
                    out_channels=self.encoder_filters[self.depth - 1],
                    kernel_size=1
                )
            elif self.mid_block == 'vortex':
                self.vortex = VortexPooling(
                    in_chs=self.encoder_filters[self.depth - 1],
                    out_chs=self.encoder_filters[self.depth - 1],
                    feat_res=(10, 10)
                )
            else:
                raise ValueError(
                    'Wrong type of mid block: {}. '
                    'Should be "dilation", "fpa", "fpa_dilation", "vortex" or None.'.format(
                        self.mid_block
                    )
                )

        if self.gau:
            self.gau_layers = self._get_gau_layers()
 
        if self.hypercolumn:
            self.hypercolumn_layers = self._get_hypercolumn_layers()
            # self.hc_conv = self.ConvBlock(in_channels=self.num_filters, # * (self.depth + 1),
            #                               out_channels=self.num_filters,
            #                               kernel_size=1,
            #                               padding=0)
            self.hc_conv = ConvBnRelu(in_channels=self.num_filters, # * (self.depth + 1),
                                      out_channels=self.num_filters,
                                      kernel_size=1,
                                      padding=0,
                                      depthwise=self.depthwise,
                                      bn_type=self.bn_type,
                                      conv_type=self.conv_type
                                      )
            self.hc_dropout = nn.Dropout2d(p=0.5)

        self.first_layer = nn.Sequential(
            OrderedDict(
                [
                    ("conv_bn_relu_1", ConvBnRelu(in_channels=self.num_input_channels,
                                                  out_channels=self.num_input_channels,
                                                  kernel_size=3,
                                                  padding=1,
                                                  depthwise=self.depthwise,
                                                  bn_type=self.bn_type,
                                                  conv_type=self.conv_type)),
                    ("conv_bn_relu_2", ConvBnRelu(in_channels=self.num_input_channels,
                                                  out_channels=self.decoder_filters[1],
                                                  kernel_size=3,
                                                  padding=1,
                                                  depthwise=self.depthwise,
                                                  bn_type=self.bn_type,
                                                  conv_type=self.conv_type))
                ]
            )
        )
        self.final_layer = nn.Conv2d(in_channels=self.num_filters,
                                     out_channels=self.num_classes,
                                     kernel_size=1)

    def _get_dilation_layers(self):
        """ Function to define dilation bottleneck layers

        :return: List of dilation bottleneck layers
        """
        dilation_layers = nn.ModuleList([])
        filters = self.encoder_filters[self.depth - 1]
        for i in range(self.dilate_depth):
            d_block = ConvBnRelu(in_channels=filters,
                                 out_channels=filters,
                                 kernel_size=3,
                                 dilation=(2 ** i),
                                 padding=(2 ** i),
                                 bn_type=self.bn_type,
                                 conv_type=self.conv_type)
            dilation_layers.append(d_block)
        return dilation_layers

    def _get_gau_layers(self):
        """ Function to define gau bottleneck layers

        :return: List of gau bottleneck layers
        """
        gau_layers = nn.ModuleList([])
        for i in range(self.depth - 1):
            rev_i = (self.depth - 1) - i
            if i == 0:
                in_ch_decoder = self.encoder_filters[rev_i]
            else:
                in_ch_decoder = self.decoder_filters[rev_i + 1]
            in_ch_encoder = self.encoder_filters[rev_i - 1]
            out_ch = self.encoder_filters[rev_i - 1]
            gau_block = GAUBlockUnet(in_ch_encoder=in_ch_encoder,
                                     in_ch_decoder=in_ch_decoder,
                                     out_ch=out_ch,
                                     conv_type=self.conv_type)
            gau_layers.append(gau_block)
        return gau_layers

    def _get_decoder(self):
        """ Function to define u-net encoder layers

        :return: List of encoder layers
        """
        decoder_layers = nn.ModuleList([])
        for i in range(self.depth):
            rev_i = (self.depth - 1) - i
            if i == 0:
                in_skip_ch = self.encoder_filters[rev_i - 1]
                in_dec_ch = self.encoder_filters[rev_i]
            elif i == self.depth - 1:
                in_skip_ch = self.decoder_filters[1]
                in_dec_ch = self.num_filters * (2 ** (rev_i + 1))
            else:
                in_skip_ch = self.encoder_filters[rev_i - 1]
                in_dec_ch = self.num_filters * (2 ** (rev_i + 1))
                
            out_channels = self.decoder_filters[rev_i]
            if self.residual:
                d_block = DecoderBlockResidual(in_skip_ch=in_skip_ch,
                                               in_dec_ch=in_dec_ch,
                                               inside_channels=out_channels,
                                               dropout_rate=self.dropout_rate,
                                               se_include=self.se_decoder,
                                               depthwise=self.depthwise,
                                               bn_type=self.bn_type,
                                               conv_type=self.conv_type)
            else:
                d_block = DecoderBlock(in_skip_ch=in_skip_ch,
                                       in_dec_ch=in_dec_ch,
                                       out_channels=out_channels,
                                       dropout_rate=self.dropout_rate,
                                       conv_type=self.conv_type)
            decoder_layers.append(d_block)
        # print(decoder_layers)
        return decoder_layers

    def _get_hypercolumn_layers(self):
        """ Function to define hypercolumn layers

        :return: List of hypercolumn layers
        """
        hc_layers = nn.ModuleList([])
        for i in range(self.depth + 1):
            if i == 0:
                in_channels = self.encoder_filters[self.depth - 1]
            else:
                in_channels = self.decoder_filters[-i]
            out_channels = self.num_filters
            hc_layers.append(
                # self.ConvBlock(in_channels=in_channels,
                #                out_channels=out_channels,
                #                kernel_size=1,
                #                padding=0)
                ConvBnRelu(in_channels=in_channels,
                           out_channels=out_channels,
                           kernel_size=1,
                           padding=0,
                           depthwise=self.depthwise,
                           bn_type=self.bn_type,
                           conv_type=self.conv_type
                           )
            )
        return hc_layers

    def _make_decoder_forward(self, x, first_skip, encoder_list):
        """ Function to make u-net decoder

        :param x: Input tenzor
        :param first_skip: First skip connection layer
        :param encoder_list: List of encoder layers
        :return: Last layer tensor and list of decoder tensors
        """
        decoder_list = []
        for i, decoder_layer in enumerate(self.decoder_layers):
            neg_i = -(i + 1)
            if i == len(self.decoder_layers) - 1:
                skip = [first_skip]
            else:
                if self.gau:
                    skip = [self.gau_layers[i](encoder_list[neg_i - 1], x)]
                else:
                    skip = [encoder_list[neg_i - 1]]
            x = decoder_layer(x, skip)
            decoder_list.append(x.clone())
        return x, decoder_list

    def _make_dilation_bottleneck(self, x):
        """ Make dilation bottleneck
        (https://github.com/lyakaap/Kaggle-Carvana-3rd-place-solution)

        :param x: Input tensor
        :return: fpa bottleneck tensor
        """
        dilated_layers = list()
        dilated_layers.append(x.unsqueeze(-1))
        for dilation_layer in self.dilation_layers:
            x = dilation_layer(x)
            dilated_layers.append(x.unsqueeze(-1))
        x = torch.cat(dilated_layers, dim=-1)
        x = torch.sum(x, dim=-1)
        del dilated_layers
        gc.collect()
        return x

    def _make_hypercolumn_forward(self, encoder_last, decoder_list):
        """ Makes variation of hypercolumn layer (https://arxiv.org/abs/1411.5752.pdf)

        :param encoder_last: last layer of encoder
        :param decoder_list: list of decoder layers
        :return: hypercolumn layer
        """
        hc = []
        h, w = decoder_list[-1].size(2), decoder_list[-1].size(3)

        first_hc = self.hypercolumn_layers[0](encoder_last)
        first_hc = F.interpolate(first_hc, size=(h, w), mode='bilinear', align_corners=False)
        hc.append(first_hc.unsqueeze(-1))
        # hc.append(first_hc)
        for i, decoder in enumerate(decoder_list):
            hc_layer = self.hypercolumn_layers[i + 1](decoder)
            hc_layer = F.interpolate(hc_layer, size=(h, w), mode='bilinear', align_corners=False)
            hc.append(hc_layer.unsqueeze(-1))
            # hc.append(hc_layer)
            # out = out + hc_layer
        out = torch.cat(hc, dim=-1)
        out = torch.sum(out, dim=-1)
        del hc
        gc.collect()
        out = self.hc_conv(out)
        # out = self.hc_dropout(out)
        return out

    def forward(self, x):
        """
        Defines the computation performed at every call.
        """
        first_skip = self.first_layer(x)

        # Get encoder features
        # x, encoder_list = self._make_encoder_forward(x)
        x, encoder_list = self._make_encoder_forward(x)

        # Middle bottlenecks
        if self.mid_block == 'dilation':
            x = self._make_dilation_bottleneck(x)
        elif self.mid_block == 'fpa':
            x = self.fpa(x)
        elif self.mid_block == 'fpa_dilation':
            x_fpa = self.fpa(x)
            x_dil = self._make_dilation_bottleneck(x)
            x = torch.cat([x_dil, x_fpa], dim=1)
            # x = x_fpa + x_dil
            x = self.fpa_dil_conv(x)
        elif self.mid_block == 'vortex':
            x = self.vortex(x)

        # Get decoder features
        x, decoder_list = self._make_decoder_forward(x, first_skip, encoder_list)

        # Make hypercolumn
        if self.hypercolumn:
            x = self._make_hypercolumn_forward(encoder_list[-1], decoder_list)

        # Make final layer
        x = self.final_layer(x)
        return x


if __name__ == '__main__':
    backbone_name = 'resnet34'
    input_size = (3, 256, 256)
    model = UnetFactory(
        backbone=backbone_name, depth=5, num_classes=1, num_filters=32, pretrained='imagenet',
        num_input_channels=3, dropout_rate=0.2, conv_type='default', depthwise=False,
        residual=True, mid_block='fpa', hypercolumn=True, gau=False, se_decoder=False
    )
    # print(model.state_dict())
    # model = backbone_factory.get_backbone(name=backbone, pretrained='imagenet')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    from torchsummary import summary
    summary(model, input_size=input_size)
