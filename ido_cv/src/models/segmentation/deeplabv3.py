# -*- coding: utf-8 -*-
"""
Module implements Deeplab v3 implementation with various
backbones

"""
import gc
import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary

from ..backbones.pretrain_parameters import encoder_dict
from ..nn_blocks.common_blocks import PartialConv2d
from ..nn_blocks.encoders import EncoderCommon
from ..nn_blocks.custom_layers.sync_batchnorm import SynchronizedBatchNorm2d
from ..nn_blocks.deeplab_blocks import ASPP
from ..nn_blocks.deeplab_blocks import Decoder


class DeepLabV3(EncoderCommon):
    def __init__(self, backbone, num_classes=1, pretrained='imagenet', unfreeze_encoder=True,
                 num_input_channels=3, bn_type='default', residual=False, se_decoder=False):

        super(DeepLabV3, self).__init__(backbone=backbone,
                                        pretrained=pretrained,
                                        depth=5,
                                        unfreeze_encoder=unfreeze_encoder)
        self.num_classes = num_classes
        self.num_input_channels = num_input_channels
        self.bn_type = bn_type
        self.residual = residual
        self.se_decoder = se_decoder
        self.encoder_filters = encoder_dict[backbone]['filters']
        self.aspp = ASPP(in_channels=self.encoder_filters[-1],
                         output_stride=16,
                         bn_type=self.bn_type)
        self.decoder_block = Decoder(low_level_inplanes=self.encoder_filters[1],
                                     num_classes=num_classes,
                                     bn_type=self.bn_type,
                                     residual=self.residual,
                                     se_decoder=self.se_decoder)
    
    def forward(self, x):
        """
        Defines the computation performed at every call.
        """
        h, w = x.size()[2], x.size()[3]
        x, encoder_list = self._make_encoder_forward(x)
        low_level_feature = encoder_list[1]
        del encoder_list
        gc.collect()
        x = self.aspp(x)
        x = self.decoder_block(x, low_level_feature)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        return x


if __name__ == '__main__':
    backbone_name = 'resnet34'
    input_size = (3, 256, 256)
    model = DeepLabV3(
        backbone=backbone_name, num_classes=1, pretrained='imagenet',
        num_input_channels=3, residual=True, se_decoder=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    summary(model, input_size=input_size)
