# -*- coding: utf-8 -*-
"""
Module implements classification autobuilder class

"""
import torch
from torch import nn
from ..backbones import backbone_factory
from ..nn_blocks.encoders import EncoderCommon


class ClassifierFactory(EncoderCommon):
    def __init__(
            self,
            backbone: str,
            depth: int,
            num_classes: int,
            avg_pool_kernel: int = 7,
            pretrained: str = 'imagenet',
            unfreeze_encoder: bool = True,
            custom_enc_start: bool = False,
            num_input_channels: int = 3,
            conv_type: str = 'default',
            bn_type: str = 'default',
            depthwise: bool = False
    ):

        super(ClassifierFactory, self).__init__(
            backbone=backbone,
            pretrained=pretrained,
            depth=depth,
            unfreeze_encoder=unfreeze_encoder,
            custom_enc_start=custom_enc_start,
            num_input_channels=num_input_channels,
            conv_type=conv_type,
            bn_type=bn_type,
            depthwise=depthwise
        )

        assert backbone in backbone_factory.BACKBONES.keys(), \
            f"Wrong name of backbone: {backbone}. " \
                f"Should be in backbones.backbone_factory.backbones.keys()"

        self.avg_pool_kernel = avg_pool_kernel
        self.avgpool = nn.AvgPool2d(avg_pool_kernel)
        self.fc = nn.Linear(self.encoder_filters[depth - 1], num_classes)

    def forward(self, x):
        encoder_list = self._make_encoder_forward(x)
        out = encoder_list[-1]
        # print(out.shape)
        out = self.avgpool(out)
        # print(out.shape)
        out = out.view(out.size(0), -1)
        # print(out.shape)
        out = self.fc(out)
        return out


if __name__ == '__main__':
    backbone_name = 'se_resnext50'
    # backbone_name = 'resnet18'
    input_size = (3, 256, 256)
    model = ClassifierFactory(
        backbone=backbone_name, depth=5, num_classes=47, pretrained='imagenet',
        unfreeze_encoder=True, avg_pool_kernel=7, custom_enc_start=False, num_input_channels=3,
        conv_type='default', bn_type='default', depthwise=False
    )
    # print(model.state_dict())
    # model = backbone_factory.get_backbone(name=backbone, pretrained='imagenet')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    from torchsummary import summary
    summary(model, input_size=input_size)

