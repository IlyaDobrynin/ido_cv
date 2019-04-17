# -*- coding: utf-8 -*-
"""
Module implements classification autobuilder class

"""
import torch
from torch import nn
from ..backbones import backbone_factory
from ..nn_blocks.encoders import EncoderCommon


class ClassifierFactory(EncoderCommon):
    def __init__(self, backbone, num_classes, pretrained='imagenet', unfreeze_encoder=True):

        super(ClassifierFactory, self).__init__(backbone=backbone,
                                                pretrained=pretrained,
                                                depth=5,
                                                unfreeze_encoder=unfreeze_encoder)

        assert backbone in backbone_factory.BACKBONES.keys(), \
            f"Wrong name of backbone: {backbone}. " \
                f"Should be in backbones.backbone_factory.backbones.keys()"

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(self.encoder_filters[-1], num_classes)

    def forward(self, x):
        x, _ = self._make_encoder_forward(x)
        out = self.avgpool(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


if __name__ == '__main__':
    backbone_name = 'se_resnext50'
    # backbone_name = 'resnet18'
    input_size = (3, 256, 256)
    model = ClassifierFactory(
        backbone=backbone_name, num_classes=1, pretrained='imagenet', unfreeze_encoder=True
    )
    # print(model.state_dict())
    # model = backbone_factory.get_backbone(name=backbone, pretrained='imagenet')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    from torchsummary import summary
    summary(model, input_size=input_size)

