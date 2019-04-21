"""
    Methods for testing mts_cv.src.models classes
"""

import torch
from torchsummary import summary
from ido_cv.src.models.detection.fpn.fpn_custom import FPNFactory


def test_fpn_custom():
    backbone_name = 'resnet50'
    input_size = (3, 512, 512)
    model = FPNFactory(
        backbone=backbone_name, depth=5, pretrained='imagenet', fpn_features=256,
        include_last_conv_layers=True, num_input_channels=3, residual=True, se_decoder=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    summary(model, input_size=input_size)


if __name__ == '__main__':
    test_fpn_custom()