import torch
import torch.nn as nn
# from .bn import ABN
from collections import OrderedDict


class VortexPooling(nn.Module):
    def __init__(self, in_chs, out_chs, feat_res=(8, 8), rate=(3, 9, 27)):
        super(VortexPooling, self).__init__()
        self.gave_pool = nn.Sequential(
            OrderedDict([
                ("gavg", nn.AdaptiveAvgPool2d((1, 1))),
                ("conv1x1", nn.Conv2d(in_chs, out_chs, kernel_size=1, bias=False)),
                ("up0", nn.Upsample(size=feat_res, mode='bilinear', align_corners=True)),
                ("bn0", nn.BatchNorm2d(num_features=out_chs))
            ])
        )

        self.conv3x3 = nn.Sequential(
            OrderedDict([
                ("conv3x3", nn.Conv2d(in_chs, out_chs, kernel_size=3, padding=1, bias=False)),
                ("bn3x3", nn.BatchNorm2d(num_features=out_chs))
            ])
        )

        vortex_conv_params = dict(
            in_channels=in_chs,
            out_channels=out_chs,
            kernel_size=3,
            bias=False
        )
        self.vortex_bra1 = nn.Sequential(
            OrderedDict([
                ("avg_pool", nn.AvgPool2d(kernel_size=rate[0], stride=1, padding=int((rate[0]-1)/2))),
                ("conv3x3", nn.Conv2d(padding=rate[0], dilation=rate[0], **vortex_conv_params)),
                ("bn3x3", nn.BatchNorm2d(num_features=out_chs))
            ])
        )

        self.vortex_bra2 = nn.Sequential(
            OrderedDict([
                ("avg_pool", nn.AvgPool2d(kernel_size=rate[1], stride=1, padding=int((rate[1]-1)/2))),
                ("conv3x3", nn.Conv2d(padding=rate[1], dilation=rate[1], **vortex_conv_params)),
                ("bn3x3", nn.BatchNorm2d(num_features=out_chs))
            ])
        )

        self.vortex_bra3 = nn.Sequential(
            OrderedDict([
                ("avg_pool", nn.AvgPool2d(kernel_size=rate[2], stride=1, padding=int((rate[2]-1)/2))),
                ("conv3x3", nn.Conv2d(padding=rate[2], dilation=rate[2], **vortex_conv_params)),
                ("bn3x3", nn.BatchNorm2d(num_features=out_chs))
            ])
        )

        self.vortex_catdown = nn.Sequential(
            OrderedDict([
                ("conv_down", nn.Conv2d(5 * out_chs, out_chs, kernel_size=1, bias=False)),
                ("bn_down", nn.BatchNorm2d(num_features=out_chs)),
                ("dropout", nn.Dropout2d(p=0.2, inplace=True))
            ])
        )

    def forward(self, x):
        out = torch.cat([self.gave_pool(x),
                         self.conv3x3(x),
                         self.vortex_bra1(x),
                         self.vortex_bra2(x),
                         self.vortex_bra3(x)], dim=1)
        out = self.vortex_catdown(out)
        return out
