import torch
import torch.nn as nn
import torch.nn.functional as F
from ..nn_blocks.se_blocks import SCSEBlock
from ..nn_blocks.common_blocks import BatchNorm
from ..nn_blocks.custom_layers.sync_batchnorm import SynchronizedBatchNorm2d


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, bn_type):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1,
                                     padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes, bn_type=bn_type)
        self.relu = nn.ReLU()
        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, in_channels, output_stride, bn_type):
        super(ASPP, self).__init__()

        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(in_channels, 256, 1, padding=0, dilation=dilations[0], bn_type=bn_type)
        self.aspp2 = _ASPPModule(in_channels, 256, 3, padding=dilations[1], dilation=dilations[1], bn_type=bn_type)
        self.aspp3 = _ASPPModule(in_channels, 256, 3, padding=dilations[2], dilation=dilations[2], bn_type=bn_type)
        self.aspp4 = _ASPPModule(in_channels, 256, 3, padding=dilations[3], dilation=dilations[3], bn_type=bn_type)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(in_channels, 256, 1, stride=1, bias=False),
                                             BatchNorm(256, bn_type=bn_type),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = BatchNorm(256, bn_type=bn_type)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Decoder(nn.Module):
    def __init__(self, low_level_inplanes, num_classes, bn_type, residual=False, se_decoder=False):
        super(Decoder, self).__init__()
        self.residual = residual
        self.se_decoder = se_decoder
        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48, bn_type=bn_type)
        self.relu = nn.ReLU()
        self.last_agg_conv = nn.Conv2d(304, 256, kernel_size=1)
        self.last_conv_line = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm(256, bn_type=bn_type),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm(256, bn_type=bn_type),
            nn.ReLU(),
            nn.Dropout(0.1))
        self.last_conv = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        if self.se_decoder:
            self.se_block = SCSEBlock(256)
        self._init_weight()

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)
    
        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear',
                          align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        
        x = self.last_agg_conv(x)
        out = self.last_conv_line(x)
        if self.se_decoder:
            out = self.se_block(out)
        if self.residual:
            out = x + out
        out = self.last_conv(out)
        return out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
