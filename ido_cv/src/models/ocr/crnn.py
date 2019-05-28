import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..nn_blocks.encoders import EncoderCommon
# from mts_cv.src.models.nn_blocks.encoders import EncoderCommon


class BLSTM(nn.Module):

    def __init__(self, in_channels, hidden_size, out_size):
        super(BLSTM, self).__init__()

        self.rnn = nn.LSTM(in_channels, hidden_size, bidirectional=True)
        self.embedding = nn.Linear(hidden_size * 2, out_size)

    def forward(self, x):
        recurrent, _ = self.rnn(x)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)  # [T * b, out_size]
        output = output.view(T, b, -1)

        return output


class CRNN(EncoderCommon):

    def __init__(self, backbone, depth=5, num_classes=1, pretrained='imagenet',
                 hidden_dim: int = 256, unfreeze_encoder=True, custom_enc_start=False,
                 num_input_channels=3, bn_type='default', conv_type='default', depthwise=False):

        super(CRNN, self).__init__(
            backbone=backbone,
            pretrained=pretrained,
            depth=depth,
            unfreeze_encoder=unfreeze_encoder,
            custom_enc_start=custom_enc_start,
            num_input_channels=num_input_channels,
            bn_type=bn_type,
            conv_type=conv_type,
            depthwise=depthwise
        )

        kernel_sizes = [3, 3, 3, 3, 3, 3, 2]
        padding_sizes = [1, 1, 1, 1, 1, 1, 0]
        stride_sizes = [1, 1, 1, 1, 1, 1, 1]
        num_filters = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = num_input_channels if i == 0 else num_filters[i - 1]
            nOut = num_filters[i]
            cnn.add_module(
                'conv{0}'.format(i),
                nn.Conv2d(
                    nIn,
                    nOut,
                    kernel_sizes[i],
                    stride_sizes[i],
                    padding_sizes[i]
                )
            )
            if batchNormalization:
                cnn.add_module(
                    'batchnorm{0}'.format(i),
                    nn.BatchNorm2d(nOut)
                )
            cnn.add_module(
                'relu{0}'.format(i),
                nn.ReLU(True)
            )

        convRelu(0)                                                                    #   3 x 48 x 256
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(kernel_size=2, stride=2))  #  64 x 24 x 128
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(kernel_size=2, stride=2))  # 128 x 12 x 64
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1)))  # 256 x 6  x 64
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3), nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1)))  # 512 x 2 x 64
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn

        # print('CRNN.encoder_layers_0', self.encoder_layers[0])
        # print('CRNN.encoder_layers_1', self.encoder_layers[1])
        # print('CRNN.encoder_layers_2', self.encoder_layers[2])

        self.feat_idx = -1
        self.avgpool = nn.AdaptiveAvgPool2d((1, None))
        self.rnn = nn.Sequential(
            BLSTM(self.encoder_filters[self.feat_idx], hidden_dim, hidden_dim),
            # BLSTM(hidden_dim, hidden_dim, hidden_dim),
            # BLSTM(hidden_dim, hidden_dim, hidden_dim),
            BLSTM(hidden_dim, hidden_dim, num_classes))

    def forward(self, x):
        # conv features

        # conv_features = self._make_encoder_forward(x)
        # last_feature = conv_features[self.feat_idx]

        last_feature = self.cnn(x)                            # b_s x 512 x 1 x 16
        # b, c, h, w = conv.size()
        # assert h == 1, "the height of conv must be 1"
        avg_pool_feature = self.avgpool(last_feature)
        squeezed_feature = avg_pool_feature.squeeze(2)        # b_s x 512 x 16
        permuted_feature = squeezed_feature.permute(2, 0, 1)  # [w, b, c]
        # rnn features
        output = self.rnn(permuted_feature)

        # print(last_feature.size())
        # print(avg_pool_feature.size())
        # print(squeezed_feature.size())
        # print(permuted_feature.size())
        # print('out', output.shape)

        return output


if __name__ == '__main__':
    backbone_name = 'resnet34'
    # backbone_name = 'se_resnext50'
    input_size = (3, 256, 256)

    model = CRNN(
        backbone=backbone_name, depth=5, num_classes=10, pretrained='imagenet',
        unfreeze_encoder=True, custom_enc_start=False, num_input_channels=3,
        bn_type='default', conv_type='default', depthwise=False
    )
    # print(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    from torchsummary import summary
    summary(model, input_size=input_size)
