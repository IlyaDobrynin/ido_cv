import torch
import torch.nn as nn

from ..nn_blocks.encoders import EncoderCommon
from ..nn_blocks.common_blocks import ConvBnRelu
from ..nn_blocks.ocr_blocks import RNNModule


class CRNNFactory(EncoderCommon):
    """
    Class makes model for OCR task. Implements CRNN approach:
        1. features from input image extracts with CNN blocks
        2. Extracted features goes into RNN layer (LSTM or GRU)
        3. Exit of RNN goes into CTC loss

        Paper: https://arxiv.org/pdf/1507.05717.pdf

    Arguments:
        backbone:           Name of the u-net encoder line.
                            Should be in backbones.backbone_factory.backbones.keys()
        depth:              Depth of the u-net encoder. Should be in [1, 5] interval.
        num_classes:        Amount of output classes to predict
        num_filters:        Number of filters in first convolution layer. All filters based on this
                            value.
        pretrained:         Name of the pretrain weights. 'imagenet' or None
        rnn_type:           Type of rnn block:
                                - 'lstm'
                                - 'gru'
        rnn_depth:          Amount of recurrent blocks in decoder
        hidden_dim:         Size of RNN block hidden layer
        unfreeze_encoder:   Flag to unfreeze encoder weights
        custom_enc_start:   Flag to replace pretrained model first layer with custom ConvBnRelu
                            layer with stride 2
        num_input_channels: Amount of input channels
        bn_type:            Decoder batchnorm type:
                                'default' - normal nn.BatchNorm2d
                                'sync' - synchronized batсhnorm
        conv_type:          Decoder conv layer type:
                                'default' - sunple nn.Conv2d
                                'partial' - Partial convolution
                                'same' - Convolution with same padding
        depthwise:          Flag to use depthwise separable convolution instead of simple
                            torch.nn.Conv2d
    """

    def __init__(
            self,
            backbone:           str,
            depth:              int = 5,
            num_classes:        int = 1,
            num_filters:        int = 256,
            pretrained:         str = 'imagenet',
            rnn_type:           str = 'lstm',
            rnn_depth:          int = 2,
            hidden_dim:         int = 256,
            unfreeze_encoder:   bool = True,
            custom_enc_start:   bool = False,
            num_input_channels: int = 3,
            bn_type:            str = 'default',
            conv_type:          str = 'default',
            depthwise:          bool = False
    ):

        super(CRNNFactory, self).__init__(
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
        self.filters_tuple = (
            num_filters,
            num_filters,
            num_filters * 2,
            num_filters * 2
        )
        self.rnn_type = rnn_type
        self.rnn_depth = rnn_depth
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Get CNN
        self.cnn = self._make_cnn()
        self.avgpool = nn.AdaptiveAvgPool2d((1, None))
        # Get RNN
        self.rnn = self._make_rnn()

    def _make_cnn(self):
        """ Method makes CNN part of encoder for extracting features from input image

        :return: CNN Sequential
        """
        cnn = nn.Sequential()
        # Get first two layers from pretrained net
        for i in range(2):
            cnn.add_module(
                name=f'encoder_{i}',
                module=self.encoder_layers[i]
            )
        # Get other CNN layers
        for i in range(1, 4):
            common_conv_params = dict(
                depthwise=self.depthwise,
                conv_type=self.conv_type,
                bn_type=self.bn_type
            )
            in_ch = self.encoder_filters[1] if i == 0 else self.filters_tuple[i - 1]
            out_ch = self.filters_tuple[i]

            cnn.add_module(
                name=f'conv_bn_relu_{i}',
                module=ConvBnRelu(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=3,
                    padding=1,
                    **common_conv_params
                )
            )
            if i < 3:
                cnn.add_module(
                    name=f'pool_{i}',
                    module=nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1))
                )
        return cnn

    def _make_rnn(self):
        """ Method makes RNN part of encoder

        :return: CNN Sequential
        """
        rnn = nn.Sequential()
        for i in range(self.rnn_depth):
            hidden_size = self.hidden_dim
            if i == 0:
                in_channels = self.filters_tuple[-1]
                out_size = self.hidden_dim
            elif i == (self.rnn_depth - 1):
                in_channels = self.hidden_dim
                out_size = self.num_classes
            else:
                in_channels = out_size = self.hidden_dim
            rnn_params = dict(
                in_channels=in_channels,
                hidden_size=hidden_size,
                out_size=out_size,
                rnn_type=self.rnn_type
            )
            rnn.add_module(f'rnn_block_{i}', RNNModule(**rnn_params))

        return rnn

    def forward(self, x):
        last_feature = self.cnn(x)
        avg_pool_feature = self.avgpool(last_feature)
        squeezed_feature = avg_pool_feature.squeeze(2)
        permuted_feature = squeezed_feature.permute(2, 0, 1)
        output = self.rnn(permuted_feature)

        return output


if __name__ == '__main__':
    # backbone_name = 'resnet34'
    backbone_name = 'se_resnext50'
    input_size = (3, 256, 256)

    model = CRNNFactory(
        backbone=backbone_name, depth=5, num_classes=10, pretrained='imagenet',
        unfreeze_encoder=True, custom_enc_start=False, num_input_channels=3,
        bn_type='default', conv_type='default', depthwise=False
    )
    # print(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    from torchsummary import summary
    summary(model, input_size=input_size)
