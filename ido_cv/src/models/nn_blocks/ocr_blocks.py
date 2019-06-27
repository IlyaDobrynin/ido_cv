from torch import nn


class CustomCNN(nn.Module):

    def __init__(
            self,
            num_input_channels: int,
            kernel_sizes:       tuple = (3, 3, 3, 3, 3, 3, 2),
            padding_sizes:      tuple = (1, 1, 1, 1, 1, 1, 0),
            stride_sizes:       tuple = (1, 1, 1, 1, 1, 1, 1),
            num_filters:        tuple = (64, 128, 256, 256, 512, 512, 512)
    ):
        super(CustomCNN, self).__init__()

        cnn = nn.Sequential()

        def convRelu(i, bn: bool = False):
            in_ch = num_input_channels if i == 0 else num_filters[i - 1]
            out_ch = num_filters[i]
            cnn.add_module(
                'conv{0}'.format(i),
                nn.Conv2d(in_ch, out_ch, kernel_sizes[i], stride_sizes[i], padding_sizes[i])
            )
            if bn:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(out_ch))
            cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)  # 3 x 48 x 256
        cnn.add_module('pooling{0}'.format(0),
                       nn.MaxPool2d(kernel_size=2, stride=2))  # 64 x 24 x 128
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1),
                       nn.MaxPool2d(kernel_size=2, stride=2))  # 128 x 12 x 64
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1)))  # 256 x 6  x 64
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1)))  # 512 x 2 x 64
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn

    def forward(self, x):
        out = self.cnn(x)
        return out


class RNNModule(nn.Module):

    def __init__(
            self,
            in_channels: int,
            hidden_size: int,
            out_size: int,
            rnn_type: str
    ):
        super(RNNModule, self).__init__()

        if rnn_type == 'lstm':
            rnn_block = nn.LSTM
        elif rnn_type == 'gru':
            rnn_block = nn.GRU
        else:
            raise ValueError(
                f"Wrong parameter rnn_type: {rnn_type}. "
                f"Should be 'lstm' or 'gru'."
            )

        self.rnn = rnn_block(in_channels, hidden_size, bidirectional=True)
        self.embedding = nn.Linear(hidden_size * 2, out_size)

    def forward(self, x):
        recurrent, _ = self.rnn(x)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)  # [T * b, out_size]
        output = output.view(T, b, -1)

        return output