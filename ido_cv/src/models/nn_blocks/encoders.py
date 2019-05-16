import gc
from collections import OrderedDict
from torch import nn

from ..nn_blocks.common_blocks import ConvBnRelu
from ..backbones import backbone_factory
from ..backbones.pretrain_parameters import encoder_dict


class EncoderCommon(nn.Module):
    def __init__(self, backbone, pretrained, depth, unfreeze_encoder, custom_enc_start=False,
                 num_input_channels=None, bn_type='default', conv_type='default', depthwise=False):
        super(EncoderCommon, self).__init__()

        assert backbone in backbone_factory.BACKBONES.keys(), \
            "Wrong name of backbone: {}. " \
            "Should be in backbones.backbone_factory.backbones.keys()".format(backbone)

        self.num_input_channels = num_input_channels
        self.depthwise = depthwise
        self.bn_type = bn_type
        self.conv_type = conv_type
        self.depth = depth
        self.backbone = backbone
        self.encoder_layers_dict = encoder_dict[backbone]['skip']
        self.encoder_filters = encoder_dict[backbone]['filters']
        self.is_featured = encoder_dict[backbone]['features']
        self.encoder = backbone_factory.get_backbone(name=self.backbone,
                                                     pretrained=pretrained,
                                                     requires_grad=unfreeze_encoder)
        if custom_enc_start:
            first_enc_layer = nn.Sequential(
                OrderedDict(
                    [
                        ("first_enc_conv_bn_relu", ConvBnRelu(
                            in_channels=self.num_input_channels,
                            out_channels=self.encoder_filters[0],
                            kernel_size=3,
                            padding=1,
                            stride=2,
                            depthwise=self.depthwise,
                            bn_type=self.bn_type,
                            conv_type=self.conv_type
                        ))
                    ]
                )
            )
        else:
            first_enc_layer = None
        self.encoder_layers = self._get_encoder(first_enc_layer)
    
    def _get_encoder(self, first_enc_layer):
        """ Function to define u-net encoder layers

        :return: List of encoder layers
        """
        encoder_list = nn.ModuleList([])
        if self.is_featured:
            for (mk, mv) in self.encoder.named_children():
                if mk == 'features':
                    if first_enc_layer is None:
                        for i in range(self.depth):
                            encoder_layer = nn.ModuleList([])
                            for layer in self.encoder_layers_dict[i]:
                                encoder_layer.append(dict(mv.named_children())[layer])
                            encoder_list.append(nn.Sequential(*encoder_layer))
                    else:
                        encoder_list.append(first_enc_layer)
                        for i in range(1, self.depth):
                            encoder_layer = nn.ModuleList([])
                            for layer in self.encoder_layers_dict[i]:
                                encoder_layer.append(dict(mv.named_children())[layer])
                            encoder_list.append(nn.Sequential(*encoder_layer))
                else:
                    continue
        else:
            if first_enc_layer is None:
                for i in range(self.depth):
                    encoder_layer = nn.ModuleList([])
                    for layer in self.encoder_layers_dict[i]:
                        encoder_layer.append(dict(self.encoder.named_children())[layer])
                    encoder_list.append(nn.Sequential(*encoder_layer))
            else:
                encoder_list.append(first_enc_layer)
                for i in range(1, self.depth):
                    encoder_layer = nn.ModuleList([])
                    for layer in self.encoder_layers_dict[i]:
                        encoder_layer.append(dict(self.encoder.named_children())[layer])
                    encoder_list.append(nn.Sequential(*encoder_layer))
        del self.encoder
        gc.collect()
        return encoder_list
    
    def _make_encoder_forward(self, x):
        """ Function to make u-net encoder

        :param x: Input tenzor
        :return: List of encoder tensors
        """
        encoder_list = []
        if self.backbone in ['pnasnet5large', 'nasnetalarge']:
            encoder_list_tmp = []
            counter = 2
            for i, outer_layer in enumerate(self.encoder_layers):
                if i < 2:
                    x = outer_layer(x)
                    encoder_list.append(x.clone())
                    encoder_list_tmp.append(x.clone())
                    continue
                else:
                    for inner_layer in outer_layer:
                        if self.backbone == 'nasnetalarge':
                            first_layer = encoder_list_tmp[counter - 1]
                            if counter == 2:
                                first_layer = encoder_list_tmp[counter - 2]
                                second_layer = encoder_list_tmp[counter - 1]
                            elif counter in (10, 17):
                                second_layer = encoder_list_tmp[counter - 3]
                            else:
                                second_layer = encoder_list_tmp[counter - 2]
                        else:
                            first_layer = encoder_list_tmp[counter - 2]
                            second_layer = encoder_list_tmp[counter - 1]
                        x = inner_layer(first_layer, second_layer)
                        encoder_list_tmp.append(x.clone())
                        counter += 1
                    encoder_list.append(x.clone())
            del encoder_list_tmp
            gc.collect()
        else:
            for encoder_layer in self.encoder_layers:
                x = encoder_layer(x)
                encoder_list.append(x)
        return encoder_list
    
    def forward(self, x):
        raise NotImplementedError
        # x, encoder_list = self._make_encoder_forward(x)
        # return x
    
    
if __name__ == '__main__':
    backbone_name = 'resnet18'
    input_size = (3, 256, 256)
    model = EncoderCommon(
        backbone=backbone_name, depth=4, pretrained='imagenet', unfreeze_encoder=True,
        custom_enc_start=True, num_input_channels=3, bn_type='default', conv_type='default',
        depthwise=False
    )

    # print(model.state_dict())
    # model = backbone_factory.get_backbone(name=backbone, pretrained='imagenet')
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    from torchsummary import summary
    summary(model, input_size=input_size)
