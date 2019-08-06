import networkx as nx


from collections import OrderedDict
import gc
import torch
from torch import nn
from torch.nn import functional as F
from ..nn_blocks.classic_unet_blocks import DecoderBlock
from ..nn_blocks.classic_unet_blocks import DecoderBlockResidual
from ..nn_blocks.classic_unet_blocks import ResidualBlock
from ..nn_blocks.se_blocks import SCSEBlock
from ..nn_blocks.encoders import EncoderCommon
from ..nn_blocks.common_blocks import Conv
from ..nn_blocks.common_blocks import ConvBnRelu
from ..nn_blocks.pan_blocks import FPABlock
from ..nn_blocks.pan_blocks import GAUBlockUnet
from ..nn_blocks.vortex_block import VortexPooling
from torch.nn import functional as F


class UnetPlusPlusFactory(EncoderCommon):

    def __init__(
            self,
            backbone: str,
            depth: int = 5,
            num_classes: int = 1,
            num_filters: int = 32,
            pretrained: str = 'imagenet',
            unfreeze_encoder: bool = True,
            custom_enc_start: bool = False,
            num_input_channels: int = 3,
            dropout_rate: float = 0.2,
            bn_type: str = 'default',
            conv_type: str = 'default',
            upscale_mode: str = 'nearest',
            depthwise: bool = False,
            residual: bool = False
    ):

        super(UnetPlusPlusFactory, self).__init__(
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

        self.num_classes = num_classes
        self.num_filters = num_filters
        self.dropout_rate = dropout_rate
        self.upscale_mode = upscale_mode
        self.residual = residual

        self._encoders_dict = self.get_encoders_dict()
        self._graph = self.get_graph()
        self._nodes_dict, self._nodes_instances_dict = self.get_nodes_dict()
        self.final_layer = nn.Conv2d(
            in_channels=self.num_filters,
            out_channels=self.num_classes,
            kernel_size=1
        )

    def get_encoders_dict(self):
        encoders_dict = {}
        encoders_list = self.encoder_layers
        encoder_filter_sizes = self.encoder_filters
        for i, encoder_block in enumerate(encoders_list):
            encoders_dict["X_{:d}_0".format(i)] = {"instance": encoder_block, "output_channels_num": encoder_filter_sizes[i]}

        return encoders_dict

    def get_graph(self):
        depth = len(self._encoders_dict)
        graph = nx.DiGraph()

        previous_encoder_index = None
        for encoder_index in self._encoders_dict:
            graph.add_node(encoder_index, attr_dict={"type": "encoder"})
            if previous_encoder_index is not None:
                graph.add_edge(previous_encoder_index, encoder_index, output_type="downscale")
            previous_encoder_index = encoder_index

        depth_column = 1
        for i in range(depth - 2, -1, -1):
            for j in range(1, depth_column + 1):
                graph.add_node("X_{:d}_{:d}".format(i, j), attr_dict={"type":"decoder"})
                graph.add_edge("X_{:d}_{:d}".format(i, j - 1), "X_{:d}_{:d}".format(i, j),
                               output_type="skip_connection")
                for k in range(j):
                    graph.add_edge("X_{:d}_{:d}".format(i, k), "X_{:d}_{:d}".format(i, j),
                                   output_type="skip_connection")
                graph.add_edge("X_{:d}_{:d}".format(i + 1, j - 1), "X_{:d}_{:d}".format(i, j), output_type="upscale")
            depth_column = depth_column + 1

        return graph


    def get_nodes_dict(self):

        decoder_parameters = dict(
            dropout_rate=self.dropout_rate,
            depthwise=self.depthwise,
            upscale_mode=self.upscale_mode,
            bn_type=self.bn_type,
            conv_type=self.conv_type,
        )
        encoders_instances_dict = {key: value["instance"] for key, value in self._encoders_dict.items()}
        nodes_instances_dict = nn.ModuleDict(encoders_instances_dict)
        nodes_dict = {key: {"output_channels_num": value["output_channels_num"]} for key, value in self._encoders_dict.items()}

        node_names = list(nx.topological_sort(self._graph))

        for node_name in node_names:
            node_type = self._graph.nodes()[node_name]["attr_dict"]["type"]
            if node_type == "encoder":
                continue
            elif node_type == "decoder":
                in_skip_ch, in_dec_ch = 0, 0
                out_channels = self.num_filters * 2 ** (int(node_name.split("_")[1]))
                for node_out, node_in, params in self._graph.edges(data=True):
                    if node_in == node_name:
                        input_channels_num = nodes_dict[node_out]["output_channels_num"]
                        if params["output_type"] == "skip_connection":
                            in_skip_ch += input_channels_num
                        elif params["output_type"] == "upscale":
                            in_dec_ch += input_channels_num
                        else:
                            raise ValueError("Wrong output type of income edge {}. it can be 'skip_connection' "
                                             "or 'upscale' only".format(params["output_type"]))
                if self.residual:
                    node_instance = DecoderBlockResidual(
                        in_skip_ch=in_skip_ch,
                        in_dec_ch=in_dec_ch,
                        out_channels=out_channels,
                        **decoder_parameters
                    )
                else:
                    node_instance = DecoderBlock(
                        in_skip_ch=in_skip_ch,
                        in_dec_ch=in_dec_ch,
                        out_channels=out_channels,
                        **decoder_parameters
                    )
                nodes_dict[node_name] = {"output_channels_num": out_channels}
                nodes_instances_dict[node_name] = node_instance

            else:
                raise ValueError("Wrong type of node {}. It can be only 'encoder' or 'decoder'".format(node_type))

        return nodes_dict, nodes_instances_dict

    def forward(self, x):

        node_names = list(nx.topological_sort(self._graph))
        data_dict = {}
        #data_dict["x"] = x

        h, w = x.size()[2], x.size()[3]

        for node_name in node_names:
            node_type = self._graph.nodes()[node_name]["attr_dict"]["type"]
            x_node = None
            skip = None
            for node_out, node_in, params in self._graph.edges(data=True):
                if node_in == node_name:
                    if params["output_type"] == "skip_connection":
                        if skip is None:
                            skip = [data_dict[node_out]]
                        else:
                            skip.append(data_dict[node_out])
                    elif params["output_type"] == "upscale":
                        if x_node is None:
                            x_node = data_dict[node_out]
                        else:
                            raise ValueError("There is more than one input in node {}. Please check graph".format(node_name))
                    elif params["output_type"] == "downscale":
                        if x_node is None:
                            x_node = data_dict[node_out]
                        else:
                            raise ValueError("There is more than one input in node {}. Please check graph".format(node_name))
                    else:
                        raise ValueError("Wrong output type of income edge {}. it can be 'skip_connection' "
                                         "or 'upscale' only".format(params["output_type"]))
            if x_node is None:
                x_node = x
                #raise ValueError("There is no one input in node {}".format(node_name))
            decoder_layer = self._nodes_instances_dict[node_name]
            if node_type == "encoder":
                data_dict[node_name] = decoder_layer(x_node)
            elif node_type == "decoder":
                data_dict[node_name] = decoder_layer(x_node, skip)

        last_node_name = node_names[-1]
        out = F.interpolate(data_dict[last_node_name], size=(h, w), mode=self.upscale_mode)
        out = self.final_layer(out)
        return out
