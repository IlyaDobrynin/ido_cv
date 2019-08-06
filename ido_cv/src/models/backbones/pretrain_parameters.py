resnet_layers = (
    ['conv1', 'bn1', 'relu'],
    ['maxpool', 'layer1'],
    ['layer2'],
    ['layer3'],
    ['layer4']
)
senet_layers = (
    ['layer0'],
    ['maxpool1', 'layer1'],
    ['layer2'],
    ['layer3'],
    ['layer4']
)
pnasnet_layers = (
    ['conv_0'],
    ['cell_stem_0'],
    ['cell_stem_1', 'cell_0', 'cell_1', 'cell_2', 'cell_3'],
    ['cell_4', 'cell_5', 'cell_6', 'cell_7'],
    ['cell_8', 'cell_9', 'cell_10', 'cell_11']
)
nasnet_layers = (
    ['conv0'],
    ['cell_stem_0'],
    ['cell_stem_1', 'cell_0', 'cell_1', 'cell_2', 'cell_3', 'cell_4', 'cell_5'],
    ['reduction_cell_0', 'cell_6', 'cell_7', 'cell_8', 'cell_9', 'cell_10', 'cell_11'],
    ['reduction_cell_1', 'cell_12', 'cell_13', 'cell_14', 'cell_15', 'cell_16', 'cell_17']
)
inceptionresnetv2_layers = (
    ['conv2d_1a', 'conv2d_2a', 'conv2d_2b'],
    ['maxpool_3a', 'conv2d_3b', 'conv2d_4a'],
    ['maxpool_5a', 'mixed_5b', 'repeat'],
    ['mixed_6a', 'repeat_1'],
    ['mixed_7a', 'repeat_2', 'block8', 'conv2d_7b']
)
inceptionv4_layers = (
    ['0', '1', '2'],
    ['3', '4'],
    ['5', '6', '7', '8', '9'],
    ['10', '11', '12', '13', '14', '15', '16', '17'],
    ['18', '19', '20', '21']
)
xception_layers = (
    ['conv1', 'bn1', 'relu1', 'conv2', 'bn2', 'relu2'],
    ['block1'],
    ['block2'],
    ['block3', 'block4', 'block5', 'block6', 'block7', 'block8', 'block9', 'block10', 'block11'],
    ['block12', 'conv3', 'bn3', 'relu3', 'conv4', 'bn4']
)
mobilenetv2_layers = (
    ['init_block', 'stage1'],
    ['stage2'],
    ['stage3'],
    ['stage4'],
    ['stage5', 'final_block']
)
airnext_layers = [
        ['init_block'],
        ['stage1'],
        ['stage2'],
        ['stage3'],
        ['stage4']
    ]

wrn_layers = [
        ['init_block'],
        ['stage1'],
        ['stage2'],
        ['stage3'],
        ['stage4']
    ]

encoder_dict = {
    'resnet18': {
        'skip': resnet_layers,
        'filters': (64, 64, 128, 256, 512),
        'features': False
    },
    'resnet34': {
        'skip': resnet_layers,
        'filters': (64, 64, 128, 256, 512),
        'features': False
    },
    'resnet50': {
        'skip': resnet_layers,
        'filters': (64, 256, 512, 1024, 2048),
        'features': False
    },
    'resnet101': {
        'skip': resnet_layers,
        'filters': (64, 256, 512, 1024, 2048),
        'features': False
    },
    'resnet152': {
        'skip': resnet_layers,
        'filters': (64, 256, 512, 1024, 2048),
        'features': False
    },
    'dilated_resnet18': {
        'skip': resnet_layers,
        'filters': (64, 64, 128, 256, 512),
        'features': False
    },
    'dilated_resnet34': {
        'skip': resnet_layers,
        'filters': (64, 64, 128, 256, 512),
        'features': False
    },
    'dilated_resnet50': {
        'skip': resnet_layers,
        'filters': (64, 256, 512, 1024, 2048),
        'features': False
    },
    'dilated_resnet101': {
        'skip': resnet_layers,
        'filters': (64, 256, 512, 1024, 2048),
        'features': False
    },
    'dilated_resnet152': {
        'skip': resnet_layers,
        'filters': (64, 256, 512, 1024, 2048),
        'features': False
    },
    'cafferesnet101': {
        'skip': resnet_layers,
        'filters': (64, 256, 512, 1024, 2048),
        'features': False
    },
    'fbresnet152': {
        'skip': resnet_layers,
        'filters': (64, 256, 512, 1024, 2048),
        'features': False
    },
    'resnext50': {
        'skip': resnet_layers,
        'filters': (64, 256, 512, 1024, 2048),
        'features': False
    },
    'resnext101': {
        'skip': resnet_layers,
        'filters': (64, 256, 512, 1024, 2048),
        'features': False
    },
    'senet154': {
        'skip': senet_layers,
        'filters': (128, 256, 512, 1024, 2048),
        'features': False
    },
    'se_resnet50': {
        'skip': senet_layers,
        'filters': (64, 256, 512, 1024, 2048),
        'features': False
    },
    'se_resnet152': {
        'skip': senet_layers,
        'filters': (64, 256, 512, 1024, 2048),
        'features': False
    },
    'se_resnext50': {
        'skip': resnet_layers,
        'filters': (64, 256, 512, 1024, 2048),
        'features': False
    },
    'se_resnext50_32x4d': {
        'skip': senet_layers,
        'filters': (64, 256, 512, 1024, 2048),
        'features': False
    },
    'se_resnext101_32x4d': {
        'skip': senet_layers,
        'filters': (64, 256, 512, 1024, 2048),
        'features': False
    },
    'pnasnet5large': {
        'skip': pnasnet_layers,
        'filters': (96, 270, 1080, 2160, 4320),
        'features': False
    },
    'nasnetalarge': {
        'skip': nasnet_layers,
        'filters': (96, 168, 1008, 2016, 4032),
        'features': False
    },
    'inceptionresnetv2': {
        'skip': inceptionresnetv2_layers,
        'filters': (64, 192, 320, 1088, 1536),
        'features': False
    },
    'xception': {
        'skip': xception_layers,
        'filters': (64, 128, 256, 728, 2048),
        'features': False
    },
    'inceptionv4': {
        'skip': inceptionv4_layers,
        'filters': (64, 192, 384, 1024, 1536),
        'features': True
    },
    'mobilenetv2_w1': {
        'skip': mobilenetv2_layers,
        'filters': (16, 24, 32, 96, 1280),
        'features': True
    },
    'mobilenetv2_wd2': {
        'skip': mobilenetv2_layers,
        'filters': (8, 12, 16, 48, 1280),
        'features': True
    },
    'mobilenetv2_wd4': {
        'skip': mobilenetv2_layers,
        'filters': (4, 6, 8, 24, 1280),
        'features': True
    },
    'mobilenetv2_w3d4': {
        'skip': mobilenetv2_layers,
        'filters': (12, 18, 24, 72, 1280),
        'features': True
    },
    'airnext50_32x4d_r2': {
        'skip': airnext_layers,
        'filters': (64, 256, 512, 1024, 2048),
        'features': True
    },
    'wrn50_2': {
        'skip': wrn_layers,
        'filters': (64, 256, 512, 1024, 2048),
        'features': True
    }

}
