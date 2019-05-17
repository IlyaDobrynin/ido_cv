from .utils import tta


TASKS_MODES = {
    'segmentation': {
        'binary': [
            'unet',
            'fpn',
            'deeplabv3'
        ],
        'multi': [
            'unet',
            'fpn',
            'deeplabv3'
        ],
    },
    'detection': {
        'all': 'retinanet'
    },
    'classification': {
        'binary': ['basic_model'],
        'multi': ['basic_model']
    }
}


LOSS_PARAMETERS = {
    'segmentation': {
        'binary': {
            "bce_jaccard": dict(
                metric='jaccard',
                weight_type=None,
                alpha=0.4
            ),
            "bce_dice": dict(
                metric='dice',
                weight_type=None,
                alpha=0.4
            )
        },
        'multi': {
            "bce_jaccard": dict(
                num_classes=11,
                metric='jaccard',
                alpha=0.3,
                class_weights=[0.1, 0.4, 0.5, 0.5, 0.2, 0.2, 0.1, 0.1, 0.2, 0.4, 0.1]
            ),
            "bce_dice": dict(
                num_classes=11,
                metric='dice',
                alpha=0.3,
                class_weights=[0.1, 0.8, 0.8, 0.8, 0.2, 0.2, 0.1, 0.1, 0.2, 0.2, 0.1]
            ),
            "lovasz": dict(
                ignore=0
            )
        }
    },
    'detection': {
        'all': {
            'focal_loss': dict()
        }

    },
    'classification': {
        'binary': {
            'bce': dict()
        },
        'multi': {
            'nll': dict(),
            'ce': dict()
        }
    }
}

METRIC_NAMES = {
    'segmentation': {
        'binary': ['dice', 'jaccard', 'm_iou'],
        'multi': ['dice', 'jaccard', 'm_iou'],
    },
    'detection': {
        'all': ['map']
    },
    'classification': {
        'binary': ['accuracy'],
        'multi': ['accuracy']
    }
}

# ToDo add detection metrics
METRIC_PARAMETERS = {
    'segmentation': {
        'binary': dict(
            activation='sigmoid',
            device='cpu'
        ),
        'multi': dict(
            activation='softmax',
            device='cpu'
        )
    },
    'detection': {
        'all': None
    },
    'classification': {
        'binary': dict(activation='sigmoid'),
        'multi': dict(activation='softmax')
    }
}

MODEL_PARAMETERS = {
    'segmentation': {
        'binary': {
            'unet': {
                'backbone': 'resnet34',
                'depth': 4,
                'num_classes': 1,
                'num_filters': 32,
                'pretrained': 'imagenet',
                'unfreeze_encoder': True,
                'custom_enc_start': False,
                'num_input_channels': 3,
                'dropout_rate': 0.2,
                'bn_type': 'default',
                'conv_type': 'default',
                'upscale_mode': 'nearest',
                'depthwise': False,
                'residual': True,
                'mid_block': None,
                'dilate_depth': 1,
                'gau': False,
                'hypercolumn': True,
                'se_decoder': True
            },
            'fpn': {
                'backbone': 'resnet34',
                'depth': 4,
                'num_classes': 1,
                'num_filters': 32,
                'pretrained': 'imagenet',
                'unfreeze_encoder': True,
                'custom_enc_start': False,
                'num_input_channels': 3,
                'dropout_rate': 0.2,
                'upscale_mode': 'nearest',
                'depthwise': False,
                'bn_type': 'default',
                'conv_type': 'default',
                'residual': True,
                'gau': False,
                'se_decoder': True
            },
            'deeplabv3': {
                    'backbone': 'dilated_resnet34',
                    'num_classes': 1,
                    'pretrained': 'imagenet',
                    'unfreeze_encoder': True,
                    'num_input_channels': 3,
                    'bn_type': 'default',
                    'conv_type': 'default',
                    'residual': True,
                    'se_decoder': True
            }
        },
        'multi': {
            'unet': {
                    'backbone': 'resnet34',
                    'depth': 4,
                    'num_classes': 11,
                    'num_filters': 32,
                    'pretrained': 'imagenet',
                    'unfreeze_encoder': True,
                    'custom_enc_start': False,
                    'num_input_channels': 3,
                    'dropout_rate': 0.2,
                    'bn_type': 'default',
                    'conv_type': 'default',
                    'upscale_mode': 'nearest',
                    'depthwise': False,
                    'residual': True,
                    'mid_block': None,
                    'dilate_depth': 1,
                    'gau': False,
                    'hypercolumn': True,
                    'se_decoder': False
            },
            'fpn': {
                    'backbone': 'resnet34',
                    'depth': 4,
                    'num_classes': 11,
                    'num_filters': 32,
                    'pretrained': 'imagenet',
                    'unfreeze_encoder': True,
                    'custom_enc_start': False,
                    'num_input_channels': 3,
                    'dropout_rate': 0.2,
                    'upscale_mode': 'nearest',
                    'depthwise': False,
                    'bn_type': 'default',
                    'conv_type': 'default',
                    'residual': True,
                    'gau': False,
                    'se_decoder': True
            },
            'deeplabv3': {
                    'backbone': 'dilated_resnet34',
                    'num_classes': 11,
                    'pretrained': 'imagenet',
                    'unfreeze_encoder': True,
                    'num_input_channels': 3,
                    'bn_type': 'default',
                    'conv_type': 'default',
                    'residual': True,
                    'se_decoder': True
            },
        },
    },
    'detection': {
        'all': {
            'RetinaNet': {
                'backbone': 'resnet34',
                'se_block': False,
                'residual': True
            }
        }
    },
    'classification': {
        'binary': {
            'basic_model': {
                    'backbone': 'resnet34',
                    'num_classes': 1,
                    'pretrained': 'imagenet',
                    'unfreeze_encoder': True
            }
        },
        'multi': {
            'basic_model': {
                    'backbone': 'resnet34',
                    'num_classes': 5,
                    'pretrained': 'imagenet',
                    'unfreeze_encoder': True
            }
        }
    }
}

OPTIMIZERS = [
    'adam',
    'sgd',
    'rmsprop'
]

CHECKPOINT_METRICS = {
    'segmentation': {
        'binary': ['loss', 'dice', 'jaccard'],
        'multi': ['loss', 'dice', 'jaccard']
    },
    'detection': {
        'all': ['loss']
    },
    'classification': {
        'binary': ['loss', 'accuracy'],
        'multi': ['loss', 'accuracy']
    }

}

TTA = {
    'segmentation': {
        'binary': {
            'nothing': tta.Nothing,
            'h_flip': tta.HFlip,
            'v_flip': tta.VFlip,
            'hv_flip': tta.HVFlip
        },
        'multi': {
            'nothing': tta.Nothing,
            'h_flip': tta.HFlip,
            'v_flip': tta.VFlip,
            'hv_flip': tta.HVFlip
        }
    },
    'detection': {
    
    },
    'classification': {
    
    }
}

SEG_MULTI_COLORS = {
    0: [0, 0, 250],             #  0 digit
    1: [50, 0, 0],              #  1 digit
    2: [0, 50, 0],              #  2 digit
    3: [0, 0, 50],              #  3 digit
    4: [50, 50, 0],             #  4 digit
    5: [50, 150, 0],            #  5 digit
    6: [150, 50, 0],            #  6 digit
    7: [50, 0, 50],             #  7 digit
    8: [50, 150, 250],          #  8 digit
    9: [0, 100, 250],           #  9 digit
    10: [0, 100, 50],           #  Handlabelled word
}