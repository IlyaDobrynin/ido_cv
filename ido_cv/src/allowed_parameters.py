from .models.detection.retinanet import RetinaNet
from .models.segmentation.unet_factory import UnetFactory
from .models.segmentation.fpn_factory import FPNFactory
from .models.segmentation.deeplabv3 import DeepLabV3
from .models.classification.classification_factory import ClassifierFactory
from .utils.loss.detection_losses import FocalLoss
from .utils.loss.segmentation_losses import BinaryBceMetric
from .utils.loss.segmentation_losses import MultiBceMetric
from .utils.loss.segmentation_losses import MultiLovasz
from .utils.loss.classification_losses import BCELoss
from .utils.loss.classification_losses import NllLoss, CELoss
from .utils.metrics.segmentation_metrics import SegmentationMetrics
from .utils.metrics.classification_metrics import ClassificationMetrics
from .utils.metrics import classification_metrics as cls_metric
from .utils import tta


LOSSES = {
    'segmentation': {
        'binary': {
            "bce_jaccard": BinaryBceMetric(metric='jaccard', weight_type=None, alpha=0.7),
            "bce_dice": BinaryBceMetric(metric='dice', weight_type=None, alpha=0.7),
            "bce_lovasz": BinaryBceMetric(metric='lovasz', weight_type=None, alpha=0.7, per_image=True)
        },
        'multi': {
            "bce_jaccard": MultiBceMetric(num_classes=11, metric='jaccard'),
            "bce_dice": MultiBceMetric(num_classes=11, metric='dice'),
            "lovasz": MultiLovasz(ignore=0)
        }
    },
    'detection': {
        'all': {
            'focal_loss': FocalLoss()
        }

    },
    'classification': {
        'binary': {
            'bce': BCELoss()
        },
        'multi': {
            'nll': NllLoss(),
            'ce': CELoss()
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

METRICS = {
    'segmentation': SegmentationMetrics,
    'detection': {
        'all': {
            'map': None,
        }
    },
    'classification': ClassificationMetrics
}

MODELS = {
    'segmentation': {
        'binary': {
            'unet': {
                'class': UnetFactory,
                'default_parameters': {
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
                }
            },
            'fpn': {
                'class': FPNFactory,
                'default_parameters': {
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
                }
            },
            'deeplabv3': {
                'class': DeepLabV3,
                'default_parameters': {
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
            }
        },
        'multi': {
            'unet': {
                'class': UnetFactory,
                'default_parameters': {
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
                    'se_decoder': False,
                }
            },
            'fpn': {
                'class': FPNFactory,
                'default_parameters': {
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
                }
            },
            'deeplabv3': {
                'class': DeepLabV3,
                'default_parameters': {
                    'backbone': 'dilated_resnet34',
                    'num_classes': 11,
                    'pretrained': 'imagenet',
                    'unfreeze_encoder': True,
                    'num_input_channels': 3,
                    'bn_type': 'default',
                    'conv_type': 'default',
                    'residual': True,
                    'se_decoder': True,
                }
            }
        }
    },
    'detection': {
        'all': {
            'RetinaNet': {
                'class': RetinaNet,
                'default_parameters': {
                    'backbone': 'resnet34',
                    'se_block': False,
                    'residual': True,
                }
            }
        },
    },
    'classification': {
        'binary': {
            'basic_model': {
                'class': ClassifierFactory,
                'default_parameters': {
                    'backbone': 'resnet34',
                    'num_classes': 1,
                    'pretrained': 'imagenet',
                    'unfreeze_encoder': True,
                }
            }
        },
        'multi': {
            'basic_model': {
                'class': ClassifierFactory,
                'default_parameters': {
                    'backbone': 'resnet34',
                    'num_classes': 5,
                    'pretrained': 'imagenet',
                    'unfreeze_encoder': True,
                }
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