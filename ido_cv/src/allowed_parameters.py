from .models.detection.retinanet import RetinaNet
from .models.segmentation.unet_factory import UnetFactory
from .models.segmentation.deeplabv3 import DeepLabV3
from .models.classification.classification_factory import ClassifierFactory
from .utils.loss.detection_losses.detection_losses import FocalLoss
from .utils.loss.segmentation_losses.binary import BceMetricBinary
from .utils.loss.segmentation_losses.multiclass import BceMetricMulti
from .utils.loss.segmentation_losses.multiclass import LovaszLoss
from .utils.loss.classification_losses.binary_classification import BCELoss
from .utils.loss.classification_losses.multi_classification import NllLoss, CELoss
from .utils.metrics import segmentation_metrics as seg_metric
from .utils.metrics import classification_metrics as cls_metric
from .utils import tta


LOSSES = {
    'segmentation': {
        'binary': {
            "bce_jaccard": BceMetricBinary(metric='jaccard', weight_type=None, alpha=0.7),
            "bce_dice": BceMetricBinary(metric='dice', weight_type=None, alpha=0.7)
        },
        'multi': {
            "bce_jaccard": BceMetricMulti(num_classes=11, metric='jaccard'),
            "bce_dice": BceMetricMulti(num_classes=11, metric='dice'),
            "lovasz": LovaszLoss(ignore=0)
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

METRICS = {
    'segmentation': {
        'binary': {
            'dice': seg_metric.get_metric,
            'jaccard': seg_metric.get_metric,
            'm_iou': seg_metric.get_metric
        },
        'multi': {
            'jaccard': seg_metric.get_metric_multi,
            'dice': seg_metric.get_metric_multi
        }
    },
    'detection': {
        'all': {
            'map': None,
        }
    },
    'classification': {
        'binary': {
            'accuracy': cls_metric.accuracy
        },
        'multi': {
            'accuracy': cls_metric.multi_accuracy
        }
    }
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
                    'num_input_channels': 3,
                    'dropout_rate': 0.2,
                    'bn_type': 'default',
                    'conv_type': 'default',
                    'depthwise': False,
                    'residual': True,
                    'mid_block': None,
                    'dilate_depth': 1,
                    'gau': False,
                    'hypercolumn': True,
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
                    'num_input_channels': 3,
                    'dropout_rate': 0.2,
                    'bn_type': 'default',
                    'conv_type': 'default',
                    'depthwise': False,
                    'residual': True,
                    'mid_block': None,
                    'dilate_depth': 1,
                    'gau': False,
                    'hypercolumn': True,
                    'se_decoder': False,
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