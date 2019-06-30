from .utils import tta
import torch.optim as optimizers


TASKS_MODES = {
    'segmentation': {
        'binary':   ['unet', 'fpn', 'deeplabv3'],
        'multi':    ['unet', 'fpn', 'deeplabv3']
    },
    'detection': {
        'all':      ['retinanet']
    },
    'classification': {
        'binary':   ['basic_model'],
        'multi':    ['basic_model']
    },
    'ocr': {
        'all':      ['crnn']
    }
}


LOSS_PARAMETERS = {
    'segmentation': {
        'binary': {
            "bce_jaccard": dict(
                metric='jaccard',
                weight_type=None,
                alpha=0.7
            ),
            "bce_dice": dict(
                metric='dice',
                weight_type=None,
                alpha=0.7
            ),
            "bce_lovasz": dict(
                metric='lovasz',
                weight_type=None,
                alpha=0.4,
                per_image=True
            )
        },
        'multi': {
            "bce_jaccard": dict(
                num_classes=22,
                metric='jaccard',
                alpha=0.3,
                # class_weights=[0.1, 0.4, 0.5, 0.5, 0.2, 0.2, 0.1, 0.1, 0.2, 0.4, 0.1]
            ),
            "bce_dice": dict(
                num_classes=22,
                metric='dice',
                alpha=0.3,
                # class_weights=[0.1, 0.8, 0.8, 0.8, 0.2, 0.2, 0.1, 0.1, 0.2, 0.2, 0.1]
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
    },
    'ocr': {
        'all': {
            'ctc': dict()
        }
    }
}

METRIC_NAMES = {
    'segmentation': {
        'binary':   ['dice', 'jaccard', 'm_iou'],
        'multi':    ['dice', 'jaccard', 'm_iou'],
    },
    'detection': {
        'all':      ['map']
    },
    'classification': {
        'binary':   ['accuracy'],
        'multi':    ['accuracy']
    },
    'ocr': {
        'all':      ['accuracy']
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
    },
    'ocr': {
        'all': dict(ignore_case=False,
                    alphabet=r'°1234567890абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦ'
                             r'ЧШЩЪЫЬЭЮЯabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@"#№%$'
                             r'%^&*();:-_=+\|/?<>~`., ')
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

OPTIMIZERS = {
    'adam': optimizers.Adam,
    'rmsprop': optimizers.RMSprop,
    'sgd': optimizers.SGD
}

OPTIMIZER_PARAMETERS = {
    'adam': dict(weight_decay=0.00001),
    'sgd': dict(),
    'rmsprop': dict(nesterov=True, momentum=0.9)
}

CHECKPOINT_METRICS = {
    'segmentation': {
        'binary':   ['loss', 'dice', 'jaccard'],
        'multi':    ['loss', 'dice', 'jaccard']
    },
    'detection': {
        'all':      ['loss']
    },
    'classification': {
        'binary':   ['loss', 'accuracy'],
        'multi':    ['loss', 'accuracy']
    },
    'ocr': {
        'all':      ['loss', 'accuracy']
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
    
    },
    'ocr': {

    }
}

class ParameterBuilder:
    """ Class for building parameters for model training, validation and predict

    """

    def __init__(self, task: str, mode: str):
        self.task = task
        self.mode = mode
        self.__loss_params_dict = LOSS_PARAMETERS
        self.__metric_params_dict = METRIC_PARAMETERS
        self.__model_params_dict = MODEL_PARAMETERS
        self.__optim_params_dict = OPTIMIZER_PARAMETERS

    @property
    def get_metric_parameters(self):
        return self.__metric_params_dict[self.task][self.mode]

    def get_loss_parameters(self, loss_name: str):
        return self.__loss_params_dict[self.task][self.mode][loss_name]

    def get_model_parameters(self, model_name: str):
        return self.__model_params_dict[self.task][self.mode][model_name]

    def get_optimizer_parameters(self, optimizer_name: str):
        return self.__optim_params_dict[optimizer_name]