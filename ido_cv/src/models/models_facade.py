# -*- coding: utf-8 -*-
"""
    Facade for all models classes

"""
from . import unet_factory
from . import deeplabv3
from . import fpn_factory
from . import classification_factory
from . import retinanet
from . import crnn

models_dict = {
    'classification': {
        'basic_model': classification_factory.ClassifierFactory
    },
    'segmentation': {
        'unet': unet_factory.UnetFactory,
        'fpn': fpn_factory.FPNFactory,
        'deeplabv3': deeplabv3.DeepLabV3
    },
    'detection': {
        'retinanet': retinanet.RetinaNet
    },
    'ocr': {
        'crnn': crnn.CRNN
    }
}


class ModelsFacade:
    """
        Class realize facade pattern for all models
        Arguments:
            task:           Task for the model:
                                - classification
                                - segmentation
                                - detection
            model_name:     Name of the architecture for the given task. See in documentation.

    """
    def __init__(self, task: str, model_name: str):

        tasks = models_dict.keys()
        if task not in tasks:
            raise ValueError(
                f"Wrong task parameter: {task}. "
                f"Should be: {[t for t in tasks]}"
            )

        model_names = models_dict[task].keys()
        if model_name not in model_names:
            raise ValueError(
                f"Wrong task parameter: {model_name}. "
                f"For {task} should be: {[m for m in model_names]}"
            )

        self.__model_class = models_dict[task][model_name]

    @property
    def get_model(self):
        """ Metod returns model class

        :return:
        """
        model_class = self.__model_class
        return model_class
