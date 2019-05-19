# -*- coding: utf-8 -*-
"""
    Facade for all models classes

"""
from . import unet_factory
from . import deeplabv3
from . import fpn_factory
from . import classification_factory
from . import retinanet


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

        if task == 'classification':
            if model_name == 'basic_model':
                self.__model_class = classification_factory.ClassifierFactory
            else:
                raise ValueError(
                    f"Wrong parameter model_name: {model_name}. "
                    f"Should be 'basic_model'."
                )
        elif task == 'segmentation':
            if model_name == 'unet':
                self.__model_class = unet_factory.UnetFactory
            elif model_name == 'fpn':
                self.__model_class = fpn_factory.FPNFactory
            elif model_name == 'deeplabv3':
                self.__model_class = deeplabv3.DeepLabV3
            else:
                raise ValueError(
                    f"Wrong parameter model_name: {model_name}. "
                    f"Should be 'unet', 'fpn' or 'deeplabv3'."
                )
        elif task == 'detection':
            if model_name in ['retinanet']:
                self.__model_class = retinanet.RetinaNet
            else:
                raise ValueError(
                    f"Wrong parameter model_name: {model_name}. "
                    f"Should be 'retinanet'."
                )
        else:
            raise ValueError(
                f"Wrong parameter task: {task}. "
                f"Should be 'classification', 'segmentation' or 'detection'."
            )

    @property
    def get_model(self):
        """ Metod returns model class

        :return:
        """
        model_class = self.__model_class
        return model_class
