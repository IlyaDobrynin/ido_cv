# -*- coding: utf-8 -*-
"""
    Facade for losses

"""
from . import classification_losses
from . import segmentation_losses
from . import detection_losses
from . import ocr_losses

losses_dict = {
    'segmentation': {
        'binary': {
            "bce_jaccard": segmentation_losses.BinaryBceMetric,
            "bce_dice": segmentation_losses.BinaryBceMetric,
            "bce_lovasz": segmentation_losses.BinaryBceMetric
        },
        'multi': {
            "bce_jaccard": segmentation_losses.MultiBceMetric,
            "bce_dice": segmentation_losses.MultiBceMetric,
            "lovasz": segmentation_losses.MultiLovasz
        }
    },
    'detection': {
        'all': {
            'focal_loss': detection_losses.FocalLoss
        }

    },
    'classification': {
        'binary': {
            'bce': classification_losses.BCELoss
        },
        'multi': {
            'nll': classification_losses.NllLoss,
            'ce': classification_losses.CELoss
        }
    },
    'ocr': {
        'all': {
            'ctc': ocr_losses.CTCLoss
        }
    }
}


class LossFacade:
    """
        Class realize facade pattern for all models
        Arguments:
            task:           Task for the model:
                                - classification
                                - segmentation
                                - detection
            mode:           Mode of training
            loss_name:      Name of the architecture for the given task. See in documentation.

    """
    def __init__(self, task: str, mode: str, loss_name: str):

        tasks = losses_dict.keys()
        if task not in tasks:
            raise ValueError(
                f"Wrong task parameter: {task}. "
                f"Should be: {[t for t in tasks]}"
            )

        modes = losses_dict[task].keys()
        if mode not in modes:
            raise ValueError(
                f"Wrong task parameter: {mode}. "
                f"For {task} should be: {[m for m in modes]}"
            )

        loss_names = losses_dict[task][mode].keys()
        if loss_name not in loss_names:
            raise ValueError(
                f"Wrong task parameter: {loss_name}. "
                f"For {mode} {task} should be: {[ln for ln in loss_names]}"
            )

        self.__loss_class = losses_dict[task][mode][loss_name]


    @property
    def get_loss(self):
        """ Metod returns model class

        :return:
        """
        loss_class = self.__loss_class
        return loss_class


if __name__ == '__main__':
    facade_class = LossFacade(task='segmentation', mode='multi', loss_name='bce_jaccard')
