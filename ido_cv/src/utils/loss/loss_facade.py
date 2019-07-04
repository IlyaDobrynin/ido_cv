# -*- coding: utf-8 -*-
"""
    Facade for losses

"""
from typing import Union
from .segmentation.binary.bce_jaccard import BCEJaccard
from .segmentation.binary.bce_dice import BCEDice
from .segmentation.binary.bce_lovasz import BCELovasz
from .segmentation.binary.focal import BinaryRobustFocalLoss2d

from .segmentation.multi.bce_jaccard import MultiBCEJaccard
from .segmentation.multi.bce_dice import MultiBCEDice
from .segmentation.multi.lovasz import MultiLovasz
from .segmentation.multi.focal import MultiRobustFocalLoss2d

from .detection.focal import FocalLoss

from .classification.bce import BCELoss
from .classification.ce import CELoss
from .classification.nll import NllLoss

from .ocr.ctc import CTCLoss


from torch.nn import Module


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

    _losses_dict = {
        'segmentation': {
            'binary': {
                'bce_jaccard': BCEJaccard,
                'bce_dice': BCEDice,
                'bce_lovasz': BCELovasz,
                'focal': BinaryRobustFocalLoss2d
            },
            'multi': {
                'bce_jaccard': MultiBCEJaccard,
                'bce_dice': MultiBCEDice,
                'lovasz': MultiLovasz,
                'focal': MultiRobustFocalLoss2d
            }
        },
        'detection': {
            'all': {
                'focal_loss': FocalLoss
            }

        },
        'classification': {
            'binary': {
                'bce': BCELoss
            },
            'multi': {
                'nll': NllLoss,
                'ce': CELoss
            }
        },
        'ocr': {
            'all': {
                'ctc': CTCLoss
            }
        }
    }

    def __init__(
            self,
            task: str,
            mode: str
    ):

        tasks = self._losses_dict.keys()
        if task not in tasks:
            raise ValueError(
                f"Wrong task parameter: {task}. "
                f"Should be: {[t for t in tasks]}"
            )
        self.task = task

        modes = self._losses_dict[task].keys()
        if mode not in modes:
            raise ValueError(
                f"Wrong task parameter: {mode}. "
                f"For {task} should be: {[m for m in modes]}"
            )
        self.mode = mode

    def get_loss_class(
            self,
            loss_definition: Union[str, Module]
    ):
        """ Metod returns model class

        :return:
        """

        if isinstance(loss_definition, str) and loss_definition \
                in self._losses_dict[self.task][self.mode].keys():
            loss_class = self._losses_dict[self.task][self.mode][loss_definition]
        elif isinstance(loss_definition, Module):
            loss_class = loss_definition
        else:
            raise ValueError(
                f"Wrong metric_definition parameter: {loss_definition}. "
                f"Should be string or an instance of torch.nn.Module."
            )

        return loss_class
