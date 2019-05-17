# -*- coding: utf-8 -*-
"""
    Facade for all models classes

"""
from . import classification_dataset
from . import detection_dataset
from . import segmentation_dataset


class DatasetFacade:
    """
        Class realize facade pattern for all datasets
        Arguments:
            task:           Task for the model:
                                - classification
                                - segmentation
                                - detection
            mode:     Mode of the task (only for segmentation:
                                - binary
                                - multi
    """
    def __init__(self, task: str, mode: str = None):

        if task == 'classification':
            self.__dataset_class = classification_dataset.ClassifyDataset
        elif task == 'segmentation':
            if mode == 'binary':
                self.__dataset_class = segmentation_dataset.BinSegDataset
            elif mode == 'multi':
                self.__dataset_class = segmentation_dataset.MultSegDataset
            else:
                raise ValueError(
                    f"Wrong parameter mode: {mode}. "
                    f"For segmentation should be 'binary' or 'multi'."
                )
        elif task == 'detection':
            self.__dataset_class = detection_dataset.RetinaDataset
        else:
            raise ValueError(
                f"Wrong parameter task: {task}. "
                f"Should be 'classification', 'segmentation' or 'detection'."
            )

    @property
    def get_dataset_class(self):
        """ Metod returns model class

        :return:
        """
        dataset_class = self.__dataset_class
        return dataset_class
