# -*- coding: utf-8 -*-
"""
    Facade for all models classes

"""
from . import classification_dataset
from . import detection_dataset
from . import segmentation_dataset
from . import ocr_dataset

datasets_dict = {
    'classification': {
        'binary': classification_dataset.ClassifyDataset,
        'multi': classification_dataset.ClassifyDataset
    },
    'segmentation': {
        'binary': segmentation_dataset.BinSegDataset,
        'multi': segmentation_dataset.MultSegDataset
    },
    'detection': {
        'all': detection_dataset.RetinaDataset
    },
    'ocr': {
        'all': ocr_dataset.OCRDataset
    }
}


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

        tasks = datasets_dict.keys()
        if task not in tasks:
            raise ValueError(
                f"Wrong task parameter: {task}. "
                f"Should be: {[t for t in tasks]}"
            )

        modes = datasets_dict[task].keys()
        if mode not in modes:
            raise ValueError(
                f"Wrong task parameter: {mode}. "
                f"For {task} should be: {[m for m in modes]}"
            )

        self.__dataset_class = datasets_dict[task][mode]

    @property
    def get_dataset_class(self):
        """ Metod returns model class

        :return:
        """
        dataset_class = self.__dataset_class
        return dataset_class
