# -*- coding: utf-8 -*-
"""
    Facade for all models classes

"""
from .abstract_metric import AbstractMetric
from .segmentation.dice import DiceMetric
from .segmentation.jaccard import JaccardMetric
from .segmentation.m_iou import MIouMetric
from .ocr.accuracy import Accuracy as OCRAccuracy
from .classification.accuracy import Accuracy as ClassifyAccuracy


metrics = {
    'classification': {
        'accuracy': ClassifyAccuracy
    },
    'segmentation': {
        'dice': DiceMetric,
        'jaccard': JaccardMetric,
        'm_iou': MIouMetric
    },
    'detection': {
        # ToDo: implement detection metrics
    },
    'ocr': {
        'accuracy': OCRAccuracy
    }
}


class MetricFacade:
    """
        Class realize facade pattern for all metrics
        Arguments:
            task:           Task for the model:
                                - classification
                                - segmentation
                                - detection
            mode:           Mode of training
            loss_name:      Name of the architecture for the given task. See in documentation.

    """
    def __init__(
            self,
            task: str
    ):

        if task not in metrics.keys():
            raise ValueError(
                f"Wrong task parameter: {task}. "
                f"Should be: {[t for t in metrics.keys()]}"
            )
        self.task = task

    def get_metric_class(
            self,
            metric_definition: (str, callable)
    ):
        """ Metod returns model class

        :return:
        """
        if isinstance(metric_definition, str) and metric_definition in metrics[self.task]:
            metrics_class = metrics[self.task][metric_definition]
        elif isinstance(metric_definition, AbstractMetric):
            metrics_class = metric_definition
        else:
            raise ValueError(
                f"Wrong metric_definition parameter: {metric_definition}. "
                f"Should be string or an instance of ido_cv.src.utils.metrics.AbstractMetric."
            )
        return metrics_class


if __name__ == '__main__':
    facade_class = MetricFacade(task='segmentation')
