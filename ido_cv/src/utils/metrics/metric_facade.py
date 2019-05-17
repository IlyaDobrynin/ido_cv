# -*- coding: utf-8 -*-
"""
    Facade for all models classes

"""
from . import classification_metrics
from . import segmentation_metrics
from . import detection_metrics


# ToDo: implement detection metrics
metrics_tasks = {
    'classification': classification_metrics.ClassificationMetrics,
    'segmentation': segmentation_metrics.SegmentationMetrics,
    'detection': None
}


class MetricsFacade:
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
    def __init__(self, task: str):

        if task not in metrics_tasks:
            raise ValueError(
                f"Wrong task parameter: {task}. "
                f"Should be: {[t for t in metrics_tasks]}"
            )

        self.__metric_class = metrics_tasks[task]


    @property
    def get_metric(self):
        """ Metod returns model class

        :return:
        """
        metrics_class = self.__metric_class
        return metrics_class


if __name__ == '__main__':
    facade_class = MetricsFacade(task='segmentation')
