from ..abstract_metric import AbstractMetric
from ..metric_utils import get_accuracy


class BaseClassificationMetric(AbstractMetric):

    @staticmethod
    def get_metric_value(trues, preds, metric_name):
        if metric_name == 'accuracy':
            metric = get_accuracy(trues=trues, preds=preds)
        else:
            raise ValueError(
                f'Wrong metric_name: {metric_name}.'
                f' Should be "accuracy"'
            )
        return metric