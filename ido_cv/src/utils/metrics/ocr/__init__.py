from ..abstract_metric import AbstractMetric


class BaseOCRMetric(AbstractMetric):

    def __init__(
            self,
            mode: str,
            alphabet: str
    ):
        assert mode in ['all'], f'Wrong mode parameter: {mode}. ' \
            f'Should be "all".'
        assert type(alphabet) == str, f'Wrong alphabet: {alphabet}. Should be str.'

    @staticmethod
    def get_metric_value(trues: list, preds: list, metric_name: str):
        if metric_name == 'accuracy':
            num_correct = 0
            for pred, true in zip(preds, trues):
                if pred == true:
                    num_correct += 1
            metric = (num_correct / len(trues))
        else:
            raise ValueError(
                f'Wrong metric_name: {metric_name}.'
                f' Should be "accuracy"'
            )
        return metric