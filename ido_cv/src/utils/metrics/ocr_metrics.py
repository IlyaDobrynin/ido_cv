import numpy as np
import torch
import torch.nn.functional as F
from ..ocr_utils import LabelConverter


class OCRMetrics:
    def __init__(self, mode: str, ignore_case: bool = False, alphabet: str = None):
        assert mode in ['all'], f'Wrong mode parameter: {mode}. ' \
            f'Should be "all".'
        assert type(alphabet) == str, f'Wrong alphabet: {alphabet}. Should be str.'
        self.converter = LabelConverter(
            alphabet=alphabet,
            ignore_case=ignore_case
        )

    def get_metric(self, trues: torch.Tensor, preds: torch.Tensor, metric_name: str):
        """ Function return metric for given set of true and predicted masks

        :param trues:
                Array of true labels:
        :param preds:
                Array of predicted labels:
        :param metric_name:
                Name of the metric.
        :return:
        """

        log_preds = F.log_softmax(preds, dim=2)
        preds_ = self.converter.best_path_decode(log_preds, strings=False)
        true_text = trues[1].data.cpu().numpy().tolist()
        true_lengths = trues[2].data.cpu().numpy().tolist()

        trues_ = []
        start_idx = 0
        for true_length in true_lengths:
            final_idx = start_idx + true_length
            true = true_text[start_idx: final_idx]
            start_idx = final_idx
            trues_.append(true)

        if metric_name == 'accuracy':
            metric = self.get_metric_value(
                trues=trues_,
                preds=preds_,
                metric_name=metric_name
            )
        else:
            raise ValueError(
                f"Wrong parameter metric_name: {metric_name}. "
                f"Should be 'accuracy'."
            )
        return metric

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