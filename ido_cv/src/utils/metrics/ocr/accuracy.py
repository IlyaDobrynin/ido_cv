import torch
import torch.nn.functional as F
from ...ocr_utils import LabelConverter
from . import BaseOCRMetric


class Accuracy(BaseOCRMetric):
    def __init__(
            self,
            mode: str,
            alphabet: str = None,
            ignore_case: bool = False
    ):
        super(Accuracy, self).__init__(
            mode=mode,
            alphabet=alphabet
        )
        self.converter = LabelConverter(
            alphabet=alphabet,
            ignore_case=ignore_case
        )
        self.metric_name = 'accuracy'

    def calculate_metric(
            self,
            trues: torch.Tensor,
            preds: torch.Tensor
    ):
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

        if self.metric_name == 'accuracy':
            metric = self.get_metric_value(
                trues=trues_,
                preds=preds_,
                metric_name=self.metric_name
            )
        else:
            raise ValueError(
                f"Wrong parameter metric_name: {self.metric_name}. "
                f"Should be 'accuracy'."
            )
        return metric

    def __str__(self):
        return 'accuracy'
