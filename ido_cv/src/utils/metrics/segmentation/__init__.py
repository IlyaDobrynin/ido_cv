
import torch
import numpy as np
from ..abstract_metric import AbstractMetric
from ..metric_utils import numpy_metric_per_image
from ..metric_utils import torch_metric
from ..metric_utils import calculate_confusion_matrix_from_arrays
from ..metric_utils import get_metric_from_matrix
#
# from ido_cv.ido_cv.src.utils.metrics.abstract_metric import AbstractMetric
# from ido_cv.ido_cv.src.utils.metrics.metric_utils import numpy_metric_per_image
# from ido_cv.ido_cv.src.utils.metrics.metric_utils import torch_metric
# from ido_cv.ido_cv.src.utils.metrics.metric_utils import calculate_confusion_matrix_from_arrays
# from ido_cv.ido_cv.src.utils.metrics.metric_utils import get_metric_from_matrix


class BaseSegmentationMetric(AbstractMetric):

    def __init__(
            self,
            mode: str,
            activation: str,
            device: str,
            threshold: float,
            per_class: bool = True,
            ignore_class: int = None
    ):
        # Assertions
        assert mode in ['binary', 'multi'], f'Wrong mode parameter: {mode}. ' \
            f'Should be binary of multi.'
        assert activation in ['sigmoid', 'softmax'], f'Wrong activation parameter: {activation}.' \
            f'Should be softmax or sigmoid.'
        assert device in ['cpu', 'gpu'], f'Wrong device parameter: {device}.' \
            f'Should be cpu or gpu.'

        self.mode = mode
        self.activation = activation
        self.device = device
        self.threshold = threshold
        self.per_class = per_class
        self.ignore_class = ignore_class

    def _get_metric_binary(
            self,
            trues: torch.Tensor,
            preds: torch.Tensor,
            metric_name: str,
            threshold: float
    ) -> float:
        """ Metric for binary segmentation

        :param metric_name:
        :param threshold:
        :return:
        """
        if self.activation != 'sigmoid':
            raise ValueError(
                f'Activation for binary metric should be "sigmoid", not {self.activation}'
            )
        trues_ = torch.squeeze(trues, dim=1)
        preds_ = torch.squeeze(torch.sigmoid(preds), dim=1)
        metric = self._get_metric_value(
            trues=trues_,
            preds=preds_,
            metric_name=metric_name,
            threshold=threshold,
            device=self.device
        )
        return metric

    def _get_metric_multi(
            self,
            trues: torch.Tensor,
            preds: torch.Tensor,
            threshold: float,
            metric_name: str,
            per_class: bool,
            ignore_class: int
    ):
        if per_class:
            metric = self._get_metric_multi_per_class(
                trues=trues,
                preds=preds,
                metric_name=metric_name,
                threshold=threshold,
                ignore_class=ignore_class
            )
        else:
            metric = self._get_metric_multi_confusion(
                trues=trues,
                preds=preds,
                metric_name=metric_name,
                ignore_class=ignore_class
            )
        return metric

    def _get_metric_multi_per_class(
            self,
            trues:          torch.Tensor,
            preds:          torch.Tensor,
            metric_name:    str,
            threshold:      float,
            ignore_class:   int = None
    ) -> np.ndarray:
        """ Alternative metric for multiclass segmentation

        :param metric_name: Name of the desired metric
        :param threshold: Class-vised threshold
        :param ignore_class: Index of class to ignore
        :return:
        """
        mul_metrics = []
        trues_ = torch.squeeze(trues, dim=1)

        if self.activation == 'softmax':
            preds_ = torch.softmax(preds, dim=1).float()
        elif self.activation == 'sigmoid':
            preds_ = torch.sigmoid(preds).float()
        else:
            raise ValueError(
                f'Activation for binary metric should be "sigmoid" or "softmax", '
                f'not {self.activation}'
            )
        for cls in range(preds_.shape[1]):
            if cls == ignore_class:
                continue
            cls_trues = trues_ == cls
            cls_preds = preds_[:, cls, :, :]

            cls_metric = self._get_metric_value(
                trues=cls_trues, preds=cls_preds, metric_name=metric_name,
                threshold=threshold, device=self.device
            )
            mul_metrics.append(cls_metric)

        mul_metrics = [x for x in mul_metrics if str(x) != 'nan']
        return np.mean(mul_metrics)

    def _get_metric_multi_confusion(self, trues: torch.Tensor, preds: torch.Tensor,
                                    metric_name: str, ignore_class: int) -> np.ndarray:
        """ Function to make multiclass metric (jaccard, dice, mean IoU)

        :param metric_name:
        :param ignore_class:
        :return:
        """
        assert self.device == 'cpu', f'Wrong device parameter: {self.device}. ' \
            f'Only cpu allowed if per_class = False'

        if self.activation == 'softmax':
            preds_ = torch.softmax(preds, dim=1).data.cpu().numpy().astype(np.float32)

        else:
            raise ValueError(
                f'if per_class = False, activation should be "softmax", not {self.activation}'
            )
        preds_ = np.argmax(preds_, axis=1)
        trues_ = np.squeeze(trues.data.cpu().numpy(), axis=1)

        confusion_matrix = calculate_confusion_matrix_from_arrays(
            preds_, trues_, preds.shape[1]
        )
        # print(confusion_matrix)
        if ignore_class is None:
            confusion_matrix = confusion_matrix[1:, 1:]
        metric = get_metric_from_matrix(confusion_matrix, metric_name, ignore_class)
        return np.mean(metric)

    @staticmethod
    def _get_metric_value(trues: torch.Tensor, preds: torch.Tensor, metric_name: str,
                          threshold: float, device: str) -> float:
        trues = trues.long()
        preds = (preds > threshold).long()

        if device == 'cpu':
            trues = trues.data.cpu().numpy().astype(np.uint8)
            preds = preds.data.cpu().numpy().astype(np.uint8)
            # metric = numpy_metric(trues=trues, preds=preds, metric_name=metric_name)
            metric = numpy_metric_per_image(
                trues=trues,
                preds=preds,
                metric_name=metric_name,
            )
        elif device == 'gpu':
            metric = torch_metric(trues=trues, preds=preds, metric_name=metric_name)
            # metric = torch_metric_per_image(trues=trues, preds=preds, metric_name=metric_name)
        else:
            raise ValueError(
                f'Wrong device parameter: {device}. Should be "cpu" or "gpu".'
            )
        return metric
