import torch
from . import BaseSegmentationMetric


class MIouMetric(BaseSegmentationMetric):

    def __init__(
            self,
            mode: str,
            activation: str,
            device: str,
            threshold: float,
            per_class: bool = True,
            ignore_class: int = None
    ):
        super(MIouMetric, self).__init__(
            mode=mode,
            activation=activation,
            device=device,
            threshold=threshold,
            per_class=per_class,
            ignore_class=ignore_class
        )
        self.metric_name = 'dice'

    def calculate_metric(
            self,
            trues: torch.Tensor,
            preds: torch.Tensor,
    ):
        """ Function return dice metric for given set of true and predicted masks

        :param trues:
                Array of true masks with shape [N, C, H, W] where:
                    N - number of images in minibatch
                    C - number of channels
                    H - height
                    W - width
        :param preds:
                Array of predicted masks with shape [N, C, H, W] where:
                    N - number of images in minibatch
                    C - number of channels
                    H - height
                    W - width
        :param metric_name:
                Name of the metric.
        :param threshold:
                Threshold for binarization of predicted mask.
        :param per_class:
                Optional flag for multiclass segmentation. If true, the per class
                            metric will be calculated.
        :param ignore_class:
                Optional flag for multiclass segmentation. Show the class that will not be included
                to the metric estimation.
        :return:
        """
        if self.mode == 'binary':
            metric = self._get_metric_binary(
                trues=trues,
                preds=preds,
                metric_name=self.metric_name,
                threshold=self.threshold
            )
        else:  # self.mode == 'multi':
            metric = self._get_metric_multi(
                trues=trues,
                preds=preds,
                metric_name=self.metric_name,
                threshold=self.threshold,
                per_class=self.per_class,
                ignore_class=self.ignore_class
            )

        return metric

    def __str__(self):
        return 'm_iou'