import torch
from . import MultiBCEMetricBase


class MultiBCEJaccard(MultiBCEMetricBase):
    def __init__(
            self,
            alpha: float = 0.3,
            ignore_class: int = None,
            class_weights: list = None
    ):
        super(MultiBCEJaccard, self).__init__(
            alpha=alpha,
            ignore_class=ignore_class
        )

        self.class_weights = class_weights
        self.metric_name = 'jaccard'

    def __call__(
            self,
            preds: torch.Tensor,
            trues: torch.Tensor,
    ):

        loss = self.make_loss(
            preds=preds,
            trues=trues,
            metric_name=self.metric_name,
            class_weights=self.class_weights
        )
        return loss
