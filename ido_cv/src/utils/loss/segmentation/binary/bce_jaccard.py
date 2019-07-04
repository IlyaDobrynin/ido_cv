import torch
from . import BceMetricBase


class BCEJaccard(BceMetricBase):

    def __init__(
            self,
            weight_type: int = None,
            alpha: float = 0.3,
            per_image: bool = None,
            ignore_class: int = None
    ):
        super(BCEJaccard, self).__init__(
            weight_type=weight_type,
            alpha=alpha,
            per_image=per_image,
            ignore_class=ignore_class
        )

        self.metric_name = 'jaccard'

    def __call__(
            self,
            preds: torch.Tensor,
            trues: torch.Tensor,
    ):
        loss = self.make_loss(
            preds=preds,
            trues=trues,
            metric_name=self.metric_name
        )
        return loss
