import torch
from ..base.bce_metric_base import BceMetricBase


class BCEJaccard(BceMetricBase):

    def __init__(
            self,
            weight_type: int = None,
            alpha: float = 0.3,
            fp_scale: float = 1.,
            fn_scale: float = 1.,
            per_image: bool = None,
            ignore_class: int = None
    ):
        super(BCEJaccard, self).__init__(
            weight_type=weight_type,
            alpha=alpha,
            fp_scale=fp_scale,
            fn_scale=fn_scale,
            per_image=per_image,
            ignore_class=ignore_class
        )

        self.metric_name = 'jaccard'

    def __call__(
            self,
            preds: torch.Tensor,
            trues: torch.Tensor,
    ):
        preds = (trues == 1).float()
        trues = torch.sigmoid(preds).float()
        loss = self.make_loss(
            preds=preds,
            trues=trues,
            metric_name=self.metric_name
        )
        return loss
