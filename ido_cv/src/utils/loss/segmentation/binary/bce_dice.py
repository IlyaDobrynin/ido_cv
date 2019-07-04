from . import BceMetricBase
import torch


class BCEDice(BceMetricBase):

    def __init__(
            self,
            weight_type: int = None,
            alpha: float = 0.3,
            per_image: bool = None,
            ignore_class: int = None
    ):
        super(BCEDice, self).__init__(
            weight_type=weight_type,
            alpha=alpha,
            per_image=per_image,
            ignore_class=ignore_class
        )

        self.metric_name = 'dice'

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
