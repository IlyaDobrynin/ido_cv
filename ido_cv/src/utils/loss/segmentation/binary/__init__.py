import torch
from torch import nn
from ...loss_utils import get_weight
from ...loss_utils import jaccard_coef
from ...loss_utils import dice_coef
from ...loss_utils import lovasz_hinge


class BceMetricBase(nn.Module):
    """ Loss defined as (1 - alpha) * BCE - alpha * SoftJaccard
    """

    def __init__(
            self,
            weight_type: int,
            alpha: float,
            per_image: bool,
            ignore_class: int
    ):
        super().__init__()
        self.weight_type = weight_type
        self.alpha = alpha
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.per_image = per_image
        self.ignore_class = ignore_class

    def make_loss(
            self,
            preds: torch.Tensor,
            trues: torch.Tensor,
            metric_name: str,
    ):

        if metric_name == 'lovasz':
            if self.per_image is None:
                raise ValueError(
                    f"If metric parameter is lovasz, parameter per_image should be set."
                )

        metric_target = (trues == 1).float()
        metric_output = torch.sigmoid(preds)

        # Weight estimation
        if self.weight_type:
            weights = get_weight(trues=trues, weight_type=self.weight_type)
        else:
            weights = None

        bce_loss = self.bce_loss(preds, trues)
        if metric_name:
            if metric_name == 'jaccard':
                metric_coef = jaccard_coef(metric_target, metric_output, weight=weights)
            elif metric_name == 'dice':
                metric_coef = dice_coef(metric_target, metric_output, weight=weights)
            elif metric_name == 'lovasz':
                metric_coef = lovasz_hinge(
                    metric_output, metric_target, self.per_image, self.ignore_class
                )
            else:
                raise NotImplementedError(
                    f"Metric {metric_name} doesn't implemented. "
                    f"Should be 'jaccard', 'dice', 'lovasz' or None."
                )
            if metric_name == 'lovasz':
                loss = metric_coef
            else:
                loss = self.alpha * bce_loss - (1 - self.alpha) * torch.log(metric_coef)
                # loss = self.alpha * bce_loss + (1 - self.alpha) * (1 - metric_coef)
        else:
            raise ValueError(
                f"Wrong metric_name: {metric_name}. "
                f"Should be 'dice', 'jaccard' or 'lovasz'."
            )

        return loss
