import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import BCELoss, BCEWithLogitsLoss
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
            fp_scale: float,
            fn_scale: float,
            per_image: bool,
            ignore_class: int
    ):
        super().__init__()
        self.weight_type = weight_type
        self.alpha = alpha
        self.fp_scale = fp_scale
        self.fn_scale = fn_scale
        self.alpha = alpha
        self.per_image = per_image
        self.ignore_class = ignore_class

    def make_loss(
            self,
            preds: torch.Tensor,
            trues: torch.Tensor,
            metric_name: str
    ):
        diff_coef = 1 + F.pairwise_distance(preds, trues).mean()
        if self.weight_type == 'area':
            area = trues.shape[2] * trues.shape[2]
            weight = 1 - torch.log((torch.sum(trues.contiguous().view(preds.size(0), -1), dim=1) / area) + 1e-12)

            if metric_name == 'lovasz':
                self.bce_loss = BCEWithLogitsLoss(reduction='none')
            elif metric_name in ['dice', 'jaccard']:
                self.bce_loss = BCELoss(reduction='none')
            else:
                raise NotImplementedError(
                    f"Metric {metric_name} doesn't implemented. "
                    f"Should be 'jaccard', 'dice', 'lovasz' or None."
                )
            bce_loss = self.bce_loss(input=preds, target=trues)
            bce_loss = torch.mean(bce_loss.contiguous().view(preds.size(0), -1), dim=1)
            bce_loss = (bce_loss * weight).mean() * diff_coef
        elif self.weight_type is None:
            if metric_name == 'lovasz':
                self.bce_loss = BCEWithLogitsLoss()
            elif metric_name in ['dice', 'jaccard']:
                self.bce_loss = BCELoss()
            else:
                raise NotImplementedError(
                    f"Metric {metric_name} doesn't implemented. "
                    f"Should be 'jaccard', 'dice', 'lovasz' or None."
                )
            bce_loss = self.bce_loss(input=preds, target=trues)

        if metric_name == 'lovasz':
            if self.per_image is None:
                raise ValueError(
                    f"If metric parameter is lovasz, parameter per_image should be set."
                )

        common_parameters = dict(
            preds=preds,
            trues=trues
        )
        if metric_name == 'jaccard':
            metric_coef = jaccard_coef(
                alpha=self.fp_scale,
                beta=self.fn_scale,
                # weight=weight,
                **common_parameters
            )
        elif metric_name == 'dice':
            metric_coef = dice_coef(
                alpha=self.fp_scale,
                beta=self.fn_scale,
                # weight=weight,
                **common_parameters
            )
        elif metric_name == 'lovasz':
            metric_coef = lovasz_hinge(
                per_image=self.per_image,
                ignore=self.ignore_class,
                **common_parameters
            )
        else:
            raise NotImplementedError(
                f"Metric {metric_name} doesn't implemented. "
                f"Should be 'jaccard', 'dice', 'lovasz' or None."
            )
        if metric_name == 'lovasz':
            loss = self.alpha * bce_loss + (1 - self.alpha) * metric_coef
        else:
            loss = self.alpha * bce_loss - (1 - self.alpha) * torch.log(metric_coef)

        return loss
