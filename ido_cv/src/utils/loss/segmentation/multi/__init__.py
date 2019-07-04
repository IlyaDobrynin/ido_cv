import torch
from torch import nn
from torch.nn import functional as F
from ...loss_utils import dice_coef
from ...loss_utils import jaccard_coef


class MultiBCEMetricBase(nn.Module):
    def __init__(
            self,
            alpha: float = 0.3,
            ignore_class: int = None
    ):
        super().__init__()
        self.alpha = alpha
        self.ignore_class = ignore_class

    def make_loss(
            self,
            preds: torch.Tensor,
            trues: torch.Tensor,
            metric_name: str,
            class_weights: list
    ):
        metric_output = F.softmax(preds, dim=1)
        num_classes = metric_output.shape[1]

        if class_weights is None:
            class_weights = [1] * num_classes
            class_weights = torch.Tensor(class_weights)
        else:
            if len(class_weights) != num_classes:
                raise ValueError(
                    f"Length od class weights should be the same as num classes. "
                    f"Give: num_classes = {num_classes}, "
                    f"len(class_weights) = {len(class_weights)}."
                )
        class_weights = torch.Tensor(class_weights)

        loss = 0
        for i in range(metric_output.shape[1]):
            if i == self.ignore_class:
                continue

            cls_weight = class_weights[i]
            metric_target_cls = (trues == i).float()
            metric_output_cls = metric_output[:, i, ...].unsqueeze(1)

            bce_loss_class = nn.BCEWithLogitsLoss()
            bce_loss = bce_loss_class(
                metric_output_cls,
                metric_target_cls
            )
            if metric_name:
                if metric_name == 'jaccard':
                    metric_coef = jaccard_coef(metric_target_cls, metric_output_cls)
                elif metric_name == 'dice':
                    metric_coef = dice_coef(metric_target_cls, metric_output_cls)
                else:
                    raise NotImplementedError(
                       f"Metric {metric_name} not implemented. "
                       f"Variants: 'jaccard;, 'dice', None")
                loss += ((1 - self.alpha) * bce_loss - self.alpha * torch.log(metric_coef)) \
                        * cls_weight
            else:
                raise ValueError(
                    f"Wrong metric_name: {metric_name}. "
                    f"Should be 'dice' or 'jaccard'."
                )

        if self.ignore_class is not None:
            ignore = 1
        else:
            ignore = 0

        loss = loss / (metric_output.shape[1] - ignore)
        return loss