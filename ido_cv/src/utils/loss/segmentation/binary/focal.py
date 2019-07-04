from functools import partial
import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from ...loss_utils import reduced_focal_loss, sigmoid_focal_loss


class BinaryRobustFocalLoss2d(_Loss):
    def __init__(
        self,
        ignore: int = None,
        reduced: bool = False,
        gamma: float = 2.0,
        alpha: float = 0.25,
        threshold: float = 0.5,
        reduction: str = "mean",
    ):
        """
        Compute focal loss for binary classification problem.
        """
        super().__init__()
        self.ignore = ignore

        if reduced:
            self.loss_fn = partial(
                reduced_focal_loss,
                gamma=gamma,
                threshold=threshold,
                reduction=reduction
            )
        else:
            self.loss_fn = partial(
                sigmoid_focal_loss,
                gamma=gamma,
                alpha=alpha,
                reduction=reduction
            )

    def forward(
            self,
            preds: torch.Tensor,
            trues: torch.Tensor
    ):
        """
        Args:
            preds: [bs; ...]
            trues: [bs; ...]
        """
        targets = trues.view(-1)
        logits = preds.view(-1)

        if self.ignore is not None:
            # Filter predictions with ignore label from loss computation
            not_ignored = targets != self.ignore
            logits = logits[not_ignored]
            targets = targets[not_ignored]

        loss = self.loss_fn(logits, targets)

        return loss