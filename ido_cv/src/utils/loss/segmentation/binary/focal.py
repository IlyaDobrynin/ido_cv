from functools import partial
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss
from ...loss_utils import reduced_focal_loss, sigmoid_focal_loss


# class BinaryRobustFocalLoss2d(_Loss):
#     def __init__(
#         self,
#         ignore: int = None,
#         reduced: bool = False,
#         gamma: float = 2.0,
#         alpha: float = 0.25,
#         threshold: float = 0.5,
#         reduction: str = "mean",
#     ):
#         """
#         Compute focal loss for binary classification problem.
#         """
#         super().__init__()
#         self.ignore = ignore
#
#         if reduced:
#             self.loss_fn = partial(
#                 reduced_focal_loss,
#                 gamma=gamma,
#                 threshold=threshold,
#                 reduction=reduction
#             )
#         else:
#             self.loss_fn = partial(
#                 sigmoid_focal_loss,
#                 gamma=gamma,
#                 alpha=alpha,
#                 reduction=reduction
#             )
#
#     def forward(
#             self,
#             preds: torch.Tensor,
#             trues: torch.Tensor
#     ):
#         """
#         Args:
#             preds: [bs; ...]
#             trues: [bs; ...]
#         """
#         targets = trues.view(-1)
#         logits = preds.view(-1)
#
#         if self.ignore is not None:
#             # Filter predictions with ignore label from loss computation
#             not_ignored = targets != self.ignore
#             logits = logits[not_ignored]
#             targets = targets[not_ignored]
#
#         loss = self.loss_fn(logits, targets)
#
#         return loss

class FocalLoss2d(nn.Module):

    def __init__(
            self,
            gamma: int = 2,
            weight=None,
            size_average: bool = True
    ):
        super(FocalLoss2d, self).__init__()

        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average

    def forward(
            self,
            preds: torch.Tensor,
            trues: torch.Tensor
    ):
        if preds.dim() > 2:
            preds = preds.contiguous().view(preds.size(0), preds.size(1), -1)
            preds = preds.transpose(1, 2)
            preds = preds.contiguous().view(-1, preds.size(2)).squeeze()
        if trues.dim() == 4:
            trues = trues.contiguous().view(trues.size(0), trues.size(1), -1)
            trues = trues.transpose(1,2)
            trues = trues.contiguous().view(-1, trues.size(2)).squeeze()
        elif trues.dim() == 3:
            trues = trues.view(-1)
        else:
            trues = trues.view(-1, 1)

        # compute the negative likelyhood
        # weight = Variable(self.weight)
        bce_loss = nn.BCEWithLogitsLoss()
        logpt = -bce_loss(preds, trues)
        pt = torch.exp(logpt)

        # compute the loss
        loss = -((1 - pt) ** self.gamma) * logpt

        # averaging (or not) loss
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()