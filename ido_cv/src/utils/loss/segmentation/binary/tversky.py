import torch
from torch import nn
from ..base.tversky_base import FocalTverskyBase


class FocalBinaryTverskyLoss(nn.Module):

    def __init__(
            self,
            alpha=0.5,
            beta=0.5,
            gamma=1.0,
            reduction='mean'
    ):
        """
        :param alpha: controls the penalty for false positives.
        :param beta: penalty for false negative.
        :param gamma : focal coefficient range[1,3]
        :param reduction: return mode
        Notes:
        alpha = beta = 0.5 => dice coeff
        alpha = beta = 1 => tanimoto coeff
        alpha + beta = 1 => F beta coeff
        add focal index -> loss=(1-T_index)**(1/gamma)
        """
        super(FocalBinaryTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, preds: torch.Tensor, trues: torch.Tensor):
        preds = torch.sigmoid(preds)
        loss_func = FocalTverskyBase(
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            reduction=self.reduction
        )
        loss = loss_func(preds, trues)
        return loss