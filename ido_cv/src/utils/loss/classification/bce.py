import torch
from torch import nn
from torch.nn import functional as F


class BCELoss(nn.Module):

    def __init__(self):
        super().__init__()

    def __call__(self, preds, trues):
        preds = torch.sigmoid(preds)
        loss = F.binary_cross_entropy(preds, trues)
        return loss