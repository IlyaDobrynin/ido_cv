import torch
from torch import nn
from torch.nn import functional as F


class NllLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def __call__(self, preds, trues):
        preds = torch.softmax(preds, dim=-1)
        trues = torch.squeeze(trues, dim=-1).long()
        # print('Multi loss:', preds, trues)

        loss = F.nll_loss(preds, trues)
        return loss