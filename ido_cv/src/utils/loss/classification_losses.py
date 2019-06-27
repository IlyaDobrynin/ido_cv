# -*- coding: utf-8 -*-
"""
Module implements classification losses

"""

import torch
from torch import nn
from torch.nn import functional as F


##################### BINARY #####################


class BCELoss(nn.Module):

    def __init__(self):
        super().__init__()

    def __call__(self, preds, trues):
        preds = torch.sigmoid(preds)
        loss = F.binary_cross_entropy(preds, trues)
        return loss


##################### MULTICLASS ##################


class NllLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def __call__(self, preds, trues):
        preds = torch.softmax(preds, dim=-1)
        trues = torch.squeeze(trues, dim=-1).long()
        # print('Multi loss:', preds, trues)

        loss = F.nll_loss(preds, trues)
        return loss


class CELoss(nn.Module):

    def __init__(self):
        super().__init__()

    def __call__(self, preds, trues):
        preds = torch.softmax(preds, dim=-1)
        trues = torch.squeeze(trues, dim=-1).long()
        loss = F.cross_entropy(preds, trues)
        return loss