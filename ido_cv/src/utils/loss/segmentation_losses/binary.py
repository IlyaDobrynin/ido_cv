# -*- coding: utf-8 -*-
"""
Module implements binary losses
Implemented losses:
1. bce_dice loss
2. bce_jaccard loss
3. focal_dice loss
4. focal_jaccard loss
5. bce_lovasz loss

ToDo: Add moar losses!

"""

from itertools import filterfalse
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def dice_coef(preds, trues, weight=None):
    """ Function returns binary dice coefficient

    :param preds: Predictions of the network
    :param trues: Ground truth labels
    :param weight: Weight to scale
    :return: dice score
    """
    eps = 1e-12
    if weight is not None:
        w = torch.autograd.Variable(weight)
        intersection = (w * preds * trues).sum()
        union = (w * preds).sum() + (w * trues).sum()
    else:
        intersection = (preds * trues).sum()
        union = (preds).sum() + (trues).sum()
    score = (2. * intersection + eps) / (union + eps)
    return score


def jaccard_coef(preds, trues, weight=None):
    """ Function returns binary jaccard coefficient (IoU)
    
    :param preds: Predictions of the network
    :param trues: Ground truth labels
    :param weight: Weight to scale
    :return: jaccard score
    """
    eps = 1e-12
    if weight is not None:
        w = torch.autograd.Variable(weight)
        intersection = (w * preds * trues).sum()
        union = (w * preds).sum() + (w * trues).sum()
    else:
        intersection = (preds * trues).sum()
        union = (preds).sum() + (trues).sum()
    score = (intersection + eps) / (union - intersection + eps)
    return score


def get_weight(trues, weight_type=0):
    """ Function returns weights to scale loss

    :param trues: Masks array
    :param weight_type: 0 or 1
    :return: weights array
    """
    if trues.shape[-1] == 128:
        kernel_size = 11
    elif trues.shape[-1] == 256:
        kernel_size = 21
    elif trues.shape[-1] == 512:
        kernel_size = 21
    elif trues.shape[-1] == 1024:
        kernel_size = 41
    else:
        raise ValueError('Unexpected image size: {}. Should be 128, 256, 512 or 1024.'.format(trues.shape[-1]))

    bin_target = torch.where(trues > 0, torch.tensor(1), torch.tensor(0))
    ave_mask = F.avg_pool2d(bin_target.float(), kernel_size=kernel_size, padding=10, stride=1)
    if weight_type == 0:
        edge_idx = (ave_mask.ge(0.01) * ave_mask.le(0.99)).float()
        weights = torch.ones_like(edge_idx, dtype=torch.float)
        weights_sum0 = weights.sum()
        weights = weights + edge_idx * 2.
        weights_sum1 = weights.sum()
        weights = weights / weights_sum1 * weights_sum0
    elif weight_type == 1:
        weights = torch.ones_like(ave_mask, dtype=torch.float)
        weights_sum0 = weights.sum()
        weights = 5. * torch.exp(-5. * torch.abs(ave_mask - 0.5))
        weights_sum1 = weights.sum()
        weights = weights / weights_sum1 * weights_sum0
    else:
        raise ValueError("Unknown weight type: {}. Should be 0, 1 or None.".format(weight_type))
    return weights


class BceMetricBinary(nn.Module):
    """
    Loss defined as (1 - alpha) * BCE - alpha * SoftJaccard
    """

    def __init__(self, metric=None, weight_type=None, is_average=False, alpha=0.3):
        super().__init__()
        self.metric = metric
        self.weight_type = weight_type
        self.is_average = is_average
        self.alpha = alpha

    def __call__(self, preds, trues):
        metric_target = (trues == 1).float()
        metric_output = torch.sigmoid(preds)

        # Weight estimation
        if self.weight_type:
            weights = get_weight(trues=trues, weight_type=self.weight_type)
        else:
            weights = None

        self.bce_loss = nn.BCEWithLogitsLoss(weight=weights)
        bce_loss = self.bce_loss(preds, trues)
        if self.metric:
            if self.metric == 'jaccard':
                metric_coef = jaccard_coef(metric_target, metric_output, weight=weights)
            elif self.metric == 'dice':
                metric_coef = dice_coef(metric_target, metric_output, weight=weights)
            else:
                raise NotImplementedError("Metric {} doesn't implemented. Variants: 'jaccard;, 'dice', None".format(self.metric))
            loss = (1 - self.alpha) * bce_loss - self.alpha * torch.log(metric_coef)
        else:
            loss = bce_loss

        return loss


class FocalMetric(nn.Module):
    def __init__(self, metric=None, alpha=0.3):
        super().__init__()
        self.focal_loss = RobustFocalLoss2d()
        self.alpha = alpha
        self.metric = metric
   
    def __call__(self, outputs, targets):
        focal_loss = (1 - self.alpha) * self.focal_loss(outputs, targets)

        metric_target = (targets == 1).float()
        metric_output = torch.sigmoid(outputs)
        if self.metric:
            if self.metric == 'jaccard':
                metric_coef = jaccard_coef(metric_target, metric_output)
            elif self.metric == 'dice':
                metric_coef = dice_coef(metric_target, metric_output)
            else:
                raise NotImplementedError("Metric {} doesn't implemented. Variants: 'jaccard;, 'dice', None".format(self.metric))
            loss = (1 - self.alpha) * focal_loss - self.alpha * torch.log(metric_coef)
        else:
            loss = focal_loss

        return loss


class RobustFocalLoss2d(nn.Module):
    # assume top 10% is outliers
    def __init__(self, gamma=2, size_average=True):
        super().__init__()
        self.gamma = gamma
        self.size_average = size_average

    def __call__(self, preds, targets, class_weight=None, type='sigmoid'):
        target = targets.view(-1, 1).long()

        if type == 'sigmoid':
            if class_weight is None:
                class_weight = [1] * 2  # [0.5, 0.5]

            prob = torch.sigmoid(preds)
            prob = prob.view(-1, 1)
            prob = torch.cat((1 - prob, prob), 1)
            select = torch.FloatTensor(len(prob), 2).zero_().cuda()
            select.scatter_(1, target, 1.)

        elif type == 'softmax':
            B, C, H, W = preds.size()
            if class_weight is None:
                class_weight = [1] * C  # [1/C]*C

            logit = preds.permute(0, 2, 3, 1).contiguous().view(-1, C)
            prob = torch.softmax(logit, 1)
            select = torch.Tensor(len(prob), C).zero_().cuda()
            select.scatter_(1, target, 1.)
        else:
            raise ValueError('Wrong type of activation function: {}. Should be "sigmoid" or "softmax".'.format(type))

        class_weight = torch.Tensor(class_weight).cuda().view(-1, 1)
        class_weight = torch.gather(class_weight, 0, target)

        prob = (prob * select).sum(1).view(-1, 1)
        prob = torch.clamp(prob, 1e-8, 1 - 1e-8)

        focus = torch.pow((1 - prob), self.gamma)
        # focus = torch.where(focus < 2.0, focus, torch.zeros(prob.size()).cuda())
        focus = torch.clamp(focus, 0, 2)

        batch_loss = - class_weight * focus * prob.log()

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss

        return loss


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """ Binary Lovasz hinge loss

      logits: [B, H, W] Variable, logits at each pixel (between -infty and +infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = (lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore)) for log, lab in
                zip(logits, labels))
        return mean(loss)
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge_flat(logits, labels):
    """ Binary Lovasz hinge loss

      logits: [P] Variable, logits at each prediction (between -infty and +infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * torch.Tensor(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), torch.Tensor(grad))
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    arr = []
    l = iter(l)
    if ignore_nan:
        l = filterfalse(np.isnan, l)
    try:
        n = 1
        acc = next(l)
        arr.append(acc)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
        arr.append(v)
    if n == 1:
        return acc
    return acc / n  # , np.asarray(arr)


class LovaszBCE(nn.Module):
    def __init__(self, bce_weight=0.5, per_image=True, ignore=None):
        super().__init__()
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.per_image = per_image
        self.ignore = ignore

    def __call__(self, logits, labels):
        return lovasz_hinge(logits, labels, self.per_image, self.ignore) \
               + self.nll_loss(logits, labels) * self.bce_weight
    
    
def make_loss(loss_name):
    """ Loss factory
    
    :param loss_name: Name of the implemented loss function
    :return: loss function
    """
    if loss_name == 'bce_jaccard':
        loss = BceMetricBinary(metric='jaccard')
    elif loss_name == 'bce_dice':
        loss = BceMetricBinary(metric='dice')
    elif loss_name == 'focal_jaccard':
        loss = FocalMetric(metric='jaccard')
    elif loss_name == 'focal_dice':
        loss = FocalMetric(metric='dice')
    elif loss_name == 'bce_lovasz':
        loss = LovaszBCE()
    else:
        raise NotImplementedError('Loss {} not implemented'.format(loss_name))
    return loss
