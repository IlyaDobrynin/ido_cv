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
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

try:
    from itertools import ifilterfalse
except ImportError: # py3k
    from itertools import filterfalse as ifilterfalse


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


class BceMetricMulti:
    def __init__(self, jaccard_weight=0.3, metric='jaccard', class_weights=None, num_classes=11):
        # if class_weights is not None:
        #     nll_weight = utils.cuda(
        #         torch.from_numpy(class_weights.astype(np.float32)))
        # else:
        #     nll_weight = None
        self.nll_loss = nn.CrossEntropyLoss()
        self.jaccard_weight = jaccard_weight
        self.num_classes = num_classes
        self.metric = metric

    def __call__(self, outputs, targets):
        targets = targets.squeeze(1)
        # loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets)
        loss = self.nll_loss(outputs, targets)
        # print('loss.loss', loss)
        # if self.from_logits:
        # loss_acc = []
        # if self.jaccard_weight:
        #     for cls in range(0, self.num_classes):
        #         jaccard_target = (targets == cls).float()
        #         jaccard_output = outputs[:, cls, ...].exp()
        #         if self.metric == 'jaccard':
        #             metric_coef = jaccard_coef
        #         elif self.metric == 'dice':
        #             metric_coef = dice_coef
        #         else:
        #             raise NotImplementedError(
        #                 f"Metric {self.metric} doesn't implemented. "
        #                 f"Variants: 'jaccard;, 'dice'"
        #             )
        #         metric_idx = metric_coef(jaccard_output, jaccard_target, weight=None)
                # print(f'loss.metric_idx {cls}', metric_idx)
                # loss_acc.append(metric_idx)
                # loss -= (torch.log(metric_idx)) * self.jaccard_weight
                # print(loss)
        # loss = (sum(loss_acc)/len(loss_acc))
        # print('loss.loss', loss)
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
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def iou(preds, labels, C, EMPTY=1., ignore=None, per_image=False):
    """
    Array of IoU for each (non ignored) class
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        iou = []
        for i in range(C):
            if i != ignore: # The ignored label is sometimes among predicted classes (ENet - CityScapes)
                intersection = ((label == i) & (pred == i)).sum()
                union = ((label == i) | ((pred == i) & (label != ignore))).sum()
                if not union:
                    iou.append(EMPTY)
                else:
                    iou.append(float(intersection) / float(union))
        ious.append(iou)
    ious = [mean(iou) for iou in zip(*ious)] # mean accross images if per_image
    return 100 * np.array(ious)


class LovaszLoss:
    
    def __init__(self, ignore=0):
        if ignore is not None:
            self.ignore = ignore
        else:
            self.ignore = None
        
    def __call__(self, outputs, targets):
        outputs = F.softmax(outputs, dim=1)
        loss = lovasz_softmax(probas=outputs, labels=targets, ignore=self.ignore)
        return loss
        

def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                          for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float() # foreground for class c
        if (classes is 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


def xloss(logits, labels, ignore=None):
    """
    Cross entropy loss
    """
    return F.cross_entropy(logits, labels, ignore_index=ignore)


def isnan(x):
    return x != x


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n
