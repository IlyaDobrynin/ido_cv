# -*- coding: utf-8 -*-
"""
    Utils for torch losses

"""
try:
    from itertools import ifilterfalse
except ImportError: # py3k
    from itertools import filterfalse as ifilterfalse
import numpy as np
import torch
from torch.nn import functional as F


def dice_coef(preds, trues, weight=None) -> torch.Tensor:
    """ Function returns binary dice coefficient

    :param preds: Predictions of the network
    :param trues: Ground truth labels
    :param weight: Weight to scale
    :return: dice score
    """
    eps = 1e-12
    if weight is not None:
        w = torch.Tensor(weight)
        intersection = (w * preds * trues).sum()
        union = (w * preds).sum() + (w * trues).sum()
    else:
        intersection = (preds * trues).sum()
        union = preds.sum() + trues.sum()
    score = (2. * intersection + eps) / (union + eps)
    return score


def jaccard_coef(preds, trues, weight=None) -> float:
    """ Function returns binary jaccard coefficient (IoU)

    :param preds: Predictions of the network
    :param trues: Ground truth labels
    :param weight: Weight to scale
    :return: jaccard score
    """
    eps = 1e-12
    if weight is not None:
        w = torch.Tensor(weight)
        intersection = (w * preds * trues).sum()
        union = (w * preds).sum() + (w * trues).sum()
    else:
        intersection = (preds * trues).sum()
        union = preds.sum() + trues.sum()
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
        raise ValueError(
            f'Unexpected image size: {trues.shape[-1]}. Should be 128, 256, 512 or 1024.'
        )

    bin_target = (trues > 0).long()  #torch.where(trues > 0, torch.Tensor(1), torch.Tensor(0))
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


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """ Binary Lovasz hinge loss

      logits: [B, H, W] Variable, logits at each pixel (between -infty and +infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = (
            lovasz_hinge_flat(
                *flatten_binary_scores(
                    log.unsqueeze(0), lab.unsqueeze(0), ignore
                )
            ) for log, lab in zip(logits, labels))
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
    # errors = (1. - logits * torch.Tensor(signs))
    errors = (1. - logits * torch.autograd.Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), torch.autograd.Variable(grad))
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
        l = ifilterfalse(np.isnan, l)
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
        loss = mean(
            lovasz_softmax_flat(
                *flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore),
                classes=classes
            ) for prob, lab in zip(probas, labels)
        )
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


def one_hot_embedding(labels, num_classes):
    '''Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N,#classes].
    '''
    y = torch.eye(num_classes)  # [D,D]

    # print('ohe.labels  ', labels, labels.shape)
    # print('ohe.y[labels]  ', y[labels], y[labels].shape)
    return y[labels]  # [N,D]


def sigmoid_focal_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
    reduction: str = "mean"
):
    """
    Compute binary focal loss between target and output logits.
    Source https://github.com/BloodAxe/pytorch-toolbelt
    See :class:`~pytorch_toolbelt.losses` for details.
    Args:
        outputs: Tensor of arbitrary shape
        targets: Tensor of the same shape as input
        reduction (string, optional):
            Specifies the reduction to apply to the output:
            "none" | "mean" | "sum" | "batchwise_mean".
            "none": no reduction will be applied,
            "mean": the sum of the output will be divided by the number of
            elements in the output,
            "sum": the output will be summed.
    See https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/loss/losses.py  # noqa: E501
    """
    targets = targets.type(outputs.type())

    logpt = -F.binary_cross_entropy_with_logits(
        outputs, targets, reduction="none"
    )
    pt = torch.exp(logpt)

    # compute the loss
    loss = -((1 - pt).pow(gamma)) * logpt

    if alpha is not None:
        loss = loss * (alpha * targets + (1 - alpha) * (1 - targets))

    if reduction == "mean":
        loss = loss.mean()
    if reduction == "sum":
        loss = loss.sum()
    if reduction == "batchwise_mean":
        loss = loss.sum(0)

    return loss


def reduced_focal_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    gamma: float = 2.0,
    reduction="mean"
):
    """
    Compute reduced focal loss between target and output logits.
    Source https://github.com/BloodAxe/pytorch-toolbelt
    See :class:`~pytorch_toolbelt.losses` for details.
    Args:
        outputs: Tensor of arbitrary shape
        targets: Tensor of the same shape as input
        reduction (string, optional):
            Specifies the reduction to apply to the output:
            "none" | "mean" | "sum" | "batchwise_mean".
            "none": no reduction will be applied,
            "mean": the sum of the output will be divided by the number of
            elements in the output,
            "sum": the output will be summed.
            Note: :attr:`size_average` and :attr:`reduce`
            are in the process of being deprecated,
            and in the meantime, specifying either of those two args
            will override :attr:`reduction`.
            "batchwise_mean" computes mean loss per sample in batch.
            Default: "mean"
    See https://arxiv.org/abs/1903.01347
    """
    targets = targets.type(outputs.type())

    logpt = -F.binary_cross_entropy_with_logits(
        outputs, targets, reduction="none"
    )
    pt = torch.exp(logpt)

    # compute the loss
    focal_reduction = ((1. - pt) / threshold).pow(gamma)
    focal_reduction[pt < threshold] = 1

    loss = -focal_reduction * logpt

    if reduction == "mean":
        loss = loss.mean()
    if reduction == "sum":
        loss = loss.sum()
    if reduction == "batchwise_mean":
        loss = loss.sum(0)

    return loss