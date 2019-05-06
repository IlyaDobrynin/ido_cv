# -*- coding: utf-8 -*-
"""
Module implements functions for calculating metrics to compare
true labels and predicted labels using numpy
"""

import numpy as np
import torch
from ..metric_utils import numpy_metric
from ..metric_utils import torch_metric
from ..metric_utils import calculate_confusion_matrix_from_arrays
from ..metric_utils import get_metric_from_matrix


def get_metric_binary(trues, preds, metric_name, device):
    if device == 'cpu':
        trues = trues.data.cpu().numpy()
        preds = preds.data.cpu().numpy()
        metric = numpy_metric(trues=trues, preds=preds, metric_name=metric_name)
    elif device == 'gpu':
        metric = torch_metric(trues=trues, preds=preds, metric_name=metric_name)
    else:
        raise ValueError(
            f'Wrong device parameter: {device}. Should be "cpu" or "gpu".'
        )
    return metric


def get_metric_multi_per_class(y_true: np.ndarray, y_pred: np.ndarray, metric_name: str,
                               ignore_class: int = 0, threshold: float = 0.5, device='cpu') -> np.ndarray:
    """ Alternative metric for multiclass segmentation

    :param y_true: Numpy ndarray of true masks
    :param y_pred: Numpy ndarray of predicted masks
    :param metric_name: Name of the desired metric
    :param ignore_class: Index of class to ignore
    :param threshold: Class-vised threshold
    :return:
    """
    mul_metrics = []
    y_true = torch.squeeze(y_true, dim=1)

    for i in range(y_pred.shape[1]):
        if i == ignore_class:
            continue
        trues = (y_true == i).long()
        preds = (y_pred[:, i, :, :] > threshold).long()
        cls_metric = get_metric_binary(trues=trues, preds=preds, metric_name=metric_name, device=device)
        mul_metrics.append(cls_metric)

    mul_metrics = [x for x in mul_metrics if str(x) != 'nan']
    # print(mul_metrics)
    return np.mean(mul_metrics)


def get_metric_multi_confusion(y_true: np.ndarray, y_pred: np.ndarray, metric_name: str,
                               ignore_class: int) -> np.ndarray:
    """ Function to make multiclass metric (jaccard, dice, mean IoU)

    :param y_true:
    :param y_pred:
    :param metric_name:
    :param ignore_class:
    :return:
    """
    # y_pred[y_pred < 0.9] = 0
    pred_classes = np.argmax(y_pred, axis=1)
    
    true_classes = y_true
    
    confusion_matrix = calculate_confusion_matrix_from_arrays(
        pred_classes, true_classes, y_pred.shape[1]
    )
    # print(confusion_matrix)
    if ignore_class is None:
        confusion_matrix = confusion_matrix[1:, 1:]
    metric = get_metric_from_matrix(confusion_matrix, metric_name, ignore_class)
    return np.mean(metric)
