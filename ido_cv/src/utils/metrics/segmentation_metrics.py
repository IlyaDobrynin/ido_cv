# -*- coding: utf-8 -*-
"""
Module implements functions for calculating metrics to compare
true labels and predicted labels using numpy
"""

import numpy as np
import cv2
from ..image_utils import resize_image


# BINARY METRICS


def get_metric(y_true, y_pred, metric_name):
    """ Function returns metric (dice, jaccard or mean IoU)
    
    :param y_true: True labels
    :param y_pred: Predicted labels
    :param metric_name:
    :return:
    """
    smooth = 1e-12
    metrics = []
    for true, pred in zip(y_true, y_pred):
        if np.count_nonzero(true) == 0 and np.count_nonzero(pred) > 0:
            metrics.append(0)
            continue
        if np.count_nonzero(true) >= 1 and np.count_nonzero(pred) == 0:
            metrics.append(0)
            continue
        if np.count_nonzero(true) == 0 and np.count_nonzero(pred) == 0:
            metrics.append(1)
            continue

        pred = resize_image(pred, size=true.shape[:2], interpolation=cv2.INTER_NEAREST)
        true_bool = np.asarray(true, dtype=bool)
        pred_bool = np.asarray(pred, dtype=bool)
        intersection = np.sum(np.logical_and(true, pred).astype(np.uint8))
        
        if metric_name == 'dice':
            metric = (2.0 * intersection + smooth) / (true_bool.sum() + pred_bool.sum() + smooth)
        elif metric_name == 'jaccard':
            metric = (intersection + smooth) / (true_bool.sum() + pred_bool.sum() - intersection + smooth)
        elif metric_name == 'm_iou':
            iou = (intersection + smooth) / (true_bool.sum() + pred_bool.sum() - intersection + smooth)
            thresholds = np.arange(0.5, 1, 0.05)
            s = []
            for thresh in thresholds:
                s.append(iou > thresh)
            metric = np.mean(s)
        else:
            raise ValueError(
                f'Wrong metric_name: {metric_name}. Should be "dice", "jaccard" or "m_iou".'
            )
        metrics.append(metric)
    return np.mean(metrics)


# MULTICLASS METRICS


def get_metric_multi_per_class(y_true: np.ndarray, y_pred: np.ndarray, metric_name: str,
                               ignore_class: int = 0, threshold: float = 0.5) -> np.ndarray:
    """ Alternative metric for multiclass segmentation

    :param y_true: Numpy ndarray of true masks
    :param y_pred: Numpy ndarray of predicted masks
    :param metric_name: Name of the desired metric
    :param ignore_class: Index of class to ignore
    :param threshold: Class-vised threshold
    :return:
    """
    mul_metrics = []
    y_true_copy = np.copy(y_true)
    y_pred_copy = np.copy(y_pred)
    for i in range(y_pred.shape[1]):
        if i == ignore_class:
            continue
        trues = (y_true_copy == i).astype(np.uint8)
        preds = (y_pred_copy[:, i, :, :] > threshold).astype(np.uint8)

        cls_metric = get_metric(y_true=trues, y_pred=preds, metric_name=metric_name)
        mul_metrics.append(cls_metric)
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


def calculate_confusion_matrix_from_arrays(prediction: np.ndarray, ground_truth: np.ndarray,
                                           nr_labels: int) -> np.ndarray:
    replace_indices = np.vstack((
        ground_truth.flatten(),
        prediction.flatten())
    ).T
    confusion_matrix, _ = np.histogramdd(
        replace_indices,
        bins=(nr_labels, nr_labels),
        range=[(0, nr_labels), (0, nr_labels)]
    )
    confusion_matrix = confusion_matrix.astype(np.uint32)
    return confusion_matrix


def get_metric_from_matrix(confusion_matrix: np.ndarray, metric_name: str,
                           ignore_class: int = None) -> list:
    metrics = []
    if ignore_class is None:
        index_list = [i for i in range(confusion_matrix.shape[0])]
    else:
        index_list = [i for i in range(confusion_matrix.shape[0]) if i != ignore_class]
    
    for index in index_list:

        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[:, index].sum() - true_positives
        false_negatives = confusion_matrix[index, :].sum() - true_positives
        
        if metric_name == 'jaccard':
            denom = true_positives + false_positives + false_negatives
        elif metric_name == 'dice':
            denom = 2 * true_positives + false_positives + false_negatives
        else:
            raise ValueError(
                f'Wrong metric name: {metric_name}. Should be dice or jaccard.'
            )
        if denom == 0:
            iou = 0
        else:
            if metric_name == 'jaccard':
                iou = float(true_positives) / denom
            else:
                iou = 2 * float(true_positives) / denom
        metrics.append(iou)
    return metrics
