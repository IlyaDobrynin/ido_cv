# -*- coding: utf-8 -*-
"""
Module implements functions for calculating metrics to compare
true labels and predicted labels using numpy
"""

import numpy as np


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

        # true = true.astype(bool)
        # pred = pred.astype(bool)
        # intersection = (true & pred).sum()
        # im_sum = true.sum() + pred.sum()
        
        # Alternate
        intersection = np.sum(np.logical_and(true, pred).astype(np.uint8))
        union = np.sum(np.logical_or(true, pred).astype(np.uint8))
        # iou = np.sum(intersection > 0) / np.sum(union > 0)
        
        if metric_name == 'dice':
            metrics.append(2.0 * (intersection + smooth) / (union + intersection + smooth))
        elif metric_name == 'jaccard':
            metrics.append((intersection + smooth) / (union + smooth))
        elif metric_name == 'm_iou':
            iou = (intersection + smooth) / (union + smooth)
            thresholds = np.arange(0.5, 1, 0.05)
            s = []
            for thresh in thresholds:
                s.append(iou > thresh)
            metrics.append(np.mean(s))
        else:
            raise ValueError(
                f'Wrong metric_name: {metric_name}. Should be "dice", "jaccard" or "m_iou".'
            )
            
    return np.mean(metrics)


# MULTICLASS METRICS


def get_metric_multi(y_true, y_pred, metric_name, ignore_class):
    """ Function to make multiclass metric (jaccard, dice, mean IoU)

    :param y_true:
    :param y_pred:
    :return:
    """
    # y_pred[y_pred < 0.9] = 0
    pred_classes = np.argmax(y_pred, axis=1)
    
    true_classes = y_true
    
    confusion_matrix = calculate_confusion_matrix_from_arrays(pred_classes,
                                                              true_classes,
                                                              y_pred.shape[1])
    # print(confusion_matrix)
    if ignore_class is None:
        confusion_matrix = confusion_matrix[1:, 1:]
    metric = get_metric_from_matrix(confusion_matrix, metric_name, ignore_class)
    return np.mean(metric)


def calculate_confusion_matrix_from_arrays(prediction, ground_truth, nr_labels):
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


def get_metric_from_matrix(confusion_matrix, metric_name, ignore_class=None):
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
