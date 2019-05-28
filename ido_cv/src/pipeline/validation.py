# -*- coding: utf-8 -*-
"""
     Script for validation process

"""
import os
import gc
import numpy as np
from torch import nn
from ..pipeline_class import Pipeline
from ..utils.common_utils import get_true_classification
from ..utils.common_utils import get_true_segmentation
from ..utils.common_utils import get_true_detection
from ..utils.common_utils import get_true_ocr


def validation(
        model:          nn.Module,
        pipeline:       Pipeline,
        dataloaders:    dict,
        data_path:      str,
        val_metrics:    list,
        save_preds:     bool = False,
        output_path:    str = '',
        **kwargs
) -> dict:
    """ Validation process

    :param model:           Model class
    :param pipeline:        Pipeline class
    :param dataloaders:     Dataloaders dict
    :param data_path:       Path to validation images
    :param val_metrics:     List of validation metrics names
    :param save_preds:      Flag to save predictions
    :param output_path:     Path to save predictions
    :param kwargs:          Dict of keyword arguments
    :return:
    """

    task = pipeline.task
    mode = pipeline.mode
    if task == 'detection':
        cls_thresh = kwargs['cls_thresh']
        nms_thresh = kwargs['nms_thresh']
        val_iou_thresholds = kwargs['val_iou_thresholds']
    else:
        cls_thresh = None
        nms_thresh = None
        val_iou_thresholds = None

    holdout_loader = dataloaders['holdout_dataloader']
    pred_df = pipeline.predict(
        model=model,
        dataloader=holdout_loader,
        cls_thresh=cls_thresh,
        nms_thresh=nms_thresh,
        save=save_preds,
        save_dir=output_path
    )
    del holdout_loader
    gc.collect()

    # Get score for detection
    if task == 'detection':
        labels_path = os.path.join(data_path, 'labels.txt')
        true_df = get_true_detection(labels_path=labels_path)
        scores = pipeline.evaluate_metrics(
            true_df=true_df,
            pred_df=pred_df,
            iou_thresholds=val_iou_thresholds
        )
        print('Average Precisions for all classes: {}'.format(scores))
        print('Mean Average Precision (mAP) for all classes: {:.4f}'.format(np.mean(scores)))

    # Get score for segmentation
    elif task == 'segmentation':
        labels_path = os.path.join(data_path, 'masks')
        true_df = get_true_segmentation(
            labels_path=labels_path,
            mode=mode
        )

        scores = pipeline.evaluate_metrics(
            true_df=true_df,
            pred_df=pred_df,
            metric_names=val_metrics
        )

    # Get score for ocr
    elif task == 'ocr':
        labels_path = os.path.join(data_path, 'labels.csv')
        true_df = get_true_ocr(
            labels_path=labels_path
        )
        scores = pipeline.evaluate_metrics(
            true_df=true_df,
            pred_df=pred_df,
            metric_names=val_metrics
        )

    # Get score for classification
    else:  # task == 'classification':
        labels_path = os.path.join(data_path, 'labels.csv')
        true_df = get_true_classification(labels_path=labels_path)
        scores = pipeline.evaluate_metrics(
            true_df=true_df,
            pred_df=pred_df,
            metric_names=val_metrics
        )
    return scores
