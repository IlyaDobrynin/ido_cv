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
        val_metrics:    list,
        tta_list:       list,
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
    :param tta_list:        List of the test-time augmentations
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
    scores = pipeline.validation(
        model=model,
        dataloader=holdout_loader,
        metric_names=val_metrics,
        validation_mode='test',
        tta_list=tta_list
    )
    return scores
