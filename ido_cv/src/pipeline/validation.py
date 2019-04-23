# -*- coding: utf-8 -*-
"""
     Script for validation process

"""
import gc
import numpy as np
from ..pipeline_class import Pipeline
from ..utils.common_utils import get_true_classification
from ..utils.common_utils import get_true_segmentation
from ..utils.common_utils import get_true_detection


def validation(model, pipeline: Pipeline, data_path: str, labels_path: str, val_metrics: list,
               batch_size: int = 1, workers: int = 1, save_preds: bool = False,
               output_path: str = '', **kwargs) -> dict:
    """ Validation process

    :param model: Model class
    :param pipeline: Pipeline class
    :param data_path: Path to validation images
    :param labels_path: Path to validation labels
    :param val_metrics: List of validation metrics names
    :param batch_size: Size of the data minibatch
    :param workers: Number of subprocesses to use for data loading
    :param save_preds: Flag to save predictions
    :param output_path: Path to save predictions
    :param kwargs: Dict of keyword arguments
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

    holdout_loader = pipeline.get_dataloaders(
        path_to_dataset=data_path,
        batch_size=batch_size,
        is_train=False,
        workers=workers,
        shuffle=False,
        augs=False
    )
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
        true_df = get_true_segmentation(labels_path=labels_path, mode=mode)
        scores = pipeline.evaluate_metrics(
            true_df=true_df,
            pred_df=pred_df,
            metric_names=val_metrics
        )

    # Get score for classification
    else:  # task == 'classification':
        true_df = get_true_classification(labels_path=labels_path)
        scores = pipeline.evaluate_metrics(
            true_df=true_df,
            pred_df=pred_df,
            metric_names=val_metrics
        )
        for k, v in scores.items():
            print(f'{k} metric: {v}')

    return scores
