# -*- coding: utf-8 -*-
"""
    Finding learning rate pipeline

"""
import os
import gc
from torch import nn
from ... import dirs
from ..pipeline_class import Pipeline


def prediction(
        model: nn.Module,
        pipeline: Pipeline,
        dataloaders: dict,
        tta_list: list = None,
        threshold: (list, float) = None,
        postprocess: bool = False,
        show_preds: bool = False,
        save_preds: bool = False,
        output_path: str = '',
        **kwargs
):
    """ Inference process

    :param model:           Model class
    :param pipeline:        Pipeline class
    :param dataloaders:     Dataloaders dict
    :param tta_list:        Test-time augmentations
    :param threshold:       Threshold to binarize predictions
    :param postprocess:     Flag for predictions postprocessing
    :param show_preds:      Flag to show predictions
    :param save_preds:      Flag to save predictions
    :param output_path:     Path to save predictions
    :param kwargs:          Dict of keyword arguments
    :return:
    """
    task = pipeline.task
    time = pipeline.time

    test_loader = dataloaders['test_dataloader']

    test_preds = pipeline.prediction(
        model=model,
        dataloader=test_loader,
        tta_list=tta_list,
        save_batch=save_preds,
        save_dir=output_path
    )

    del test_loader
    gc.collect()

    # For segmentation
    if task == 'segmentation':
        # if postprocess:
        #     test_pred_df = pipeline.postprocessing(
        #         pred_df=test_pred_df,
        #         threshold=threshold,
        #         save=save_preds,
        #         save_dir=output_path,
        #         obj_size=20
        #     )

        if show_preds:
            pipeline.visualize_preds(preds=test_preds)

    return test_preds
