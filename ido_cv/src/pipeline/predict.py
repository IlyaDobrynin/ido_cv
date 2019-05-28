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
        data_path: str = None,
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
    :param data_path:       Path to images to predict
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
    if task == 'detection':
        cls_thresh = kwargs['cls_thresh']
        nms_thresh = kwargs['nms_thresh']
    else:
        cls_thresh = None
        nms_thresh = None

    test_loader = dataloaders['test_dataloader']
    test_pred_df = pipeline.predict(
        model=model,
        dataloader=test_loader,
        cls_thresh=cls_thresh,
        nms_thresh=nms_thresh,
        tta_list=tta_list
    )

    del test_loader
    gc.collect()

    # For segmentation
    if task == 'segmentation':
        if postprocess:
            test_pred_df = pipeline.postprocessing(
                pred_df=test_pred_df,
                threshold=threshold,
                save=save_preds,
                save_dir=output_path,
                obj_size=20
            )

        if show_preds:
            pipeline.visualize_preds(preds_df=test_pred_df,
                                     images_path=os.path.join(data_path, 'images'))

    if task == 'classification':
        if save_preds:
            out_path = dirs.make_dir(
                relative_path=f"preds/{task}/{time}",
                top_dir=output_path
            )
            test_pred_df.to_csv(path_or_buf=os.path.join(out_path, f"preds.csv"), index=False)

    if task == 'ocr':
        if postprocess:
            test_pred_df = pipeline.postprocessing(
                pred_df=test_pred_df,
                save=save_preds,
                save_dir=output_path
            )
