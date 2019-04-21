# -*- coding: utf-8 -*-
"""
  Script for finding optimal learning rate

"""
import gc
from ..pipeline_class import Pipeline


def find_lr(pipeline: Pipeline, model_name: str, path_to_dataset: str, batch_size: int = 5, workers: int = 1,
            shuffle_dataset: bool = False, use_augs: bool = False, device_ids: list = None,
            cudnn_benchmark: bool = True, path_to_weights: str = None, lr_factor: int = 10):
    print('-' * 30, ' FINDING LEARNING RATE ', '-' * 30)
    # Get dataloader for learning rate finding process
    dataloader = pipeline.get_dataloaders(
        path_to_dataset=path_to_dataset, #os.path.join(args['data_path'], 'train'),
        batch_size=batch_size,
        is_train=True,
        workers=workers,
        shuffle=shuffle_dataset,
        augs=use_augs
    )
    # Get model
    find_lr_model, _, _ = pipeline.get_model(
        model_name=model_name,
        device_ids=device_ids,
        cudnn_bench=cudnn_benchmark,
        path_to_weights=path_to_weights
    )
    # Find learning rate process
    optimum_lr = pipeline.find_lr(
        model=find_lr_model,
        dataloader=dataloader,
        lr_reduce_factor=lr_factor,
        verbose=1,
        show_graph=True
    )
    del dataloader
    gc.collect()
    return optimum_lr