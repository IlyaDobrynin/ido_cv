# -*- coding: utf-8 -*-
"""
  Script for finding optimal learning rate

"""
import gc
from ..pipeline_class import Pipeline


def find_lr(pipeline: Pipeline, model_name: str, path_to_dataset: str, batch_size: int = 5,
            workers: int = 1, shuffle_dataset: bool = False, use_augs: bool = False,
            device_ids: list = None, cudnn_benchmark: bool = True, path_to_weights: str = None,
            lr_factor: int = 10) -> float:
    """ Function to find optimal learning rate

    :param pipeline: Pipeline class
    :param model_name: Name of the model
    :param path_to_dataset: Path to train images
    :param batch_size: Size of the data minibatch
    :param workers: Number of subprocesses to use for data loading
    :param shuffle_dataset: Flag to shuffle data in dataloader
    :param use_augs: Flag to use augmentations
    :param device_ids: List of devices to train on. [-1] for training on CPU.
    :param cudnn_benchmark: Flag to make cudnn benchmark
           (https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936)
    :param path_to_weights: Path to the pretrained weights
    :param lr_factor: Factor to divide learning rate with min loss
    :return:
    """
    # Get dataloader for learning rate finding process
    dataloader = pipeline.get_dataloaders(
        path_to_dataset=path_to_dataset,
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
