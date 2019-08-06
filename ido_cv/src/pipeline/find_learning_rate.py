# -*- coding: utf-8 -*-
"""
  Script for finding optimal learning rate

"""
import gc
from ..pipeline_class import Pipeline


def find_lr(
        pipeline:           Pipeline,
        model_name:         str,
        model_parameters:   dict,
        dataloaders:        dict,
        loss_name:          str,
        optim_name:         str,
        device_ids:         list = None,
        cudnn_benchmark:    bool = True,
        path_to_weights:    str = None,
        lr_factor:          int = 10
) -> float:
    """ Function to find optimal learning rate

    :param pipeline:            Pipeline class
    :param model_name:          Name of the model
    :param model_parameters:    Model initial parameters
    :param dataloaders:         Dataloaders dict
    :param loss_name:           Loss name. Depends on given task.
    :param optim_name:          Name of the model optimizer
    :param device_ids:          List of devices to train on. [-1] for training on CPU.
    :param cudnn_benchmark:     Flag to make cudnn benchmark
                                (https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936)
    :param path_to_weights:     Path to the pretrained weights
    :param lr_factor:           Factor to divide learning rate with min loss
    :return:
    """
    # Get model
    find_lr_model, _, _ = pipeline.get_model(
        model_name=model_name,
        device_ids=device_ids,
        cudnn_bench=cudnn_benchmark,
        path_to_weights=path_to_weights,
        model_parameters=model_parameters
    )

    find_lr_loader = dataloaders['find_lr_dataloader']

    # Find learning rate process
    optimum_lr = pipeline.find_lr(
        model=find_lr_model,
        dataloader=find_lr_loader,
        loss_name=loss_name,
        optim_name=optim_name,
        lr_reduce_factor=lr_factor,
        verbose=1,
        show_graph=True
    )
    del find_lr_loader
    gc.collect()
    return optimum_lr
