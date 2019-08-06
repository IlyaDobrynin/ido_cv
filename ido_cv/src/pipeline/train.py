# -*- coding: utf-8 -*-
"""
  Script for training process

"""
import gc
from torch import nn
from ..pipeline_class import Pipeline


def train(
        model:              nn.Module,
        pipeline:           Pipeline,
        dataloaders:        dict,
        loss_name:          str,
        optim_name:         str,
        model_save_dir:     str,
        val_metrics:        list,
        checkpoint_metric:  str,
        first_step:         int = 0,
        best_measure:       float = 0,
        first_epoch:        int = 0,
        epochs:             int = 1,
        n_best:             int = 1,
        scheduler:          str = 'rop',
        patience:           int = 10,
        learning_rate:      float = 0.0001
) -> nn.Module:
    """ Training function

    :param model:               Model class
    :param pipeline:            Pipeline class
    :param dataloaders:         Dataloaders dict
    :param loss_name:           Loss name. Depends on given task.
    :param optim_name:          Name of the model optimizer
    :param model_save_dir:      Path to save model files
    :param val_metrics:         List of validation metrics names
    :param checkpoint_metric:   Name of the metric, which will be monitored for early stopping
    :param first_step:          Number of step (amount of trained minibatches) to start from
    :param first_epoch:         Number of epoch to start from
    :param best_measure:        Value of best measure
    :param epochs:              Maximum number of epochs to train
    :param n_best:              Amount of best weights that will be saved after training
    :param scheduler:           Short name of the learning rate policy scheduler
    :param patience:            Amount of epochs to stop training if loss doesn't improve
    :param learning_rate:       Initial learning rate
    :return:
    """

    train_loader = dataloaders['train_dataloader']
    val_loader = dataloaders['val_dataloader']

    model = pipeline.train(
        model=model,
        lr=learning_rate,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_name=loss_name,
        optim_name=optim_name,
        metric_names=val_metrics,
        best_measure=best_measure,
        first_step=first_step,
        first_epoch=first_epoch,
        chp_metric=checkpoint_metric,
        n_epochs=epochs,
        n_best=n_best,
        scheduler=scheduler,
        patience=patience,
        save_dir=model_save_dir
    )

    print('\nTraining process done. Weights are here: {}/weights\n'.format(model_save_dir))
    del train_loader, val_loader
    gc.collect()
    return model

