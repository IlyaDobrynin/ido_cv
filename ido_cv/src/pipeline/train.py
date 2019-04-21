# -*- coding: utf-8 -*-
"""
  Script for training process

"""
import gc
from ..pipeline_class import Pipeline


def train(model, pipeline: Pipeline, train_data_path: str, val_data_path: str, model_save_dir: str,
          val_metrics: list, checkpoint_metric: str, batch_size: int = 1, first_step: int = 0,
          best_measure: float = 0, first_epoch: int = 0, epochs: int = 1, n_best: int = 1,
          scheduler: str = 'rop', workers: int = 1, shuffle_train: bool = False, augs: bool = False,
          patience: int = 10, learning_rate: float = 0.0001):
    """ Training function

    :param model: Model class
    :param pipeline: Pipeline class
    :param train_data_path: Path to train images
    :param val_data_path: Path to validation images
    :param model_save_dir: Path to save model files
    :param val_metrics: List of validation metrics names
    :param checkpoint_metric: Name of the metric, which will be monitored for early stopping
    :param batch_size: Size of the data minibatch
    :param first_step: Number of step (amount of trained minibatches) to start from
    :param first_epoch: Number of epoch to start from
    :param best_measure: Value of best measure
    :param epochs: Maximum number of epochs to train
    :param n_best: Amount of best weights that will be saved after training
    :param scheduler: Short name of the learning rate policy scheduler
    :param workers: Number of subprocesses to use for data loading
    :param shuffle_train: Flag to shuffle train data in dataloader
    :param augs: Flag to use augmentations
    :param patience: Amount of epochs to stop training if loss doesn't improve
    :param learning_rate: Initial learning rate
    :return:
    """

    print('-' * 30, ' TRAINING ', '-' * 30)
    train_loader = pipeline.get_dataloaders(
        path_to_dataset=train_data_path,
        # path_to_labels=train_labels_path,
        batch_size=batch_size,
        is_train=True,
        workers=workers,
        shuffle=shuffle_train,
        augs=augs
    )
    val_loader = pipeline.get_dataloaders(
        path_to_dataset=val_data_path,
        # path_to_labels=val_labels_path,
        batch_size=batch_size,
        is_train=True,
        workers=workers,
        shuffle=False,
        augs=False
    )

    model = pipeline.train(
        model=model,
        lr=learning_rate,
        train_loader=train_loader,
        val_loader=val_loader,
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

