# -*- coding: utf-8 -*-
"""
Module implements base class for classification pipeline

"""
import os
import gc
import copy
import warnings
from pathlib import Path
from abc import ABC, abstractmethod
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imsave

import torch
import torch.optim as optimizers
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm import tqdm

from .. import dirs
from . import allowed_parameters

from .models.models_facade import ModelsFacade
from .data_utils.dataset_facade import DatasetFacade
from .utils.loss.loss_facade import LossesFacade
from .utils.metrics.metric_facade import MetricsFacade

from .data_utils.encoder_alt import DataEncoder
from .data_utils.data_augmentations import Augmentations

from .utils.get_model_config import ConfigParser
from .utils import model_utils
from .utils.image_utils import unpad_bboxes
from .utils.image_utils import resize_bboxes
from .utils.image_utils import draw_images
from .utils.image_utils import draw_image_boxes
from .utils.image_utils import delete_small_instances
from .utils.metric_utils import get_opt_threshold
from .utils.metrics.detection_metrics import mean_ap

pd.set_option('display.max_columns', 10)

TASKS_MODES = allowed_parameters.TASKS_MODES
METRIC_PARAMETERS = allowed_parameters.METRIC_PARAMETERS
MODEL_PARAMETERS = allowed_parameters.MODEL_PARAMETERS
LOSS_PARAMETERS = allowed_parameters.LOSS_PARAMETERS
OPTIMIZERS = allowed_parameters.OPTIMIZERS


class AbstractPipeline(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def get_dataloaders(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abstractmethod
    def validation(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass


class Pipeline(AbstractPipeline):
    """
    Class implements basic logic of detection pipeline
    Arguments:
        task:               Task for pipeline:
                                - classification
                                - segmentation
                                - detection
        mode:               Mode for given task. Depends on given task.
        loss_name:          Loss name. Depends on given task.
        optim_name:         Name of the model optimizer:
                                - 'adam'
                                - 'sgd'
        time:               Current time.
        allocate_on:        Device(s) for locating model
        tta_list:           List of test-time augmentations.
                            Currently implemented only for binary segmentation.
        random_seed:        Random seed for fixating model random state.
        img_size_orig:      Original size of input images (
        img_size_target:    Size of input images for the model input


    """

    def __init__(self, task: str, mode: str, loss_name: str = None, optim_name: str = None,
                 time: str = None, allocate_on: str = 'cpu', tta_list: list = None,
                 random_seed: int = 1, img_size_orig: (int, tuple) = None,
                 img_size_target: (int, tuple) = None):

        super(Pipeline, self).__init__()

        tasks = TASKS_MODES.keys()
        if task not in tasks:
            raise ValueError(
                f'Wrong task parameter: {task}. '
                f'Should be {tasks}.'
            )

        modes = TASKS_MODES[task].keys()
        if mode not in modes:
            raise ValueError(
                f'Wrong mode parameter: {mode}. '
                f'Should be {modes}.'
            )

        self.task = task
        self.mode = mode

        if allocate_on in ['cpu', 'gpu']:
            self.allocate_on = allocate_on
        else:
            raise ValueError(
                f'Wrong allocate_on parameter: {allocate_on}. Should be "cpu" or "gpu"'
            )

        if self.task == 'segmentation':
            self.tta_list = tta_list

        # Get loss for training
        if loss_name is not None:
            loss_facade = LossesFacade(self.task, self.mode, loss_name)
            loss_parameters = LOSS_PARAMETERS[self.task][self.mode][loss_name]
            self.criterion = loss_facade.get_loss(**loss_parameters)

        # Get metrics for training
        metric_facade = MetricsFacade(task=self.task)
        metric_parameters = METRIC_PARAMETERS[self.task][self.mode]
        self.val_metric_class = metric_facade.get_metric(mode=self.mode, **metric_parameters)

        # Get optimizer
        if optim_name is not None:
            self.optim_name = optim_name

        self.img_size_orig = img_size_orig
        self.img_size_target = img_size_target
        self.random_seed = random_seed
        self.time = time

    def get_dataloaders(self, dataset_class: Dataset = None, path_to_dataset: str = None,
                        data_file: np.ndarray = None, batch_size: int = 1, is_train: bool = True,
                        label_colors: dict = None, workers: int = 1, shuffle: bool = False,
                        augs: bool = False) -> DataLoader:
        """ Function to make train data loaders

        :param dataset_class:   Dataset class
        :param path_to_dataset: Path to the images
        :param data_file:       Data file
        :param batch_size:      Size of data minibatch
        :param is_train:        Flag to specify dataloader type (train or test)
        :param label_colors:    Colors for labeling for multiclass segmentation
        :param workers:         Number of multithread workers
        :param shuffle:         Flag for random shuffling train dataloader samples
        :param augs:            Flag to add augmentations
        :return:
        """

        if (dataset_class is None) and (path_to_dataset is None) and (data_file is None):
            raise ValueError(
                f"At list one from dataset_class path_to_dataset or data_file parameters "
                f"should be filled!"
            )
        if (self.task == 'segmentation') \
                and (self.mode == 'multi') \
                and (label_colors is None):
            raise ValueError(
                f"Provide label_colors for multiclass segmentation task!"
            )

        if augs:
            augmentations = Augmentations(is_train).transform
        else:
            augmentations = None

        if dataset_class is None:
            # ToDo: make input parameters for dataset universal
            if self.task == 'detection':
                dataset_parameters = dict(
                    root=path_to_dataset,
                    labels_file=os.path.join(path_to_dataset, 'labels.csv'),
                    initial_size=self.img_size_orig,
                    model_input_size=self.img_size_target,
                    train=is_train,
                    augmentations=augmentations
                )
            else:
                dataset_parameters = dict(
                    data_path=path_to_dataset,
                    data_file=data_file,
                    train=is_train,
                    label_colors=label_colors,
                    initial_size=self.img_size_orig,
                    model_input_size=self.img_size_target,
                    augmentations=augmentations
                )
            facade_class = DatasetFacade(task=self.task, mode=self.mode)
            dataset_class = facade_class.get_dataset_class(**dataset_parameters)

        dataloader = DataLoader(dataset_class,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=workers,
                                collate_fn=dataset_class.collate_fn)
        return dataloader

    def get_model(self, model_name: str = None, path_to_weights: str = None, save_path: str = None,
                  device_ids: list = None, cudnn_bench: bool = False,  verbose: bool = False,
                  model_parameters: dict = None, show_model: bool = False) -> tuple:
        """ Function returns model, allocated to the given gpu's

        :param model_name: Class of the model
        :param path_to_weights: Path to the trained weights
        :param save_path: Path to the trained weights
        :param device_ids: List of the gpu's
        :param cudnn_bench: Flag to include cudnn benchmark
        :param verbose: Flag to show info
        :param model_parameters: Path to the trained weights
        :param show_model: Flag to show model
        :return:
        """

        # Make initial model parameters for training
        initial_parameters = dict(
            epoch=0,
            step=0,
            best_measure=0
        )

        # Get model parameters
        if path_to_weights is None:
            if model_parameters is None:
                model_parameters = MODEL_PARAMETERS[self.task][self.mode][model_name]
                print(
                    f'Pretrained weights are not provided. '
                    f'Train process runs with defauld {model_name} parameters: \n{model_parameters}'
                )

            # Get model class
            facade = ModelsFacade(task=self.task, model_name=model_name)
            model = facade.get_model(**model_parameters)
        else:

            if not os.path.exists(path_to_weights):
                raise ValueError(f'Wrong path to weights: {path_to_weights}')

            # Load model class and weights
            path_to_model = Path(path_to_weights).parents[1]

            # Get saved configs
            cfg_path = os.path.join(path_to_model, 'hyperparameters.yml')
            if not os.path.exists(cfg_path):
                raise ValueError(
                    f"Path {cfg_path} does not exists."
                )
            cfg_parser = ConfigParser(cfg_type='model', cfg_path=cfg_path)
            model_parameters = cfg_parser.parameters['model_parameters']
            model_name = cfg_parser.parameters['model_name']

            # Get path to saved model class
            model_class_path = os.path.join(path_to_model, 'model_class')

            if self.allocate_on == 'cpu':
                device = torch.device('cpu')
                model = torch.load(str(model_class_path), map_location=device)
                model_dict = torch.load(str(path_to_weights), map_location=device)

                model_weights = {key.replace('module.', ''): value
                                 for key, value in model_dict['model_weights'].items()
                                 if 'module' in key}
            else:
                model = torch.load(str(model_class_path))
                model_dict = torch.load(str(path_to_weights))
                model_weights = {key.replace('module.', ''): value
                                 for key, value in model_dict['model_weights'].items()
                                 if 'module' in key}

            model.load_state_dict(model_weights)
            initial_parameters['epoch'] = model_dict['epoch']
            initial_parameters['step'] = model_dict['step']
            initial_parameters['best_measure'] = model_dict['best_measure']

        if save_path is not None:
            torch.save(model, os.path.join(save_path, 'model_class'))
        else:
            msg = "Model class will not save"
            warnings.warn(msg, Warning)

        # Locate model into device
        model = model_utils.allocate_model(model, device_ids=device_ids, cudnn_bench=cudnn_bench)

        if verbose:
            print(f'Train process runs with {model_name} parameters:')
            for param_name, param_val in model_parameters.items():
                print(f'{param_name}: {param_val}')
            if show_model:
                from torchsummary import summary
                summary_device = 'cpu' if self.allocate_on == 'cpu' else 'cuda'
                summary(
                    model=model,
                    input_size=(3, self.img_size_target, self.img_size_target),
                    device=summary_device
                )

        return model, initial_parameters, model_parameters

    def find_lr(self, model: nn.Module, dataloader: DataLoader, lr_reduce_factor: int = 1,
                verbose: int = 1, show_graph: bool = True, init_value: float = 1e-8,
                final_value: float = 10., beta: float = 0.98) -> float:
        """ Function find optimal learning rate for the given model and parameters

        :param model:               Input model
        :param dataloader:          Input data loader
        :param lr_reduce_factor:    Factor to
        :param verbose:             Flag to show output information
        :param show_graph:          Flag to showing graphic
        :param init_value:          Start LR
        :param final_value:         Max stop LR
        :param beta: Smooth         value
        :return:
        """
        # Get optimizer
        optimizer_dict = dict(
            adam=optimizers.Adam(model.parameters(), lr=0.5, weight_decay=0.000001),
            rmsprop=optimizers.RMSprop(model.parameters(), lr=0.5),
            sgd=optimizers.SGD(model.parameters(), lr=0.5, nesterov=True, momentum=0.9)
        )
        optimizer = optimizer_dict[self.optim_name]

        # Set initial parameters
        num = len(dataloader) - 1
        mult = (final_value / init_value) ** (1 / num)
        lr = init_value
        optimizer.param_groups[0]['lr'] = lr
        avg_loss = 0.
        best_loss = 0.
        batch_num = 0
        lr_dict = dict()

        # Train the model for one epoch
        model.train()
        for data in tqdm(dataloader):
            batch_num += 1

            # Get the loss for this mini-batch of inputs/outputs
            inputs = model_utils.cuda(data[0], self.allocate_on)
            with torch.no_grad():
                if self.task == 'segmentation':
                    targets = model_utils.cuda(data[1], self.allocate_on)
                elif self.task == 'detection':
                    targets = (
                        model_utils.cuda(data[1], self.allocate_on),
                        model_utils.cuda(data[2], self.allocate_on)
                    )
                else:  # self.task == 'classification'
                    targets = model_utils.cuda(data[1], self.allocate_on)

            optimizer.zero_grad()
            preds = model(inputs)

            loss = self.criterion(preds, targets)

            # Compute the smoothed loss
            avg_loss = beta * avg_loss + (1 - beta) * loss.data.item()
            smoothed_loss = avg_loss / (1 - beta ** batch_num)

            # Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > 4 * best_loss:
                # return lr_dict
                break

            # Record the best loss
            if smoothed_loss < best_loss or batch_num == 1:
                best_loss = smoothed_loss

            # Store the values
            lr_dict[np.log10(lr)] = smoothed_loss

            # Do the optimizer step
            loss.backward()
            optimizer.step()

            # Update the lr for the next step
            lr *= mult
            optimizer.param_groups[0]['lr'] = lr

        del model, optimizer
        gc.collect()

        # Find optimum learning rate
        min_lr = 10 ** min(lr_dict, key=lr_dict.get)
        print(min_lr)
        opt_lr = min_lr / lr_reduce_factor
        if verbose == 1:
            print(f'Current learning rate: {opt_lr:.10f}')
            # Show graph if necessary
            if show_graph:
                m = -5
                plt.plot(list(lr_dict.keys())[10:m], list(lr_dict.values())[10:m])
                plt.show()
        return opt_lr

    def train(self, model: nn.Module, lr: float, train_loader: DataLoader, val_loader: DataLoader,
              metric_names: list, best_measure: float, first_step: int = 0, first_epoch: int = 0,
              chp_metric: str = 'loss', n_epochs: int = 1, n_best: int = 1, scheduler: str = 'rop',
              patience: int = 10, save_dir: str = '') -> nn.Module:
        """ Training pipeline function

        :param model:           Input model
        :param lr:              Initial learning rate
        :param train_loader:    Train dataloader
        :param val_loader:      Validation dataloader
        :param metric_names:    Metrics to print (available are 'dice', 'jaccard', 'm_iou')
        :param best_measure:    Best value of the used criterion
        :param first_step:      Initial step (number of batch)
        :param first_epoch:     Initial epoch
        :param chp_metric:      Criterion to save best weights ('loss' or one of metrics)
        :param n_epochs:        Overall amount of epochs to learn
        :param n_best:          Amount of best-scored weights to save
        :param scheduler:       Name of the lr scheduler policy:
                                    - rop for ReduceOnPlateau policy
                                    - None for ni policy
        :param patience:        Amount of epochs to stop training if loss doesn't improve
        :param save_dir:        Path to save training model weights
        :return:                Trained model
        """

        # Get optimizer
        # ToDo: make optimizer in the get_model method
        optimizer_dict = dict(
            adam=optimizers.Adam(model.parameters(), lr=lr, weight_decay=0.00001),
            rmsprop=optimizers.RMSprop(model.parameters(), lr=lr),
            sgd=optimizers.SGD(model.parameters(), lr=lr, nesterov=True, momentum=0.9)
        )
        optimizer = optimizer_dict[self.optim_name]

        # Get learning rate scheduler policy
        if scheduler == 'rop':
            mode = 'min' if chp_metric in ['loss'] else 'max'
            scheduler = ReduceLROnPlateau(optimizer, mode=mode, factor=0.5,
                                          patience=int(patience / 2), verbose=True)
        else:
            raise ValueError(
                'Wrong scheduler parameter: {}. Should be "rop"'.format(scheduler)
            )

        # Make log file
        log_path = os.path.join(save_dir, r'logs.log')
        log_file = open(log_path, mode='a', encoding='utf8')

        # Make dir for weights save
        save_weights_path = dirs.make_dir(relative_path='weights', top_dir=save_dir)
        best_model_wts = copy.deepcopy(model.state_dict())
        
        # Main training loop
        early_stop_measure = []
        for e in range(first_epoch, first_epoch + n_epochs + 1):
            model.train()
            tq = tqdm(total=(len(train_loader.dataset)))
            tq.set_description(
                'Epoch {}, lr {:.10f}'.format(e, optimizer.param_groups[0].get('lr'))
            )
            losses = []
            for data in train_loader:
                # Locate data on devices
                inputs = model_utils.cuda(data[0], self.allocate_on)
                with torch.no_grad():
                    if self.task == 'segmentation':
                        targets = model_utils.cuda(data[1], self.allocate_on)
                    elif self.task == 'detection':
                        loc_targets = data[1]
                        cls_targets = data[2]
                        targets = (
                            model_utils.cuda(loc_targets, self.allocate_on),
                            model_utils.cuda(cls_targets, self.allocate_on)
                        )
                    else:  # self.task == 'classification'
                        targets = model_utils.cuda(data[1], self.allocate_on)

                # Set gradients to zero
                optimizer.zero_grad()
                # Make prediction
                outputs = model(inputs)
                # Calculate loss
                loss = self.criterion(outputs, targets)
                losses.append(loss.item())
                loss.backward()
                # Make optimizer step
                optimizer.step()
                first_step += 1
                # Update progress bar
                tq.update(inputs.shape[0])
                train_loss = np.mean(losses)
                tq.set_postfix(loss='{:.5f}'.format(train_loss))

            # Close progress bar after epoch
            tq.close()
            del tq
            gc.collect()

            # Calculate validation metrics
            val_metrics = self.validation(
                model=model, dataloader=val_loader, metric_names=metric_names
            )

            # Write epoch parameters to log file
            model_utils.write_event(log_file, first_step, e, **val_metrics)
            val_loss = val_metrics['loss']

            # Make scheduler step
            if scheduler:
                scheduler.step(val_metrics[chp_metric])

            # Save weights if best
            if chp_metric in ['loss']:
                measure = 1 / val_metrics[chp_metric]
                early_stop_mode = 'min'
            else:
                measure = val_metrics[chp_metric]
                early_stop_mode = 'max'

            early_stop_measure.append(val_metrics[chp_metric])

            if measure > best_measure:
                best_weights_name = os.path.join(
                    save_weights_path, f'epoch-{e}_{chp_metric}-{val_metrics[chp_metric]:.5f}.pth'
                )
                print(
                    f'Validation {chp_metric} has improved to {val_metrics[chp_metric]:.5f}.'
                    f' Save weights to {best_weights_name}'
                )
                best_model_wts = copy.deepcopy(model.state_dict())
                state_dict = dict(
                    model_weights=best_model_wts,
                    epoch=e,
                    step=first_step,
                    best_measure=best_measure
                )
                best_measure = measure
                torch.save(state_dict, best_weights_name)

            # Stop training if early stop criterion
            if model_utils.early_stop(early_stop_measure, patience=patience, mode=early_stop_mode):
                print('Early stopping')
                break

        # Remove all weights but n best
        reverse = True if chp_metric in ['loss'] else False
        model_utils.remove_all_but_n_best(
            weights_dir=save_weights_path, n_best=n_best, reverse=reverse
        )

        # Load best model weights
        model.load_state_dict(best_model_wts)

        return model

    def validation(self, model: nn.Module, dataloader: DataLoader, metric_names: list,
                   verbose: int = 1) -> dict:
        """ Function to make validation of the model

        :param model: Input model to validate
        :param dataloader: Validation dataloader
        :param metric_names:
        :param verbose: Flag to include output information
        :return:
        """
        model.eval()
        losses = list()
        metric_values = dict()
        for m_name in metric_names:
            metric_values[m_name] = list()

        with torch.no_grad():
            for data in dataloader:
                inputs = model_utils.cuda(data[0], self.allocate_on)
                if self.task == 'segmentation':
                    targets = model_utils.cuda(data[1], self.allocate_on)
                elif self.task == 'detection':
                    loc_target = data[1]
                    cls_target = data[2]
                    targets = (
                        model_utils.cuda(loc_target, self.allocate_on),
                        model_utils.cuda(cls_target, self.allocate_on)
                    )
                else:  # self.task == 'classification':
                    targets = model_utils.cuda(data[1], self.allocate_on)

                # Make predictions
                outputs = model(inputs)

                # Calculate loss
                loss = self.criterion(outputs, targets)
                losses.append(loss.item())

                # Get parameters for metrics calculation function
                if self.task == 'segmentation':
                    get_metric_parameters = dict(
                        trues=targets, preds=outputs, threshold=0.5
                    )
                    if self.mode == 'multi':
                        get_metric_parameters['per_class'] = True
                        get_metric_parameters['ignore_class'] = None
                elif self.task == 'detection':
                    # TODO implement detection metric
                    raise NotImplementedError
                else:  # self.task == 'classification'
                    get_metric_parameters = dict(trues=targets, preds=outputs)

                # Calculate metrics for the batch of images
                for m_name in metric_names:
                    metric_values[m_name] += [
                        self.val_metric_class.get_metric(
                            metric_name=m_name, **get_metric_parameters
                        )
                    ]

        # Calculate mean metrics for all images
        out_metrics = dict()
        out_metrics['loss'] = np.mean(losses).astype(np.float64)
        for m_name in metric_names:
            out_metrics[m_name] = np.mean(metric_values[m_name]).astype(np.float64)

        # Show information if needed
        if verbose == 1:
            string = ''
            for key, value in out_metrics.items():
                string += '{}: {:.5f} '.format(key, value)
            print(string)
        return out_metrics

    def predict(self, model: nn.Module, dataloader: DataLoader, save: bool = False,
                save_dir: str = '', **kwargs) -> pd.DataFrame:
        """ Function to make predictions

        :param model:       Input model
        :param dataloader:  Test dataloader
        :param save:        Flag to save results
        :param save_dir:    Path to save results
        :return:
        """
        model.eval()

        # Get predictions dictionary
        predictions = dict()
        predictions['names'] = []
        if self.task == 'detection':
            predictions['boxes'] = []
            predictions['labels'] = []
            predictions['scores'] = []
        elif self.task == 'segmentation':
            predictions['masks'] = None
        else:  # self.task == 'classification'
            predictions['labels'] = None

        # Main prediction loop
        for batch_idx, (inputs, names) in tqdm(enumerate(dataloader), total=(len(dataloader))):
            with torch.no_grad():
                for name in names:
                    predictions['names'].append(name)

                inputs = model_utils.cuda(inputs, self.allocate_on)

                # Get predictions for detection task
                if self.task == 'detection':
                    cls_thresh = kwargs['cls_thresh']
                    nms_thresh = kwargs['nms_thresh']
                    # TODO Make this work with different models (currently - only RetinaNet)
                    outputs = model(inputs)
                    encoder = DataEncoder(input_size=self.img_size_target)
                    loc_preds = outputs[0]
                    cls_preds = outputs[1]
                    for loc, cls, name in zip(loc_preds, cls_preds, names):
                        preds = encoder.decode(loc.data.cpu().squeeze(),
                                               cls.data.cpu().squeeze(),
                                               cls_thresh=cls_thresh,
                                               nms_thresh=nms_thresh)
                        if preds is not None:
                            boxes, labels, scores = preds[0], preds[1], preds[2]
                            boxes = boxes.data.cpu().numpy().astype(np.uint8)
                            boxes = resize_bboxes(boxes=boxes, size_from=self.img_size_target,
                                                  size_to=max(self.img_size_orig))
                            boxes = unpad_bboxes(boxes=boxes, img_shape=self.img_size_orig)
                            labels = labels.data.cpu().numpy()
                            scores = scores.data.cpu().numpy()

                            # Sort boxes, labels and scores on x-coord
                            preds = [(box, label, score) for box, label, score in zip(boxes,
                                                                                      labels,
                                                                                      scores)]
                            preds = np.asarray(sorted(preds, key=lambda x: x[0][0]))
                            predictions['boxes'].append(np.asarray([pred[0] for pred in preds]))
                            predictions['labels'].append(np.asarray([pred[1] for pred in preds]))
                            predictions['scores'].append(np.asarray([pred[2] for pred in preds]))
                        else:
                            predictions['boxes'].append([])
                            predictions['labels'].append([])
                            predictions['scores'].append([])

                # Get predictions for binary segmentation
                elif self.task == 'segmentation':
                    if self.mode == 'binary':
                        tta_stack = []
                        for tta_class in self.tta_list:
                            tta_operation = tta_class(allocate_on=self.allocate_on)
                            tta_out = tta_operation(model, inputs)
                            tta_stack.append(tta_out)
                        tta_stack = np.squeeze(np.mean(tta_stack, axis=0), 1)

                        if predictions['masks'] is None:
                            predictions['masks'] = tta_stack
                        else:
                            predictions['masks'] = np.append(predictions['masks'], tta_stack, 0)

                    # Get predictions for multiclass segmentation task
                    else:  # self.mode == 'multi'
                        outputs = model(inputs)
                        outputs = torch.softmax(outputs, dim=1)
                        out_masks = np.moveaxis(outputs.data.cpu().numpy(), 1, -1).astype(np.float32)

                        if predictions['masks'] is None:
                            predictions['masks'] = out_masks
                        else:
                            predictions['masks'] = np.append(predictions['masks'], out_masks, 0)

                # Get predictions for classification task
                else:  # self.mode == 'classification'
                    if self.mode == 'binary':
                        outputs = model(inputs)
                        outputs = torch.sigmoid(outputs)
                        preds = np.round(np.squeeze(outputs.data.cpu().numpy(), axis=1)).astype(np.uint8)
                        # print('prediction.outputs', preds, preds.shape)
                        if predictions['labels'] is None:
                            predictions['labels'] = preds
                        else:
                            predictions['labels'] = np.append(predictions['labels'], preds, 0)
                    else:  # self.mode == 'multi'
                        outputs = model(inputs)
                        outputs = torch.softmax(outputs, dim=-1)
                        preds = np.argmax(outputs.data.cpu().numpy(), axis=-1).astype(np.uint8)
                        if predictions['labels'] is None:
                            predictions['labels'] = preds
                        else:
                            predictions['labels'] = np.append(predictions['labels'], preds, 0)

        # Make predictions dataframe
        pred_df = pd.DataFrame()
        pred_df['names'] = predictions['names']
        if self.task == 'detection':
            pred_df['boxes'] = predictions['boxes']
            pred_df['labels'] = predictions['labels']
            pred_df['scores'] = predictions['scores']
        elif self.task == 'segmentation':
            masks = [msk for msk in predictions['masks']]
            pred_df['masks'] = masks
        else:  # self.task == 'classification'
            pred_df['labels'] = predictions['labels']

        # Save result
        if save:
            csv_path = os.path.join(save_dir, r'{}.csv'.format(self.time))
            pred_df.to_csv(path_or_buf=csv_path, sep=';')
            print("Predictions saved. Path: {}".format(csv_path))
        return pred_df

    def postprocessing(self, pred_df: pd.DataFrame, threshold: (list, float) = 0.,
                       hole_size: int = None, obj_size: int = None, save: bool = False,
                       save_dir: str = '') -> pd.DataFrame:
        """ Function for predicted data postprocessing

        :param pred_df:     Dataset woth predicted images and masks
        :param threshold:   Binarization thresholds
        :param hole_size:   Minimum hole size to fill
        :param obj_size:    Minimum obj size
        :param save:        Flag to save results
        :param save_dir:    Save directory
        :return:
        """

        if self.task == 'segmentation':
            if self.mode == 'binary':
                masks = np.copy(pred_df['masks'].values)

                postproc_masks = [
                    delete_small_instances(
                        (mask > threshold).astype(np.uint8), obj_size=obj_size, hole_size=hole_size
                    ) for mask in masks
                ]
                pred_df['masks'] = postproc_masks

                # Save result
                if save:
                    for row in pred_df.iterrows():
                        name = row[1]['names']
                        mask = row[1]['masks']
                        out_path = dirs.make_dir(f'preds/{self.task}/{self.time}', top_dir=save_dir)
                        imsave(fname=os.path.join(out_path, name), arr=mask)

            else:  # self.mode == 'multi'
                postproc_masks = []
                masks = np.copy(pred_df['masks'].values)
                for mask in masks:
                    threshold = [threshold] * mask.shape[-1] if type(threshold) == float \
                        else threshold
                    postproc_mask = np.zeros(shape=mask.shape[:2], dtype=np.uint8)
                    for ch in range(0, mask.shape[-1]):
                        if ch == 0:
                            threshold_0 = 0.5
                            mask_ch = (mask[:, :, ch] > threshold_0).astype(np.uint8)
                        else:
                            mask_ch = (mask[:, :, ch] > threshold[ch - 1]).astype(np.uint8)
                        mask_ch = delete_small_instances(
                            mask_ch,
                            obj_size=obj_size,
                            hole_size=hole_size
                        )
                        postproc_mask[mask_ch > 0] = ch

                    postproc_masks.append(postproc_mask)
                pred_df['masks'] = postproc_masks

                # Save result
                if save:
                    for row in pred_df.iterrows():
                        name = row[1]['names']
                        mask = row[1]['masks']
                        out_path = dirs.make_dir(f'preds/{self.task}/{self.time}', top_dir=save_dir)
                        imsave(fname=os.path.join(out_path, name), arr=mask)

        return pred_df

    def evaluate_metrics(self, true_df: pd.DataFrame, pred_df: pd.DataFrame,
                         ignore_class: int = None, label_colors: dict = None,
                         iou_thresholds: list = None, metric_names: list = None,
                         verbose: int = 1) -> dict:
        """ Common function to evaluate metrics

        :param true_df:
        :param pred_df:
        :param ignore_class:
        :param label_colors:
        :param iou_thresholds:
        :param metric_names:
        :param verbose:
        :return:
        """

        if self.task == 'detection':
            out_metrics = mean_ap(true_df=true_df, pred_df=pred_df, iou_thresholds=iou_thresholds)
        elif self.task == 'segmentation':
            if metric_names is None:
                raise ValueError(
                    f"Wrong metric_names parameter: {metric_names}."
                )

            all_df = true_df.merge(pred_df, on='names', suffixes=['_true', '_pred'])
            masks_t = np.asarray([mask for mask in all_df['masks_true'].values])
            masks_p = np.asarray([mask for mask in all_df['masks_pred'].values])

            if self.mode == 'binary':
                out_metrics = dict()
                for m_name in metric_names:
                    threshold, metric_value = get_opt_threshold(
                        masks_t, masks_p, metric_name=m_name
                    )
                    out_metrics[m_name] = {'threshold': threshold, 'value': metric_value}
                if verbose == 1:
                    for k, v in out_metrics.items():
                        print(f'{k} metric: ')
                        for m_k, m_v in v.items():
                            print(f'- best {m_k}: {m_v:.5f}')

            else:  # self.mode == 'multi'
                out_metrics = dict()
                colors = label_colors
                for i, (class_name, class_color) in enumerate(colors.items()):
                    if i == ignore_class:
                        continue
                    class_masks_t = np.all(masks_t == class_color, axis=-1).astype(np.uint8)
                    class_masks_p = masks_p[..., i + 1]

                    # for img_idx in range(0, 10):
                    #     draw_images([class_masks_t[img_idx, ...], class_masks_p[img_idx, ...]])

                    out_metrics[class_name] = dict()
                    for m_name in metric_names:
                        threshold, metric_value = get_opt_threshold(
                            class_masks_t, class_masks_p, metric_name=m_name
                        )
                        out_metrics[class_name][m_name] = {
                            'threshold': threshold,
                            'value': metric_value
                        }

                if verbose == 1:
                    for class_name, class_values in out_metrics.items():
                        print(f'\nClass {class_name}: ')
                        for k, v in class_values.items():
                            print(f'{k} metric: ')
                            for m_k, m_v in v.items():
                                print(f'- best {m_k}: {m_v:.5f}')

        else:  # self.task == 'classification'
            all_df = true_df.merge(pred_df, on='names', suffixes=['_true', '_pred'])
            labels_t = all_df['labels_true'].values.astype(np.uint8)
            labels_p = all_df['labels_pred'].values
            out_metrics = dict()
            for m_name in metric_names:
                out_metrics[m_name] = self.val_metric_class.get_metric_value(
                    labels_t, labels_p, m_name
                )

        return out_metrics

    def visualize_preds(self, preds_df: pd.DataFrame, images_path: str):
        """ Function for simple visualization

        :param preds_df: Dataframe with predicted images
        :param images_path: Path to images
        :return:
        """
        # Get visualization for detection task
        if self.task == 'detection':
            for row in preds_df.iterrows():
                name = row[1]['names']
                boxes = row[1]['boxes'].tolist()
                labels = row[1]['labels']
                scores = row[1]['scores']
                image = np.asarray(cv2.imread(filename=os.path.join(images_path, name)),
                                   dtype=np.uint8)
                h_i, w_i = image.shape[0], image.shape[1]
                h, w = self.img_size_orig
                if h != h_i and w != w_i:
                    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
                draw_image_boxes(image, boxes, labels, scores)

        elif self.task == 'segmentation':
            # Get visualization for binary segmentation task
            if self.mode == 'binary':
                for row in preds_df.iterrows():
                    name = row[1]['names']
                    image = np.asarray(cv2.imread(filename=os.path.join(images_path, name)), dtype=np.uint8)
                    mask = row[1]['masks']

                    # msk_img = np.copy(image)
                    # matching = np.all(np.expand_dims(mask, axis=-1) > 0.1, axis=-1)
                    # msk_img[matching, :] = [0, 0, 0]
                    draw_images([image, mask], orient='vertical')

            # Get visualization for multiclass segmentation task
            else:  # self.mode == 'multi'
                for row in preds_df.iterrows():
                    name = row[1]['names']
                    image = np.asarray(cv2.imread(filename=os.path.join(images_path, name)),
                                       dtype=np.uint8)
                    mask = row[1]['masks']
                    draw_images([image, mask], orient='vertical')
