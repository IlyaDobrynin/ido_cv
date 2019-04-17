# -*- coding: utf-8 -*-
"""
Module implements base class for classification pipeline

"""
import os
import gc
import copy
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

import cv2
from skimage.io import imsave
import matplotlib.pyplot as plt

import torch
import torch.optim as optimizers
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from .. import dirs
from .utils.get_model_config import ConfigParser
from . import allowed_parameters
from .data_utils.datasets.detection_dataset import RetinaDataset
from .data_utils.encoder_alt import DataEncoder
from .data_utils.datasets.segmentation_dataset import BinSegDataset
from .data_utils.datasets.segmentation_dataset import MultSegDataset
from .data_utils.datasets.classification_dataset import ClassifyDataset
from .data_utils.data_augmentations import Augmentations
from .data_utils import data_postprocessing as postproc
from .utils import model_utils
from .utils.images_transform import unpad_bboxes, resize_bboxes, unpad, resize
from .utils.utils import get_opt_threshold
from .utils.metrics.detection_metrics import mean_ap
from .utils.model_utils import write_event

pd.set_option('display.max_columns', 10)

MODELS = allowed_parameters.MODELS
METRICS = allowed_parameters.METRICS
LOSSES = allowed_parameters.LOSSES
OPTIMIZERS = allowed_parameters.OPTIMIZERS


class AbstractPipeline(ABC):
    
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
        model_name (str): name of the model
        criterion (method): training loss function to minimize
        optim_name (str): name of the model optimizer:
                                'adam'
                                'SGD'
        allocate_on (str): Device for locating model
        img_size_orig (int, tuple): original size of input images
        img_size_target (int): target size of input images
        random_seed (int): random seed
        time (date): current time
    """

    def __init__(self, task, mode, loss_name=None, optim_name=None, allocate_on='cpu', tta_list=None,
                 img_size_orig=None, img_size_target=None, random_seed=1, time=None):

        super(Pipeline, self).__init__()

        if task in ['segmentation', 'classification']:
            if mode in ['binary', 'multi']:
                self.task = task
                self.mode = mode
            else:
                raise ValueError(
                    f'Wrong mode parameter: {mode}.'
                    'Should be "binary" or "multi".'
                )
        elif task == 'detection':
            if mode == 'all':
                self.task = task
                self.mode = mode
            else:
                raise ValueError(
                    f'Wrong mode parameter: {mode}.'
                    'Should be "all" for detection task.'
                )
        else:
            raise ValueError(
                f'Wrong task parameter: {task}. '
                'Should be "segmentation_b", "segmentation_m", "detection", "classification"'
            )

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
            self.criterion = LOSSES[self.task][self.mode][loss_name]

        # Get optimizer
        if optim_name is not None:
            self.optim_name = optim_name

        self.img_size_orig = img_size_orig
        self.img_size_target = img_size_target
        self.random_seed = random_seed
        self.time = time

    def get_dataloaders(self, data_file=None, path_to_dataset=None, path_to_labels=None, batch_size=1, is_train=True,
                        workers=1, shuffle=False, augs=False):
        """ Function to make train data loaders

        :param path_to_dataset: Path to the images
        :param data_file: Data file
        :param path_to_labels: Path to the images labels
        :param batch_size: Size of data minibatch
        :param is_train: Flag to specify dataloader type (train or test)
        :param workers: Number of multithread workers
        :param shuffle: Flag for random shuffling train dataloader samples
        :return:
        """
        if augs:
            augmentations = Augmentations(is_train).transform
        else:
            augmentations = None
        if self.task == 'detection':
            dataset_class = RetinaDataset(root=path_to_dataset,
                                          labels_file=path_to_labels,
                                          initial_size=self.img_size_orig,
                                          model_input_size=self.img_size_target,
                                          train=is_train,
                                          augmentations=augmentations)
        elif self.task == 'segmentation':
            if self.mode == 'binary':
                dataset_class = BinSegDataset(
                    data_path=path_to_dataset,
                    data_file=data_file,
                    train=is_train,
                    initial_size=self.img_size_orig,
                    model_input_size=self.img_size_target,
                    add_depth=False,
                    augmentations=augmentations
                )
            else:  # self.mode == 'multi':
                dataset_class = MultSegDataset(
                    data_path=path_to_dataset,
                    data_file=data_file,
                    train=is_train,
                    initial_size=self.img_size_orig,
                    model_input_size=self.img_size_target,
                    add_depth=False,
                    augmentations=augmentations
                )
        else:  # self.task == 'classification'
            dataset_class = ClassifyDataset(
                data_path=path_to_dataset,
                data_file=data_file,
                train=is_train,
                initial_size=self.img_size_orig,
                model_input_size=self.img_size_target,
                augmentations=augmentations
            )

        dataloader = DataLoader(dataset_class,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=workers,
                                collate_fn=dataset_class.collate_fn)
        return dataloader

    def get_model(self, model_name: str, device_ids: list=None, cudnn_bench: bool=False,
                  path_to_weights: str=None, model_parameters: dict=None) -> tuple:
        """ Function returns model, allocated to the given gpu's

        :param model_name: Class of the model
        :param device_ids: List of the gpu's
        :param cudnn_bench: Flag to include cudnn benchmark
        :param path_to_weights: Path to the trained weights
        :return:
        """
        # Get model parameters
        if path_to_weights is None:
            if model_parameters is None:
                model_parameters = MODELS[self.task][self.mode][model_name]['default_parameters']
                print(
                    f'Pretrained weights are not provided. Train process runs with defauld {model_name} parameters:\n'
                    f'{model_parameters}'
                )
        else:
            path_to_model = Path(path_to_weights).parents[1]
            cfg_path = os.path.join(path_to_model, 'hyperparameters.yml')
            if not os.path.exists(cfg_path):
                raise ValueError(
                    f"Path {cfg_path} does not exists."
                )
            cfg_parser = ConfigParser(cfg_type='model', cfg_path=cfg_path)
            model_parameters = cfg_parser.parameters['model_parameters']

        # Get model class
        model_class = MODELS[self.task][self.mode][model_name]['class']
        model = model_class(**model_parameters)

        # Locate model into device
        model = model_utils.allocate_model(model, device_ids=device_ids, cudnn_bench=cudnn_bench)

        # Make initial model parameters for training
        initial_parameters = dict(
            epoch=0,
            step=0,
            best_measure=0
        )

        # Load model weights
        if path_to_weights is not None:
            if os.path.exists(path_to_weights):
                if self.allocate_on == 'cpu':
                    state = torch.load(str(path_to_weights), map_location={'cuda:0': 'cpu',
                                                                           'cuda:1': 'cpu'})
                    model_state = {key.replace('module.', ''): value for key, value in
                                   state['model'].items()}
                else:
                    state = torch.load(str(path_to_weights))
                    model_state = state['model']

                model.load_state_dict(model_state)

                initial_parameters['epoch'] = state['epoch']
                initial_parameters['step'] = state['epoch']
                initial_parameters['best_measure'] = state['best_measure']

                # return model, initial_parameters
            else:
                raise ValueError(f'Wrong path to weights: {path_to_weights}')

        return model, initial_parameters, model_parameters

    def find_lr(self, model, dataloader, lr_reduce_factor=1, verbose=1, show_graph=True,
                init_value=1e-8, final_value=10., beta=0.98):
        """ Function find optimal learning rate for the given model and parameters

        :param model: Input model
        :param dataloader: Input data loader
        :param lr_reduce_factor: Factor to
        :param verbose: Flag to show output information
        :param show_graph: Flag to showing graphic
        :param init_value: Start LR
        :param final_value: Max stop LR
        :param beta: Smooth value
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
        for data in dataloader:
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

    def train(self, model, lr, train_loader, val_loader, metric_names, best_measure, first_step=0,
              first_epoch=0, chp_metric='loss', n_epochs=1, n_best=1, scheduler='rop', patience=10,
              save_dir=''):
        """ Training pipeline function

        :param model: Input model
        :param lr: Initial learning rate
        :param train_loader: Train dataloader
        :param val_loader: Validation dataloader
        :param metric_names: Metrics to print (available are 'dice', 'jaccard', 'm_iou')
        :param best_measure: Best value of the used criterion
        :param first_step: Initial step (number of batch)
        :param first_epoch: Initial epoch
        :param chp_metric: Criterion to save best weights ('loss' or one of metrics)
        :param n_epochs: Overall amount of epochs to learn
        :param n_best: Amount of best-scored weights to save
        :param scheduler: Name of the lr scheduler policy:
                            - rop for ReduceOnPlateau policy
                            - None for ni policy
        :param patience: Amount of epochs to stop training if loss doesn't improve
        :param save_dir: Path to save training model weights
        :return: Trained model
        """

        # Get optimizer
        optimizer_dict = dict(
            adam=optimizers.Adam(model.parameters(), lr=lr, weight_decay=0.00001),
            rmsprop=optimizers.RMSprop(model.parameters(), lr=lr),
            sgd=optimizers.SGD(model.parameters(), lr=lr, nesterov=True, momentum=0.9)
        )
        optimizer = optimizer_dict[self.optim_name]

        # Get metric functions
        metrics = dict()
        for m_name in metric_names:
            metrics[m_name] = METRICS[self.task][self.mode][m_name]

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

        best_weights_name = os.path.join(
                    save_weights_path, f'epoch-{first_epoch}_{chp_metric}-{best_measure:.5f}.pth'
                )
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
            val_metrics = self.validation(model=model, dataloader=val_loader, metrics=metrics)

            # Write epoch parameters to log file
            write_event(log_file, first_step, e, **val_metrics)
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
                    model=best_model_wts,
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
        model_utils.remove_all_but_n_best(weights_dir=save_weights_path,
                                          n_best=n_best,
                                          reverse=reverse)

        # Load best model weights
        model.load_state_dict(best_model_wts)

        return model

    def validation(self, model, dataloader, metrics, verbose=1):
        """ Function to make validation of the model

        :param model: Input model to validate
        :param dataloader: Validation dataloader
        :param metrics: Metrics to print (available are 'dice', 'jaccard', 'm_iou')
        :param verbose: Flag to include output information
        :return: Validation score and loss
        """
        model.eval()
        losses = list()
        metric_values = dict()
        for m_name in metrics.keys():
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

                # Calculate metrics
                # class_metrics = {k: [] for k in range(1, 11)}
                for m_name, m_func in metrics.items():
                    if self.task == 'segmentation':
                        if self.mode == 'binary':
                            true_batch = np.squeeze(targets.data.cpu().numpy(), axis=1)
                            pred_batch = np.squeeze(outputs.data.cpu().numpy(), axis=1).astype(np.float32)
                            metric_values[m_name] += [
                                m_func(true_batch, (pred_batch > 0.5), metric_name=m_name)
                            ]
                        else:  # self.mode == 'multi'
                            outputs = torch.softmax(outputs, dim=1)
                            true_batch = np.squeeze(targets.data.cpu().numpy(), axis=1)
                            pred_batch = outputs.data.cpu().numpy().astype(np.float32)
                            metric_values[m_name] += [
                                m_func(true_batch, pred_batch, metric_name=m_name, ignore_class=None)
                            ]
                    elif self.task == 'detection':
                        # TODO implement detection metric
                        metric_values[m_name] += None
                    else:  # self.task == 'classification'
                        if self.mode == 'binary':
                            true_batch = np.squeeze(targets.data.cpu().numpy(), axis=1).astype(np.uint8)
                            outputs = torch.sigmoid(outputs)
                            pred_batch = np.round(np.squeeze(outputs.data.cpu().numpy(), axis=1)).astype(np.uint8)
                            metric_values[m_name] += [
                                m_func(true_batch, pred_batch)
                            ]
                        else:  # self.mode == 'multi'
                            # TODO сделать мультиклассовую классификацию
                            true_batch = np.squeeze(targets.data.cpu().numpy(), axis=-1).astype(np.uint8)
                            outputs = torch.softmax(outputs, dim=-1)
                            pred_batch = np.argmax(outputs.data.cpu().numpy(), axis=-1).astype(np.uint8)
                            metric_values[m_name] += [
                                m_func(true_batch, pred_batch)
                            ]

        out_metrics = dict()
        out_metrics['loss'] = np.mean(losses).astype(np.float64)
        for m_name in metrics.keys():
            out_metrics[m_name] = np.mean(metric_values[m_name]).astype(np.float64)

        # Show information if needed
        if verbose == 1:
            string = ''
            for key, value in out_metrics.items():
                string += '{}: {:.5f} '.format(key, value)
            print(string)
        return out_metrics

    def predict(self, model, dataloader, cls_thresh=None, nms_thresh=None, save=False, save_dir=''):
        """ Function to make predictions

        :param model: Input model
        :param dataloader: Test dataloader
        :param cls_thresh: Threshold for class probability
        :param nms_thresh: Threshold for non-maximum suppression
        :param save: Flag to save results
        :param save_dir: Path to save results
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

                        tta_stack = np.asarray([
                            unpad(
                                resize(img, size=self.img_size_orig[1]),
                                img_shape=self.img_size_orig)
                            for img in tta_stack
                        ], dtype=np.float32)

                        if predictions['masks'] is None:
                            predictions['masks'] = tta_stack
                        else:
                            predictions['masks'] = np.append(predictions['masks'], tta_stack, 0)

                    # Get predictions for multiclass segmentation task
                    else:  # self.mode == 'multi'
                        outputs = model(inputs)
                        outputs = torch.softmax(outputs, dim=1)
                        out_masks = np.moveaxis(outputs.data.cpu().numpy(), 1, -1)
                        out_masks = np.asarray(
                            [unpad(img, img_shape=self.img_size_orig) for img in out_masks],
                            dtype=np.float32
                        )
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

    def postprocessing(self, pred_df, threshold=0., hole_size=None, obj_size=None, save=False, save_dir=''):
        """ Function for predicted data postprocessing

        :param pred_df: Dataset woth predicted images and masks
        :param threshold: Binarization thresholds
        :param save: Flag to save results
        :param save_dir: Save directory
        :return:
        """

        if self.task == 'segmentation':
            if self.mode == 'binary':
                masks = np.copy(pred_df['masks'].values)

                postproc_masks = [
                    postproc.delete_small_instances(
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
                    mask = np.argmax(mask, axis=-1)
                    # postproc_mask = postproc.delete_small_instances(mask,
                    #                                                 obj_size=obj_size,
                    #                                                 hole_size=hole_size)

                    postproc_masks.append(mask)

                pred_df['masks'] = postproc_masks

                # for i, row in enumerate(pred_df.iterrows()):
                #     mask_post = row[1]['masks']
                #     self._draw_images([mask_post])

                # Save result
                if save:
                    for row in pred_df.iterrows():
                        name = row[1]['names']
                        mask = row[1]['masks']
                        out_path = dirs.make_dir(f'preds/{self.task}/{self.time}', top_dir=save_dir)
                        imsave(fname=os.path.join(out_path, name), arr=mask)

        return pred_df

    def evaluate_metrics(self, true_df, pred_df, iou_thresholds=None, metric_names=None):
        """ Common function to evaluate metrics

        :param true_df:
        :param pred_df:
        :param iou_thresholds:
        :param metric_names:
        :return:
        """

        if self.task == 'detection':
            out_metrics = mean_ap(true_df=true_df, pred_df=pred_df, iou_thresholds=iou_thresholds)
        elif self.task == 'segmentation':
            if metric_names is not None:
                metrics = dict()
                for m_name in metric_names:
                    metrics[m_name] = METRICS[self.task][self.mode][m_name]
            else:
                raise ValueError(
                    f"Wrong metric_names parameter: {metric_names}."
                )

            all_df = true_df.merge(pred_df, on='names', suffixes=['_true', '_pred'])
            masks_t = np.asarray([mask for mask in all_df['masks_true'].values])
            masks_p = np.asarray([mask for mask in all_df['masks_pred'].values])

            if self.mode == 'binary':
                out_metrics = dict()
                for m_name in metric_names:
                    threshold, metric_value = get_opt_threshold(masks_t, masks_p, metric_name=m_name)
                    out_metrics[m_name] = {'threshold': threshold, 'value': metric_value}
            else:  # self.mode == 'multi'
                masks_p = np.moveaxis(masks_p, -1, 1)
                out_metrics = dict()
                for i in range(1, 11):

                    class_masks_t = (masks_t == i)
                    class_masks_p = masks_p[:, i, :, :]
                    for m_name in metric_names:
                        threshold, metric_value = get_opt_threshold(class_masks_t,
                                                                    class_masks_p,
                                                                    metric_name=m_name)
                        out_metrics[m_name] = {'threshold': threshold, 'value': metric_value}

                    print(f'Class {i - 1}: ', out_metrics)
        else:  # self.task == 'classification'
            if metric_names is not None:
                metrics = dict()
                for m_name in metric_names:
                    metrics[m_name] = METRICS[self.task][self.mode][m_name]
            else:
                raise ValueError(
                    f"Wrong metric_names parameter: {metric_names}."
                )

            if self.mode == 'binary':
                # pred_df['names'] = pred_df['names'].apply(lambda x: x[:-4])
                all_df = true_df.merge(pred_df, on='names', suffixes=['_true', '_pred'])
                labels_t = all_df['labels_true'].values.astype(np.uint8)
                labels_p = all_df['labels_pred'].values

                # print('pipeline.evaluate_classification_metrics', labels_t, labels_p)
                out_metrics = dict()
                for m_name in metric_names:
                    out_metrics[m_name] = metrics[m_name](labels_t, labels_p)
            else:  # if self.mode == 'multi'
                all_df = true_df.merge(pred_df, on='names', suffixes=['_true', '_pred'])
                labels_t = all_df['labels_true'].values.astype(np.uint8)
                labels_p = all_df['labels_pred'].values
                out_metrics = dict()
                for m_name in metric_names:
                    out_metrics[m_name] = metrics[m_name](labels_t, labels_p)

        return out_metrics

    def visualize_preds(self, preds_df, images_path):
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
                self._draw_image_boxes(image, boxes, labels, scores)

        # Get visualization for binary segmentation task
        elif self.task == 'segmentation':
            if self.mode == 'binary':
                for row in preds_df.iterrows():
                    name = row[1]['names']
                    image = np.asarray(cv2.imread(filename=os.path.join(images_path, name)), dtype=np.uint8)
                    mask = row[1]['masks']

                    msk_img = np.copy(image)
                    matching = np.all(np.expand_dims(mask, axis=-1) > 0.1, axis=-1)
                    msk_img[matching, :] = [0, 0, 0]

                    self._draw_images([image, mask])

            # Get visualization for multiclass segmentation task
            else:  # self.mode == 'multi'
                for row in preds_df.iterrows():
                    name = row[1]['names']
                    image = np.asarray(cv2.imread(filename=os.path.join(images_path, name)),
                                       dtype=np.uint8)
                    mask = row[1]['masks']
                    visual_mask = self.convert_multilabel_mask(mask=mask, how='class2rgb', n_classes=11)
                    self._draw_images([image, visual_mask])

    def get_true_labels(self, labels_path):
        """ Function to get true labels dataframe

        :param labels_path: Path to labels file
        :return:
        """
        true_df = pd.DataFrame()

        # Get true labels for detection
        if self.task == 'detection':
            true_names, true_boxes, true_labels = [], [], []
            with open(labels_path) as f:
                lines = f.readlines()
            for line in lines:
                splited = line.strip().split()
                true_names.append(splited[0])
                num_boxes = (len(splited) - 1) // 5

                box = [
                    [int(splited[1 + 5 * i]), int(splited[2 + 5 * i]), int(splited[3 + 5 * i]), int(splited[4 + 5 * i])]
                    for i in range(num_boxes)
                ]
                label = [int(splited[5 + 5 * i]) for i in range(num_boxes)]

                # box, label = [], []
                # for i in range(num_boxes):
                #     xmin = splited[1 + 5 * i]
                #     ymin = splited[2 + 5 * i]
                #     xmax = splited[3 + 5 * i]
                #     ymax = splited[4 + 5 * i]
                #     c = splited[5 + 5 * i]
                #     box.append([int(xmin), int(ymin), int(xmax), int(ymax)])
                #     label.append(int(c))
                true_boxes.append(box)
                true_labels.append(label)
            true_df['names'] = true_names
            true_df['boxes'] = true_boxes
            true_df['labels'] = true_labels

        # Get true labels for binary segmentation
        elif self.task == 'segmentation':
            if self.mode == 'binary':
                true_masks = []
                mask_names = os.listdir(os.path.join(labels_path, 'masks'))
                for mask_name in mask_names:
                    mask = cv2.imread(filename=os.path.join(labels_path, 'masks', mask_name), flags=0)
                    mask = mask / 255.
                    true_masks.append(mask)
                true_df['names'] = mask_names
                true_df['masks'] = true_masks

            # Get true labels for multiclass segmentation
            else:  # self.mode == 'multi'
                true_masks = []
                mask_names = os.listdir(os.path.join(labels_path, 'masks'))
                for mask_name in mask_names:
                    mask = cv2.imread(filename=os.path.join(labels_path, 'masks', mask_name))
                    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
                    mask = self.convert_multilabel_mask(mask, how='rgb2class', n_classes=11)
                    true_masks.append(mask)

                dates_file = os.path.join(labels_path, 'true_dates.txt')
                true_dates = []
                with open(dates_file) as f:
                    lines = f.readlines()
                for line in lines:
                    splited = line.strip().split()
                    true_dates.append(splited[1:])
                true_df['names'] = mask_names
                true_df['masks'] = true_masks
                true_df['dates'] = true_dates

        else:  # self.task == 'classification'
            true_df = pd.read_csv(filepath_or_buffer=os.path.join(labels_path, 'labels.csv'),
                                  sep=';')
                                  # header=None,
                                  # names=['names', 'labels'])
        return true_df

    @staticmethod
    def convert_multilabel_mask(mask, how='rgb2class', n_classes=2):
        """ Function for multilabel mask convertation

        :param mask:
        :param how:
        :return:
        """
        colors = allowed_parameters.SEG_MULTI_COLORS
        if how == 'rgb2class':
            out_mask = np.zeros(shape=(mask.shape[0], mask.shape[1]), dtype=np.uint8)
            for cls in range(n_classes):
                if cls == 11:
                    continue
                matching = np.all(mask == colors[cls], axis=-1)
                out_mask[matching] = cls + 1

        elif how == 'class2rgb':
            out_mask = np.zeros(shape=(mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            for cls in range(n_classes):
                if cls == 0:
                    continue
                matching = (mask[:, :] == cls)
                out_mask[matching, :] = colors[cls - 1]
        else:
            raise ValueError(
                f"Wrong parameter how: {how}. Should be 'rgb2class or class2rgb."
            )
        return out_mask

    @staticmethod
    def _draw_images(images_list):
        n_images = len(images_list)
        fig = plt.figure()
        for i, image in enumerate(images_list):
            ax = fig.add_subplot(1, n_images, i + 1)
            ax.imshow(image)
        plt.show()


    @staticmethod
    def _draw_image_boxes(image, boxes, labels, scores):
        draw = image.copy()
        print("-"*30, "\n")
        preds = [(box, label, score) for box, label, score in zip(boxes, labels, scores)]
        # preds = sorted(preds, key=lambda x: x[0][0])
        # print(preds)
        for pred in preds:
            box = pred[0]
            label = pred[1]
            score = pred[2]
            draw = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
            plt.annotate(label, xy=(box[0], box[1]), color='green')
            plt.annotate('{:.2f}'.format(score), xy=(box[0], box[3]+10), color='green')
            print(box, label, score)
        plt.imshow(draw)
        plt.show()

    @staticmethod
    def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.show()
