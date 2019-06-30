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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .. import dirs
from . import allowed_parameters
from .allowed_parameters import ParameterBuilder
from .models import ModelFacade
from .utils.data import DatasetFacade
from .utils.loss import LossFacade
from .utils.metrics import MetricFacade
from .utils.configs import ConfigParser
from .utils import model_utils
from .utils.image_utils import draw_images
from .utils.image_utils import delete_small_instances
from .utils.pipeline_utils import get_data
from .utils.pipeline_utils import train_one_epoch
from .utils.pipeline_utils import validate_train
from .utils.pipeline_utils import validate_test
from .utils.pipeline_utils import predict

pd.set_option('display.max_columns', 10)

TASKS_MODES = allowed_parameters.TASKS_MODES
OPTIMIZERS = allowed_parameters.OPTIMIZERS
TTA = allowed_parameters.TTA


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
    def prediction(self, *args, **kwargs):
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
        time:               Current time.
        allocate_on:        Device(s) for locating model
        tta_list:           List of test-time augmentations.
                            Currently implemented only for binary segmentation.
        random_seed:        Random seed for fixating model random state.
        img_size_orig:      Original size of input images (
        img_size_target:    Size of input images for the model input


    """

    def __init__(
            self,
            task:           str,
            mode:           str,
            allocate_on:    str = 'cpu',
            time:           str = None,
            random_seed:    int = 1,
            **kwargs
    ):

        super(Pipeline, self).__init__()

        # Check if task is in list of existed tasks:
        # ['segmentation', 'classification', 'detection', 'ocr']
        self.tasks_modes = TASKS_MODES
        tasks = self.tasks_modes.keys()
        if task not in tasks:
            raise ValueError(
                f'Wrong task parameter: {task}. '
                f'Should be {tasks}.'
            )
        # Check if mode is in list of existed modes for this task (see allowed_parameters.py)
        modes = self.tasks_modes[task].keys()
        if mode not in modes:
            raise ValueError(
                f'Wrong mode parameter: {mode}. '
                f'Should be {modes}.'
            )

        self.task = task
        self.mode = mode

        # Make location parameter. Shows where to locate model
        if allocate_on in ['cpu', 'gpu']:
            self.allocate_on = allocate_on
        else:
            raise ValueError(
                f'Wrong allocate_on parameter: {allocate_on}. Should be "cpu" or "gpu"'
            )

        # Make ParameterBuilder object
        self.parameter_builder = ParameterBuilder(task=self.task, mode=self.mode)

        # Get metrics for training
        metric_facade = MetricFacade(task=self.task)
        metric_parameters = self.parameter_builder.get_metric_parameters
        self.val_metric_class = metric_facade.get_metric(mode=self.mode, **metric_parameters)

        self.random_seed = random_seed
        self.time = time

        if self.task == 'segmentation' and self.mode == 'multi':
            self.label_colors = kwargs['label_colors']

        if self.task == 'ocr':
            self.alphabet_characters = kwargs['alphabet_characters']
            self.alphabet_dict = {
                self.alphabet_characters[i]: i for i in range(len(self.alphabet_characters))
            }

    def get_dataloaders(
            self,
            dataset_class:      Dataset = None,
            path_to_dataset:    str = None,
            data_file:          pd.DataFrame = None,
            batch_size:         int = 1,
            is_train:           bool = True,
            workers:            int = 1,
            shuffle:            bool = False,
            common_augs              = None,
            train_time_augs          = None
    ) -> DataLoader:
        """ Function to make train data loaders

        :param dataset_class:   Dataset class
        :param path_to_dataset: Path to the images
        :param data_file:       Data file
        :param batch_size:      Size of data minibatch
        :param is_train:        Flag to specify dataloader type (train or test)
        :param workers:         Number of multithread workers
        :param shuffle:         Flag for random shuffling train dataloader samples
        :param common_augs:     Augmentations common for all images
        :param train_time_augs: Augmentations of train images to make bofore model input
        :return:
        """

        if (dataset_class is None) and (path_to_dataset is None) and (data_file is None):
            raise ValueError(
                f"At list one from dataset_class path_to_dataset or data_file parameters "
                f"should be filled!"
            )
        if (self.task == 'segmentation') \
                and (self.mode == 'multi') \
                and is_train \
                and (self.label_colors is None):
            raise ValueError(
                f"Provide label_colors for multiclass segmentation task!"
            )

        if dataset_class is None:
            # ToDo: make input parameters for dataset universal
            if self.task == 'detection':
                raise NotImplementedError(
                    f"Detection not implemented"
                )
            else:
                dataset_parameters = dict(
                    data_path=path_to_dataset,
                    data_file=data_file,
                    train=is_train,
                    common_augs=common_augs,
                    train_time_augs=train_time_augs
                )

            if self.task == 'segmentation' and self.mode == 'multi':
                dataset_parameters['label_colors'] = self.label_colors
            elif self.task == 'ocr':
                dataset_parameters['alphabet_dict'] = self.alphabet_dict

            facade_class = DatasetFacade(task=self.task, mode=self.mode)
            dataset_class = facade_class.get_dataset_class(**dataset_parameters)

        dataloader = DataLoader(
            dataset=dataset_class,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=workers,
            collate_fn=dataset_class.collate_fn
        )
        return dataloader

    def get_model(
            self,
            model_name:         str = None,
            path_to_weights:    str = None,
            save_path:          str = None,
            device_ids:         list = None,
            cudnn_bench:        bool = False,
            model_parameters:   dict = None,
            verbose:            int = 1,
            show_model:         bool = False
    ) -> tuple:
        """ Function returns model, allocated to the given gpu's

        :param model_name: Class of the model
        :param path_to_weights: Path to the trained weights
        :param save_path: Path to the trained weights
        :param device_ids: List of the gpu's
        :param cudnn_bench: Flag to include cudnn benchmark
        :param model_parameters: Path to the trained weights
        :param verbose: Flag to show info
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
                model_parameters = self.parameter_builder.get_model_parameters(model_name)
                print(
                    f'Pretrained weights are not provided. '
                    f'Train process runs with defauld {model_name} parameters: \n{model_parameters}'
                )

            # Get model class
            model_facade = ModelFacade(task=self.task, model_name=model_name)
            model = model_facade.get_model(**model_parameters)
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
                # print(model_class_path)
                model = torch.load(model_class_path)
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

        if verbose == 1:
            print(f'Train process runs with {model_name} parameters:')
            for param_name, param_val in model_parameters.items():
                print(f'{param_name}: {param_val}')
            if show_model:
                from torchsummary import summary
                summary_device = 'cpu' if self.allocate_on == 'cpu' else 'cuda'
                summary(
                    model=model,
                    input_size=(3, 256, 256),
                    device=summary_device
                )

        return model, initial_parameters, model_parameters

    def find_lr(
            self,
            model:              nn.Module,
            dataloader:         DataLoader,
            loss_name:          str,
            optim_name:         str,
            lr_reduce_factor:   int = 1,
            verbose:            int = 1,
            show_graph:         bool = True,
            init_value:         float = 1e-8,
            final_value:        float = 10.,
            beta:               float = 0.98
    ) -> float:
        """ Function find optimal learning rate for the given model and parameters

        :param model:               Input model
        :param dataloader:          Dataloader
        :param loss_name:           Loss name. Depends on given task.
        :param optim_name:          Name of the model optimizer
        :param lr_reduce_factor:    Factor to
        :param verbose:             Flag to show output information
        :param show_graph:          Flag to showing graphic
        :param init_value:          Start LR
        :param final_value:         Max stop LR
        :param beta: Smooth         value
        :return:
        """
        # Get optimizer
        optimizer_parameters = self.parameter_builder.get_optimizer_parameters(
            optimizer_name=optim_name
        )
        optimizer = OPTIMIZERS[optim_name](
            params=model.parameters(),
            lr=init_value,
            **optimizer_parameters
        )

        # Get criterion
        loss_facade = LossFacade(task=self.task, mode=self.mode, loss_name=loss_name)
        loss_parameters = self.parameter_builder.get_loss_parameters(loss_name=loss_name)
        criterion = loss_facade.get_loss(**loss_parameters)

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
            # Get data
            inputs, targets = get_data(task=self.task, data=data, allocate_on=self.allocate_on)
            # Set gradients to zero
            optimizer.zero_grad()
            # Make prediction
            outputs = model(inputs)
            # Calculate loss
            loss = criterion(outputs, targets)
            # Compute the smoothed loss
            avg_loss = beta * avg_loss + (1 - beta) * loss.data.item()
            smoothed_loss = avg_loss / (1 - beta ** batch_num)
            # Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > 4 * best_loss:
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
        opt_lr = min_lr / lr_reduce_factor

        # Show info if verbose == 1
        if verbose == 1:
            print(f'Current learning rate: {opt_lr:.10f}')
            # Show graph if necessary
            if show_graph:
                m = -5
                plt.plot(list(lr_dict.keys())[10:m], list(lr_dict.values())[10:m])
                plt.show()

        return opt_lr

    def train(
            self,
            model:          nn.Module,
            lr:             float,
            train_loader:   DataLoader,
            val_loader:     DataLoader,
            loss_name:      str,
            optim_name:     str,
            metric_names:   list,
            best_measure:   float,
            first_step:     int = 0,
            first_epoch:    int = 0,
            chp_metric:     str = 'loss',
            n_epochs:       int = 1,
            n_best:         int = 1,
            scheduler:      str = 'rop',
            patience:       int = 10,
            save_dir:       str = ''
    ) -> nn.Module:
        """ Training pipeline function

        :param model:           Input model
        :param lr:              Initial learning rate
        :param train_loader:    Train dataloader
        :param val_loader:      Validation dataloader
        :param loss_name:       Loss name. Depends on given task.
        :param optim_name:      Name of the model optimizer
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
        optimizer_parameters = self.parameter_builder.get_optimizer_parameters(
            optimizer_name=optim_name
        )
        optimizer = OPTIMIZERS[optim_name](
            params=model.parameters(),
            lr=lr,
            **optimizer_parameters
        )
        # Get criterion
        loss_facade = LossFacade(task=self.task, mode=self.mode, loss_name=loss_name)
        loss_parameters = self.parameter_builder.get_loss_parameters(loss_name=loss_name)
        criterion = loss_facade.get_loss(**loss_parameters)
        # Get learning rate scheduler policy
        # ToDo: make callbacks facade
        if scheduler == 'rop':
            mode = 'min' if chp_metric in ['loss'] else 'max'
            scheduler = ReduceLROnPlateau(
                optimizer=optimizer,
                mode=mode,
                factor=0.5,
                patience=int(patience / 2),
                verbose=True
            )
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
            # Train one epoch
            first_step = train_one_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                epoch=e,
                allocate_on=self.allocate_on,
                step=first_step,
                task=self.task
            )
            # Calculate validation metrics
            val_metrics = self.validation(
                model=model,
                dataloader=val_loader,
                metric_names=metric_names,
                criterion=criterion,
                validation_mode='train'
            )
            # Write epoch parameters to log file
            model_utils.write_event(log_file, first_step, e, **val_metrics)
            # Make scheduler step
            if scheduler:
                scheduler.step(metrics=val_metrics[chp_metric])
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
            stop_criterion = model_utils.early_stop(
                    metric=early_stop_measure, patience=patience, mode=early_stop_mode
            )
            if stop_criterion:
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

    def validation(
            self,
            model:              nn.Module,
            dataloader:         DataLoader,
            metric_names:       list,
            validation_mode:    str,
            criterion                =None,
            tta_list:           list = None,
            ignore_class:       int = None,
            verbose:            int = 1,
            **kwargs
    ) -> dict:
        """ Function to make validation of the model

        :param model:           Input model to validate
        :param dataloader:      Validation dataloader
        :param metric_names:    Metrics to print (available are 'dice', 'jaccard', 'm_iou')
        :param validation_mode: Validation mode:
                                    - 'train' - validate every batch, without threcholds calculation
                                    - 'test' - validate all data, include thresholds calculation
        :param criterion:       Function to calculate loss
        :param tta_list:
        :param ignore_class:
        :param verbose:         Flag to include output information
        :return:
        """
        assert validation_mode in ['train', 'test'], f"Wrong mode parameter: {validation_mode}." \
            f" Should be 'train' or 'test'"

        if validation_mode == 'train':
            if criterion is None:
                raise ValueError(
                    "If validation mode is 'train', criterion function should be provided!"
                )
            out_metrics = validate_train(
                model=model,
                criterion=criterion,
                dataloader=dataloader,
                allocate_on=self.allocate_on,
                val_metric_class=self.val_metric_class,
                task=self.task,
                mode=self.mode,
                metric_names=metric_names,
                verbose=verbose
            )
        else:  # validation_mode == 'test':
            # Get dict with predictions
            pred_kwargs = dict()
            if self.task == 'segmentation':
                if self.mode == 'binary':
                    if tta_list is None:
                        raise ValueError(
                            f"If validation mode is 'test', tta_list should be provided!"
                        )
                    pred_kwargs['tta_list'] = tta_list
            if self.task == 'ocr':
                pred_kwargs['alphabet_characters'] = self.alphabet_characters
                pred_kwargs['alphabet_dict'] = self.alphabet_dict

            predictions = predict(
                model=model,
                dataloader=dataloader,
                with_labels=True,
                allocate_on=self.allocate_on,
                task=self.task,
                mode=self.mode,
                **pred_kwargs
            )

            # Validate predictions
            val_kwargs = dict()
            if self.task == 'segmentation':
                if self.mode == 'multi':
                    val_kwargs['ignore_class'] = ignore_class
                    val_kwargs['label_colors'] = self.label_colors
            if self.task == 'ocr':
                val_kwargs['alphabet_dict'] = self.alphabet_dict

            out_metrics = validate_test(
                predictions=predictions,
                val_metric_class=self.val_metric_class,
                task=self.task,
                mode=self.mode,
                metric_names=metric_names,
                verbose=verbose,
                **val_kwargs
            )
        return out_metrics

    def prediction(
            self,
            model:      nn.Module,
            dataloader: DataLoader,
            tta_list:   list = None,
            save_batch: bool = False,
            with_labels: bool = False,
            save_dir:   str = ''
    ) -> dict:
        """ Function to make predictions

        :param model:       Input model
        :param dataloader:  Test dataloader
        :param save_batch:  Flag to save results
        :param save_dir:    Path to save results
        :param tta_list:    List of the test-time augmentations
        :return:
        """
        pred_kwargs = dict()
        if self.task == 'segmentation':
            if self.mode == 'binary':
                if tta_list is None:
                    raise ValueError(
                        f"If validation mode is 'test', tta_list should be provided!"
                    )
                pred_kwargs['tta_list'] = tta_list
        if self.task == 'ocr':
            pred_kwargs['alphabet_characters'] = self.alphabet_characters
            pred_kwargs['alphabet_dict'] = self.alphabet_dict

        predictions = predict(
            model=model,
            dataloader=dataloader,
            with_labels=with_labels,
            allocate_on=self.allocate_on,
            task=self.task,
            mode=self.mode,
            save_batch=save_batch,
            save_dir=save_dir,
            **pred_kwargs
        )

        return predictions

    # def postprocessing(
    #         self,
    #         predictions:    dict,
    #         threshold:      (list, float) = 0.,
    #         hole_size:      int = None,
    #         obj_size:       int = None,
    #         save:           bool = False,
    #         save_dir:       str = ''
    # ) -> pd.DataFrame:
    #     """ Function for predicted data postprocessing
    #
    #     :param pred_df:     Dataset with predicted images and masks
    #     :param threshold:   Binarization thresholds
    #     :param hole_size:   Minimum hole size to fill
    #     :param obj_size:    Minimum obj size
    #     :param save:        Flag to save results
    #     :param save_dir:    Save directory
    #     :return:
    #     """
    #
    #     if self.task == 'segmentation':
    #         if self.mode == 'binary':
    #             masks = np.copy(predictions['labels_pred'].values)
    #
    #             postproc_masks = [
    #                 delete_small_instances(
    #                     (mask > threshold).astype(np.uint8), obj_size=obj_size, hole_size=hole_size
    #                 ) for mask in masks
    #             ]
    #             predictions['labels_pred'] = postproc_masks
    #
    #         else:  # self.mode == 'multi'
    #             postproc_masks = []
    #             masks = np.copy(pred_df['masks'].values)
    #             for mask in masks:
    #                 threshold = [threshold] * mask.shape[-1] if type(threshold) == float \
    #                     else threshold
    #                 postproc_mask = np.zeros(shape=mask.shape[:2], dtype=np.uint8)
    #                 for ch in range(0, mask.shape[-1]):
    #                     if ch == 0:
    #                         threshold_0 = 0.5
    #                         mask_ch = (mask[:, :, ch] > threshold_0).astype(np.uint8)
    #                     else:
    #                         mask_ch = (mask[:, :, ch] > threshold[ch - 1]).astype(np.uint8)
    #                     mask_ch = delete_small_instances(
    #                         mask_ch,
    #                         obj_size=obj_size,
    #                         hole_size=hole_size
    #                     )
    #                     postproc_mask[mask_ch > 0] = ch
    #
    #                 postproc_masks.append(postproc_mask)
    #             pred_df['masks'] = postproc_masks
    #
    #             # Save result
    #             if save:
    #                 for row in pred_df.iterrows():
    #                     name = row[1]['names']
    #                     mask = row[1]['masks']
    #                     out_path = dirs.make_dir(f'preds/{self.task}/{self.time}', top_dir=save_dir)
    #                     imsave(fname=os.path.join(out_path, name), arr=mask)
    #
    #     return pred_df

    def visualize_preds(
            self,
            preds:       dict,
            threshold: float = 0.1,
            with_labels: bool = False
    ):
        """ Function for simple visualization

        :param preds_df:    Dataframe with predicted images
        :param images_path: Path to images
        :return:
        """
        # Get visualization for detection task
        if self.task == 'detection':
            # ToDo: refactor detection line
            raise NotImplementedError(
                f"Detection task not implemented"
            )

        elif self.task == 'segmentation':
            names = preds['names']
            images = preds['images']
            masks = preds['labels_pred']

            if with_labels:
                masks_true = preds['labels_true']

            # Get visualization for binary segmentation task
            if self.mode == 'binary':

                for i in range(len(names)): #name, image, mask in zip(names, images, masks):
                    name = names[i]
                    image = images[i]
                    mask_p = masks[i]

                    if with_labels:
                        mask_t = masks_true[i]
                    msk_img = np.copy(image)
                    matching = np.all(np.expand_dims(mask_p, axis=-1) > threshold, axis=-1)
                    msk_img[matching, :] = [0, 0, 0]

                    if with_labels:
                        draw_images([image, (mask_p > threshold).astype(np.uint8), msk_img, mask_t])
                    else:
                        draw_images([image, (mask_p > threshold).astype(np.uint8), msk_img])

            # Get visualization for multiclass segmentation task
            else:  # self.mode == 'multi'
                for name, image, mask in zip(names, images, masks):
                    draw_images([image, mask])
