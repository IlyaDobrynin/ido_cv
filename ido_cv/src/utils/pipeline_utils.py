import os
import gc
from typing import List, Tuple, Dict
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from skimage.io import imsave
import warnings
from . import model_utils
from .. import allowed_parameters
from .ocr_utils import LabelConverter
from .image_utils import draw_images
from .metrics.metric_utils import get_opt_threshold

warnings.simplefilter("ignore")

TTA = allowed_parameters.TTA


# def find_learning_rate
def get_data(task: str, data: Tuple, allocate_on: str) -> Tuple:
    """ Function returns tensors for train/val processes

    :param task:        Process task:
                            - segmentation
                            - detection
                            - classification
                            - ocr
    :param data:        Tuple with data tensors
    :param allocate_on: Param shows where to locate tensors ('cpu' or 'gpu')
    :return:
    """
    inputs = model_utils.cuda(data[0], allocate_on)
    with torch.no_grad():
        if task == 'segmentation':
            targets = model_utils.cuda(data[1], allocate_on)
        elif task == 'detection':
            loc_targets = data[1]
            cls_targets = data[2]
            targets = (
                model_utils.cuda(loc_targets, allocate_on),
                model_utils.cuda(cls_targets, allocate_on)
            )
        elif task == 'ocr':
            targets_text = data[1]
            target_lengths = data[2]
            targets = (
                inputs,
                targets_text,
                model_utils.cuda(target_lengths, allocate_on)
            )
        else:  # self.task == 'classification'
            targets = model_utils.cuda(data[1], allocate_on)

    return inputs, targets


def train_one_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        criterion,
        optimizer,
        allocate_on:    str,
        epoch: int,
        step: int,
        task: str,
) -> int:
    """ Function for training one epoch

    :param model:       Model to train
    :param dataloader:  Dataloader class
    :param criterion:   Loss function to optimize
    :param optimizer:   Optimizer class
    :param allocate_on: Param shows where to locate tensors ('cpu' or 'gpu')
    :param epoch:       Epoch number
    :param step:        Step number
    :param task:        Process task:
                            - segmentation
                            - detection
                            - classification
                            - ocr
    :return:
    """
    tq = tqdm(total=(len(dataloader.dataset)))
    tq.set_description(
        'Epoch {}, lr {:.10f}'.format(epoch, optimizer.param_groups[0].get('lr'))
    )
    losses = []

    for data in dataloader:
        # Get data, located on devices
        inputs, targets = get_data(task=task, data=data, allocate_on=allocate_on)
        # Set gradients to zero
        optimizer.zero_grad()
        # Make prediction
        outputs = model(inputs)
        # Calculate loss
        loss = criterion(outputs, targets)
        losses.append(loss.item())
        loss.backward()
        # Make optimizer step
        optimizer.step()
        step += 1
        # Update progress bar
        tq.update(inputs.shape[0])
        train_loss = np.mean(losses)
        tq.set_postfix(loss='{:.5f}'.format(train_loss))
    tq.close()

    return step


def validate_train(
        criterion,
        metrics_dict: dict,
        model: nn.Module,
        dataloader: DataLoader,
        allocate_on: str,
        task: str,
        mode: str,
        verbose: int
) -> Dict:
    """ Function to make validation during training (per batch)

    :param criterion:           Loss function to optimize
    :param metrics_dict:        Names of metrics to validate
    :param model:               Model to train
    :param dataloader:          Dataloader class
    :param allocate_on:         Param shows where to locate tensors ('cpu' or 'gpu')
    :param task:                Process task:
                                    - segmentation
                                    - detection
                                    - classification
                                    - ocr
    :param mode:                Process mode
    :param verbose:             Flag to show some info
    :return:
    """
    metric_values = dict()
    for m_name in metrics_dict.keys():
        metric_values[m_name] = list()
    losses = list()
    with torch.no_grad():
        for data in dataloader:
            # Get inputs and targets
            inputs, targets = get_data(task=task, data=data, allocate_on=allocate_on)
            # Make predictions
            outputs = model(inputs)
            # Perform validation while training
            loss = criterion(outputs, targets)
            losses.append(loss.item())
            # Calculate metrics for the batch of images
            for m_name in metrics_dict.keys():
                metric_values[m_name] += [
                    metrics_dict[m_name].calculate_metric(
                        trues=targets, preds=outputs
                    )
                ]
    # Calculate mean metrics for all images
    out_metrics = dict()
    if criterion is not None:
        out_metrics['loss'] = np.mean(losses).astype(np.float64)
        for m_name in metrics_dict.keys():
            out_metrics[m_name] = np.mean(metric_values[m_name]).astype(np.float64)
        # Show information if needed
        if verbose == 1:
            string = ''
            for key, value in out_metrics.items():
                string += '{}: {:.5f} '.format(key, value)
            print(string)

    return out_metrics


def validate_test(
        metrics_dict: Dict,
        predictions: List,
        task: str,
        mode: str,
        verbose: int,
        **kwargs
) -> Dict:
    """ function to validate data after training process

    :param metrics_dict:        Names of metrics to validate
    :param predictions:         Predictions dictionary
    :param task:                Process task:
                                    - segmentation
                                    - detection
                                    - classification
                                    - ocr
    :param mode:                Process mode
    :param verbose:             Flag to show some info
    :param kwargs:
    :return:
    """
    out_metrics = dict()
    # Get metrics for detection task
    if task == 'detection':
        #ToDo: implement detection
        raise NotImplementedError(
            f"Validation for detection is not implemented"
        )

    elif task == 'segmentation':
        if metrics_dict is None:
            raise ValueError(
                f"Wrong metrics_dict parameter: {metrics_dict}."
            )
        masks_p = np.asarray([pred[2] for pred in predictions])
        masks_t = np.asarray([pred[3] for pred in predictions])

        # Get metrics for binary segmentation
        if mode == 'binary':
            for m_name in metrics_dict.keys():
                threshold, metric_value = get_opt_threshold(
                    masks_t, masks_p, metric_name=m_name
                )
                out_metrics[m_name] = {'threshold': threshold, 'value': metric_value}
            if verbose == 1:
                for k, v in out_metrics.items():
                    print(f'{k} metric: ')
                    for m_k, m_v in v.items():
                        print(f'- best {m_k}: {m_v:.5f}')

        # Get metrics for multiclass segmentation
        else:  # self.mode == 'multi'
            if kwargs['label_colors'] is None:
                raise ValueError(
                    f"Provide label_colors for multiclass segmentation task!"
                )
            for i, (class_name, class_color) in enumerate(kwargs['label_colors'].items()):
                if i == kwargs['ignore_class']:
                    continue
                class_masks_t = np.all(masks_t == class_color, axis=-1).astype(np.uint8)
                class_masks_p = masks_p[..., i + 1]
                out_metrics[class_name] = dict()
                for m_name in metrics_dict.keys():
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
    # Get metrics for ocr task
    else:  # task in ['ocr', 'classification']:
        labels_t = [pred[3] for pred in predictions]
        labels_p = [pred[2] for pred in predictions]
        for m_name in metrics_dict.keys():
            out_metrics[m_name] = metrics_dict[m_name].get_metric_value(
                labels_t, labels_p, m_name
            )
    return out_metrics


def _get_true_labels(
        data_batch: tuple,
        task: str,
        mode: str
) -> List:
    """ Function returns true labels from given batch

    :param data_batch:  Minibatch of data (images and labels)
    :param task:        Process task:
                            - segmentation
                            - detection
                            - classification
                            - ocr
    :param mode:        Process mode
    :return:
    """

    if task == 'segmentation':
        if mode == 'binary':
            true_labels = np.squeeze(
                np.moveaxis(data_batch[1].data.numpy(), 1, -1), -1
            )
        else:  # self.mode == 'multi'
            true_labels = np.moveaxis(data_batch[1].data.numpy(), 1, -1)
    elif task == 'detection':
        raise NotImplementedError(
            f"Detection task not implemented"
        )
    elif task == 'ocr':
        true_labels = []
        true_text = data_batch[1].data.numpy().tolist()
        true_lengths = data_batch[2].data.numpy().tolist()
        start_idx = 0
        for true_length in true_lengths:
            final_idx = start_idx + true_length
            true = true_text[start_idx: final_idx]
            start_idx = final_idx
            true_labels.append(true)
    else:  # self.mode == 'classification'
        # Get predictions for binary classification
        if mode == 'binary':
            true_labels = data_batch[1].data.numpy()
        else:  # self.mode == 'multi'
            true_labels = data_batch[1].data.numpy()

    return true_labels


def _get_predicts(
        task: str,
        with_labels: bool,
        names_list: list,
        images_list: np.ndarray,
        labels_pred_batch_list: list,
        labels_true_batch_list: list = None,
        **kwargs
) -> List:
    """

    :param task:                    Process task:
                                        - segmentation
                                        - detection
                                        - classification
                                        - ocr
    :param with_labels:             Flag to return true labels along with predicted labels
                                    (for validation after training)
    :param names_list:              List with images names
    :param images_list:             List with images
    :param labels_pred_batch_list:
    :param labels_true_batch_list:
    :param kwargs:
    :return:
    """
    predictions = list()

    if task == 'detection':
        # ToDo: implement detection
        raise NotImplementedError(
            f"Validation for detection is not implemented"
        )

    elif task == 'segmentation':
        labels_pred_list = np.concatenate(labels_pred_batch_list, axis=0)
        if with_labels:
            labels_true_list = np.concatenate(labels_true_batch_list, axis=0)

    elif task == 'ocr':
        labels_pred_list_ = []
        for labels_p_batch in labels_pred_batch_list:
            for i in range(len(labels_p_batch)):
                labels_pred_list_.append(labels_p_batch[i])

        alphabet_dict = kwargs['alphabet_dict']
        labels_pred_list = []
        for i, label_p in enumerate(labels_pred_list_):
            label_p_ = ''
            for char in label_p:
                for k, v in alphabet_dict.items():
                    if char == v:
                        label_p_ += k
            labels_pred_list.append(label_p_)

        if with_labels:
            labels_true_list_ = []
            for labels_t_batch in labels_true_batch_list:
                for i in range(len(labels_t_batch)):
                    labels_true_list_.append(labels_t_batch[i])
            labels_true_list = []
            for i, label_t in enumerate(labels_true_list_):
                label_t_ = ''
                for char in label_t:
                    for k, v in alphabet_dict.items():
                        if char == v:
                            label_t_ += k
                labels_true_list.append(label_t_)

    else:  # task == 'classification'
        labels_pred_list = np.concatenate(labels_pred_batch_list, axis=0)
        if with_labels:
            labels_true_list = np.concatenate(labels_true_batch_list, axis=0)

    for idx, name in enumerate(names_list):
        if with_labels:
            predictions.append([name, images_list[idx], labels_pred_list[idx], labels_true_list[idx]])
        else:
            predictions.append([name, images_list[idx], labels_pred_list[idx]])
    return predictions


def predict(
    model: nn.Module,
    dataloader: DataLoader,
    with_labels: bool,
    allocate_on: str,
    task: str,
    mode: str,
    save_batch: bool = False,
    save_dir: str = None,
    disable_tqdm: bool = False,
    **kwargs
):
    """ Function for inference process

    :param model:       Model to train
    :param dataloader:  Dataloader class
    :param with_labels: Flag to return true labels along with predicted labels
                        (for validation after training)
    :param allocate_on: Param shows where to locate tensors ('cpu' or 'gpu')
    :param task:        Process task:
                                    - segmentation
                                    - detection
                                    - classification
                                    - ocr
    :param mode:        Process mode
    :param save_batch:  Flag to save predictions after every batch
    :param save_dir:    Path to save predictions
    :param disable_tqdm:
    :param kwargs:
    :return:
    """

    names_list = []
    images_batch_list = []
    labels_pred_batch_list = []
    labels_true_batch_list = []

    model.eval()
    with torch.no_grad():
        for data_batch in tqdm(dataloader, disable=disable_tqdm):
            names = data_batch[-1]
            images = np.moveaxis(data_batch[0].data.numpy(), 1, -1)
            if not save_batch:
                for name in names:
                    names_list.append(name)
                images_batch_list.append(images)

            inputs = model_utils.cuda(data_batch[0], allocate_on)

            # Get true labels
            if with_labels:
                true_labels = _get_true_labels(
                    data_batch=data_batch,
                    task=task,
                    mode=mode
                )
                labels_true_batch_list.append([tl for tl in true_labels])

            # Get predicted labels
            if task == 'segmentation':
                if save_batch:
                    out_masks_path = os.path.join(save_dir, 'masks')
                    os.makedirs(out_masks_path, exist_ok=True)
                # Get predictions for binary segmentation
                if mode == 'binary':
                    tta_list = kwargs['tta_list']
                    tta = [TTA[task][mode][i] for i in tta_list]
                    tta_stack = []
                    for tta_class in tta:
                        tta_operation = tta_class(allocate_on=allocate_on)
                        tta_out = tta_operation(model, inputs)
                        tta_stack.append(tta_out)
                    pred_labels = np.squeeze(np.mean(tta_stack, axis=0), 1)
                    del tta_stack
                    gc.collect()

                    if save_batch:
                        for name, mask in zip(names, pred_labels):
                            imsave(os.path.join(out_masks_path, name), arr=mask)

                # Get predictions for multiclass segmentation
                else:  # self.mode == 'multi'
                    outputs = model(inputs)
                    outputs = F.softmax(outputs, dim=1)
                    pred_labels = np.moveaxis(outputs.data.cpu().numpy(), 1, -1).astype(np.float32)

                    if save_batch:
                        for name, mask in zip(names, pred_labels):
                            out_masks_folder = os.path.join(out_masks_path, name[:-4])
                            os.makedirs(out_masks_folder, exist_ok=True)
                            for ch in range(1, mask.shape[-1]):
                                mask_class = mask[..., ch]
                                imsave(
                                    os.path.join(out_masks_folder, f'{ch}_{name}'),
                                    arr=mask_class
                                )

            elif task == 'detection':
                raise NotImplementedError(
                    f"Detection task not implemented"
                )

            elif task == 'ocr':
                outputs = model(inputs)
                log_preds = F.log_softmax(outputs, dim=2)
                converter = LabelConverter(
                    alphabet=kwargs['alphabet_characters'],
                    ignore_case=False
                )
                pred_labels = converter.best_path_decode(log_preds, strings=False)
                if len(pred_labels) == 0:
                    pred_labels.append([])                
                if isinstance(pred_labels[0], int):
                    pred_labels = [pred_labels]
                if save_batch:
                    pred_labels_str = []
                    for i, label_p in enumerate(pred_labels):
                        label_p_ = ''
                        for char in label_p:
                            for k, v in kwargs['alphabet_dict'].items():
                                if char == v:
                                    label_p_ += k
                        pred_labels_str.append(label_p_)

                    with open(os.path.join(save_dir, 'labels.txt'), 'a') as f:
                        for name, label in zip(names, pred_labels_str):
                            outstr = f"{name} {label}\n"
                            f.write(outstr)

            else:  # self.mode == 'classification'
                # Get predictions for binary classification
                if mode == 'binary':
                    outputs = model(inputs)
                    outputs = F.sigmoid(outputs)
                    pred_labels = np.round(
                        np.squeeze(outputs.data.cpu().numpy(), axis=1)
                    ).astype(np.uint8)

                # Get predictions for multiclass classification
                else:  # self.mode == 'multi'
                    outputs = model(inputs)
                    outputs = F.softmax(outputs, dim=-1)
                    pred_labels = np.argmax(outputs.data.cpu().numpy(), axis=-1).astype(np.uint8)

                if save_batch:
                    with open(os.path.join(save_dir, 'labels.txt'), 'a') as f:
                        for name, label in zip(names, pred_labels):
                            outstr = f"{name} {label}\n"
                            f.write(outstr)

            if save_batch:
                del pred_labels
                gc.collect()
            else:
                labels_pred_batch_list.append([pl for pl in pred_labels])

    if not save_batch:
        images_list = np.concatenate(images_batch_list, axis=0)
        predictions = _get_predicts(
            task=task,
            mode=mode,
            with_labels=with_labels,
            names_list=names_list,
            images_list=images_list,
            labels_pred_batch_list=labels_pred_batch_list,
            labels_true_batch_list=labels_true_batch_list,
            **kwargs
        )

        return predictions
