# -*- coding: utf-8 -*-
"""
Module implements functions for work with pytorch models.

"""
import os
import json
from datetime import datetime
import numpy as np
import torch
from torch import nn
from torch.backends import cudnn
from torchsummary import summary


def cuda(variable, allocate_on):
    """ Function places variable into gpu if gpu available

    :param variable: variable
    :return:
    """
    return variable.cuda() if allocate_on != 'cpu' else variable.cpu() # async=True


def remove_all_but_n_best(weights_dir, n_best=1, reverse=False):
    """ Remove all weights but n_best

    :param weights_dir: Path to weights to remove
    :param n_best: Amount of weights to save
    :return:
    """
    weights_names = os.listdir(weights_dir)
    weights_names.sort(key=lambda x: x[-11:-4], reverse=reverse)
    weights_to_delete = weights_names[:len(weights_names) - n_best]
    for name in weights_to_delete:
        os.remove(path=os.path.join(weights_dir, name))


def early_stop(metric, patience, mode='min'):
    """ Function to early stopping training process

    :param metric: Metric value
    :param patience: Number of epochs to patience
    :param mode: Mode. Should be 'min' or 'max'
    :return:
    """
    if mode == 'min':
        index = np.argmin(metric)
    elif mode == 'max':
        index = np.argmax(metric)
    else:
        raise ValueError('Mode should be "min" or "max", give - {}'.format(mode))
    if len(metric) - index > patience:
        return True
    return False


def allocate_model(model, device_ids=None, show_info=False, cudnn_bench=False, default_size=28):
    """ Allocate model to the given gpus

    :param model: Input model
    :param device_ids: List of the gpu indexes
    :param show_info: Flag to show model parameters
    :param cudnn_bench: List of the gpu indexes
    :param default_size: D
    :return: model
    """
    if device_ids[0] == -1:
        return model
    else:
        if torch.cuda.is_available():
            model = nn.DataParallel(model, device_ids=device_ids).cuda()
            cudnn.benchmark = cudnn_bench
        else:
            raise SystemError('GPU device not found')
        if show_info:
            input_size = (3, default_size, default_size)
            summary(model, input_size=input_size)
    return model


def save_model(model_dict, name, save_path):
    """ Function for saving model parameters dictionary. Save it in the JSON file

    :param model_dict: Model parameters dictionary
    :param name: Model name. Should be in pretrain_model_names
    :param save_path:
    :return:
    """
    json_path = os.path.join(save_path, r'{}_parameters.json'.format(name))
    with open(json_path, 'w') as file:
        json.dump(obj=model_dict, fp=file)
    return json_path


def write_event(log, step, epoch, **data):
    """ Function to write event to log file
    ToDo: Redisign logging system

    :param log: Log file
    :param step: current step
    :param epoch: Current epoch
    :param data: Additional parameters
    :return:
    """
    data['step'] = step
    data['epoch'] = epoch
    data['dt'] = datetime.now().isoformat()
    log.write(json.dumps(data, sort_keys=True))
    log.write('\n')
    log.flush()
