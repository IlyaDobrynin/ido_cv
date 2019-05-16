# -*- coding: utf-8 -*-
"""
Module implements functions for work with pytorch models.

"""
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys
import re
import shutil
import tempfile
import pickle
import hashlib
import zipfile
import json
from datetime import datetime
from requests import get as urlopen
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch.backends import cudnn
from torchsummary import summary

HASH_REGEX = re.compile(r'-([a-f0-9]*)\.')


def cuda(variable, allocate_on):
    """ Function places variable into gpu if gpu available

    :param variable: variable
    :return:
    """
    return variable.cuda() if allocate_on != 'cpu' else variable.cpu()


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


def save_model_class(model_class: nn.Module, save_path: str):
    """ Function for saving model parameters dictionary. Save it in the JSON file

    :param model_dict: Model parameters dictionary
    :param name: Model name. Should be in pretrain_model_names
    :param save_path:
    :return:
    """
    with open(save_path, "wb") as file:
        pickle.dump(model_class, file)
    return save_path


<<<<<<< HEAD
def load_model_class(model_class_path):
=======
def load_model_class(model_class_path: str):
>>>>>>> c7f28de1b0c28221b7a44d0f3f810c71718d0496
    """ Function unpickle model class

    :param model_class_path: Path to model class
    :return:
    """
    with open(model_class_path, "rb") as file:
        unpickled = pickle.loads(file.read())
    return unpickled


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


def load_seafile_url(url: str, model_dir: str = None, map_location: str = None) -> tuple:
    """ Function loads .zip archive with weights and (optionally) model meta-information
    # TODO Make not only for seafile

    :param url: URL to the model archive
    :param model_dir: Path to save archive
    :param map_location: Where to locate model
    :return:
    """
    if model_dir is None:
        torch_home = os.path.expanduser(os.getenv('TORCH_HOME', '~/.torch'))
        model_dir = os.getenv('TORCH_MODEL_ZOO', os.path.join(torch_home, 'models'))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    u = urlopen(url, stream=True)
    headers = u.headers
    if 'Content-Disposition' in headers.keys():
        filename = headers['Content-Disposition'].split('"')[-2]
        hash_prefix = HASH_REGEX.search(filename).group(1)
    else:
        raise RuntimeError(
            f'Wrong url: {url}. It have no "Content-Disposition" in keys.'
        )

    if filename[-4:] != '.zip':
        raise RuntimeError(
            f'File should be .zip archive.'
        )
    cached_file = os.path.join(model_dir, filename)

    if not os.path.exists(cached_file):
        sys.stderr.write(f'Downloading: "{url}" to {cached_file}\n')
        _download_url_to_file(url=url, hash_prefix=hash_prefix, dst=cached_file)

    zip_ref = zipfile.ZipFile(cached_file, 'r')
    zip_ref.extractall(cached_file[:-4])
    zip_ref.close()

    weights_folder = os.path.join(cached_file[:-4], 'weights')
    if os.path.exists(weights_folder):
        weights_name = os.listdir(weights_folder)
        if len(weights_name) != 1:
            raise RuntimeError(
                f'Weights folder should have only one weight. Got {len(weights_name)}.'
            )
        weights_path = os.path.join(weights_folder, weights_name[0])
    else:
        raise RuntimeError(
            f'File {weights_folder} does not exists.'
        )

    return weights_path, torch.load(weights_path, map_location=map_location)


def _download_url_to_file(url: str, hash_prefix: str, dst: str):
    """ Function downloads the file for given url into dst path if the given hash_prefix
        is equal to the file prefix

    :param url: Link to the file
    :param hash_prefix: Prefix of the file
    :param dst: Destination folder to save file
    """
    file_size = None
    u = urlopen(url, stream=True)
    if "Content-Length" in u.headers.keys():
        file_size = int(u.headers['Content-Length'])
    u = u.raw
    f = tempfile.NamedTemporaryFile(delete=False)
    try:
        if hash_prefix is not None:
            sha256 = hashlib.sha256()
        with tqdm(total=file_size) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                if hash_prefix is not None:
                    sha256.update(buffer)
                    pbar.update(len(buffer))
        f.close()
        if hash_prefix is not None:
            digest = sha256.hexdigest()
            if digest[-len(hash_prefix):] != hash_prefix:
                raise RuntimeError(
                    f'Invalid hash value '
                    f'(expected "{hash_prefix}", got "{digest[:len(hash_prefix)]}")'
                )
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)
